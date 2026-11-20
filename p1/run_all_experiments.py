#!/usr/bin/env python3
"""Run the six Part I optimizer experiments with fail-fast metadata recording.

This runner is intentionally separate from the training scaffold. It creates one
fresh config copy per run, changes only optimizer_name, that optimizer's LR field,
and out_dir, then invokes the documented train.py interface.
"""
from __future__ import annotations

import argparse
import ast
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import shlex
import statistics
import subprocess
import sys
import threading
import traceback

from checkpoint_io import RunInterrupted, run_monitored


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_ROOT / "config/train_shakespeare_char.py"
OPTIMIZERS = (
    ("adamw", "learning_rate"),
    ("muon", "muon_learning_rate"),
    ("sgdm", "sgdm_learning_rate"),
)
REQUIRED_TRAIN_FILES = (
    "config.json",
    "environment.json",
    "metrics.csv",
    "summary.json",
    "run.log",
    "last.pt",
)
LR_POLICY = {
    "version": 1,
    "minimum_metrics_rows": 3,
    "rise_threshold_rule": "max(0.02, 0.02 * abs(first_validation_loss))",
    "unstable_rise_fraction": 0.30,
    "ending_above_best_threshold": "max(0.05, 0.02 * abs(first_validation_loss))",
    "slow_relative_improvement_below": 0.10,
    "factors": {
        "unstable_or_ending_above_best": 0.5,
        "slow_and_smooth": 2.0,
        "stable_default": 1.5,
    },
    "rationale": (
        "Use the default run's validation curve. Halve the LR when losses rise "
        "meaningfully or finish materially above their best value; double it only "
        "when improvement is slow and the curve is smooth; otherwise probe a "
        "moderately higher LR. The same rule is applied to all three optimizers."
    ),
}


class SweepError(RuntimeError):
    """A configuration, training, or verification failure that must stop the sweep."""


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_text(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def read_literal_config(path):
    """Read the simple top-level literal assignments used by course configs."""
    values = {}
    tree = ast.parse(Path(path).read_text(encoding="utf-8"), filename=str(path))
    for statement in tree.body:
        if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
            continue
        target = statement.targets[0]
        if not isinstance(target, ast.Name):
            continue
        try:
            values[target.id] = ast.literal_eval(statement.value)
        except (ValueError, TypeError):
            continue
    return values


def shell_command(command):
    return shlex.join(str(part) for part in command)


def tee_log_to_terminal(path, stop_event):
    """Follow a subprocess log file and echo new output without replacing it."""
    with Path(path).open("r", encoding="utf-8", errors="replace") as stream:
        while True:
            line = stream.readline()
            if line:
                print(line, end="", flush=True)
                continue
            if stop_event.is_set():
                break
            stop_event.wait(0.10)


class SweepRunner:
    def __init__(self, base_config, output_root):
        self.base_config = Path(base_config).resolve()
        self.output_root = Path(output_root).resolve()
        self.base_text = self.base_config.read_text(encoding="utf-8")
        self.base_values = read_literal_config(self.base_config)
        self.records_by_name = {}
        self.current_record = None
        self.metadata_path = self.output_root / "sweep.json"
        self.commands_path = self.output_root / "commands.log"
        self.metadata = {
            "schema_version": 1,
            "status": "initializing",
            "created_at_utc": utc_now(),
            "script": {
                "path": str(Path(__file__).resolve()),
                "argv": [str(value) for value in sys.argv],
                "command": shell_command([sys.executable, *sys.argv]),
                "sha256": sha256(Path(__file__).resolve()),
            },
            "project_root": str(PROJECT_ROOT),
            "output_root": str(self.output_root),
            "base_config": {
                "path": str(self.base_config),
                "sha256": sha256(self.base_config),
                "values": self.base_values,
            },
            "expected_run_count": 6,
            "lr_selection_policy": LR_POLICY,
            "preflight": [],
            "runs": [],
            "postprocessing": [],
            "failure": None,
        }

    def save(self):
        # run_records.load_runs() uses this exact compatibility field for plot.py.
        self.metadata["trials"] = self.metadata["runs"]
        atomic_json(self.metadata_path, self.metadata)

    def validate_inputs(self):
        if not self.base_config.is_file():
            raise SweepError(f"Base config does not exist: {self.base_config}")
        required = {
            "optimizer_name",
            "learning_rate",
            "muon_learning_rate",
            "sgdm_learning_rate",
            "max_tokens",
            "max_seconds",
        }
        missing = sorted(required - set(self.base_values))
        if missing:
            raise SweepError(f"Base config is missing required settings: {missing}")
        for _, field in OPTIMIZERS:
            value = self.base_values[field]
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise SweepError(f"{field} must be a positive finite number")
        if self.base_values["optimizer_name"] not in {name for name, _ in OPTIMIZERS}:
            raise SweepError("Base optimizer_name is not adamw, muon, or sgdm")
        for filename in ("data/shakespeare_char/train.bin", "data/shakespeare_char/val.bin",
                         "data/shakespeare_char/meta.pkl"):
            if not (PROJECT_ROOT / filename).is_file():
                raise SweepError(f"Required dataset file is missing: {filename}")

    def prepare(self):
        self.validate_inputs()
        try:
            self.output_root.mkdir(parents=True, exist_ok=False)
        except FileExistsError as exc:
            raise SweepError(f"Output directory already exists; choose a fresh one: {self.output_root}") from exc
        (self.output_root / "configs").mkdir()
        (self.output_root / "logs").mkdir()
        self.metadata["status"] = "running"
        self.save()

    def environment_snapshot(self):
        try:
            import torch
        except ImportError as exc:
            raise SweepError("PyTorch is not importable in this Python environment") from exc
        if not hasattr(torch.optim, "Muon"):
            raise SweepError("Native torch.optim.Muon is unavailable; PyTorch >=2.9 is required")
        if not torch.cuda.is_available():
            raise SweepError(
                "CUDA is unavailable. Run the full six-experiment sweep in the supplied "
                "CUDA environment or classroom workstation."
            )
        devices = []
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            devices.append({
                "index": index,
                "name": properties.name,
                "total_memory_bytes": properties.total_memory,
                "capability": list(torch.cuda.get_device_capability(index)),
            })
        git_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        git_status = subprocess.run(
            ["git", "status", "--short"],
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.metadata["environment"] = {
            "python": sys.version,
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": True,
            "cuda_devices": devices,
            "git_commit": git_commit.stdout.strip() if git_commit.returncode == 0 else None,
            "git_status_short": git_status.stdout.splitlines() if git_status.returncode == 0 else None,
            "source_sha256": {
                path.name: sha256(path)
                for path in sorted(PROJECT_ROOT.glob("*.py"))
            },
        }
        self.save()

    def append_command(self, command, log_path):
        line = (
            f"[{utc_now()}] cwd={PROJECT_ROOT}\n"
            f"{shell_command(command)}\n"
            f"log={log_path}\n"
        )
        with self.commands_path.open("a", encoding="utf-8") as stream:
            stream.write(line)
            stream.flush()
            os.fsync(stream.fileno())

    def execute(self, command, log_path, extra_env=None):
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"
        if extra_env:
            env.update(extra_env)
        self.append_command(command, log_path)
        print(f"Running: {shell_command(command)}", flush=True)
        print(f"Live output: {log_path}  (saved and echoed to this terminal)", flush=True)
        with log_path.open("w", encoding="utf-8") as stream:
            stop_event = threading.Event()
            echo_thread = threading.Thread(
                target=tee_log_to_terminal,
                args=(log_path, stop_event),
                name=f"tee-{log_path.name}",
                daemon=True,
            )
            echo_thread.start()
            try:
                return run_monitored(
                    [str(part) for part in command],
                    cwd=PROJECT_ROOT,
                    env=env,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                )
            finally:
                stop_event.set()
                echo_thread.join(timeout=2)

    def run_check(self, name, command, log_path, extra_env=None):
        record = {
            "name": name,
            "command": [str(part) for part in command],
            "shell_command": shell_command(command),
            "log": str(log_path),
            "started_at_utc": utc_now(),
            "status": "running",
        }
        self.metadata["preflight"].append(record)
        self.save()
        try:
            result = self.execute(command, log_path, extra_env)
        except BaseException as exc:
            record.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                          finished_at_utc=utc_now(), error=f"{type(exc).__name__}: {exc}")
            self.save()
            raise
        record.update(returncode=result.returncode, finished_at_utc=utc_now())
        if result.returncode != 0:
            record["status"] = "failed"
            self.save()
            raise SweepError(f"{name} failed with exit code {result.returncode}; see {log_path}")
        record["status"] = "completed"
        self.save()

    def preflight(self):
        self.environment_snapshot()
        (self.output_root / "preflight/matplotlib").mkdir(parents=True)
        self.run_check(
            "check_solution",
            [sys.executable, "check_solution.py"],
            self.output_root / "logs/check_solution.log",
            {"MPLCONFIGDIR": str(self.output_root / "preflight/matplotlib")},
        )
        smoke_dir = self.output_root / "preflight/cpu-smoke"
        self.run_check(
            "cpu_smoke",
            [
                sys.executable,
                "train.py",
                "config/smoke.py",
                "--device=cpu",
                f"--out_dir={smoke_dir}",
            ],
            self.output_root / "logs/cpu_smoke.log",
        )
        summary_path = smoke_dir / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("status") != "completed":
            raise SweepError(f"CPU smoke run did not complete: {summary_path}")

    def make_config(self, name, optimizer, lr_field, learning_rate):
        config_path = self.output_root / "configs" / f"{name}.py"
        run_dir = self.output_root / name
        changes = {
            "optimizer_name": optimizer,
            lr_field: learning_rate,
            "out_dir": str(run_dir),
        }
        text = self.base_text
        for key, value in changes.items():
            pattern = re.compile(rf"(?m)^{re.escape(key)}\s*=.*$")
            if len(pattern.findall(text)) != 1:
                raise SweepError(f"Base config must contain exactly one top-level {key} assignment")
            text = pattern.sub(f"{key} = {value!r}", text, count=1)
        parsed = read_literal_config_from_text(text, str(config_path))
        for key, value in changes.items():
            if parsed.get(key) != value:
                raise SweepError(f"Generated config did not preserve the requested {key}")
        atomic_text(config_path, text)
        return config_path, run_dir

    def make_record(self, name, kind, optimizer, lr_field, learning_rate, config_path, run_dir,
                    lr_selection=None):
        record = {
            "name": name,
            "kind": kind,
            "status": "planned",
            "optimizer": optimizer,
            "optimizer_name": optimizer,
            "lr_field": lr_field,
            "learning_rate": learning_rate,
            "lr_selection": lr_selection,
            "config": str(config_path.resolve()),
            "config_sha256": sha256(config_path),
            "run_dir": str(run_dir.resolve()),
            "protocol": (
                "tokens" if self.base_values["max_tokens"] > 0
                and self.base_values["max_seconds"] == 0
                else "time"
            ),
            "returncode": None,
            "command": None,
            "shell_command": None,
            "log": str((self.output_root / "logs" / f"{name}.log").resolve()),
            "analysis_commands": [],
        }
        self.records_by_name[name] = record
        self.metadata["runs"].append(record)
        self.save()
        return record

    def run_training(self, record):
        self.current_record = record
        command = [sys.executable, "train.py", record["config"]]
        record.update(
            status="running",
            command=command,
            shell_command=shell_command(command),
            started_at_utc=utc_now(),
        )
        self.save()
        started = datetime.now(timezone.utc)
        try:
            result = self.execute(command, Path(record["log"]))
        except BaseException as exc:
            record.update(
                status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                finished_at_utc=utc_now(),
                error=f"{type(exc).__name__}: {exc}",
            )
            self.save()
            raise
        record.update(
            returncode=result.returncode,
            finished_at_utc=utc_now(),
            duration_seconds=(datetime.now(timezone.utc) - started).total_seconds(),
        )
        if result.returncode != 0:
            record["status"] = "failed"
            self.save()
            raise SweepError(
                f"{record['name']} failed with exit code {result.returncode}; see {record['log']}"
            )
        missing = [
            filename for filename in REQUIRED_TRAIN_FILES
            if not (Path(record["run_dir"]) / filename).is_file()
        ]
        if missing:
            record["status"] = "failed"
            record["error"] = f"Missing required output files: {missing}"
            self.save()
            raise SweepError(record["error"])
        summary = json.loads(
            (Path(record["run_dir"]) / "summary.json").read_text(encoding="utf-8")
        )
        if summary.get("status") != "completed":
            record["status"] = "failed"
            record["error"] = f"Training summary status is {summary.get('status')}"
            self.save()
            raise SweepError(record["error"])
        record.update(
            status="completed",
            returncode=0,
            summary={
                "step": summary["step"],
                "tokens": summary["tokens"],
                "train_seconds": summary["train_seconds"],
                "wall_seconds": summary["wall_seconds"],
                "endpoint_scores": summary["endpoint_scores"],
                "best_val_loss": summary["best_val_loss"],
                "budget": summary["budget"],
            },
        )
        self.current_record = None
        self.save()

    def select_alternative_lr(self, default_record):
        metrics_path = Path(default_record["run_dir"]) / "metrics.csv"
        with metrics_path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        values = []
        for row in rows:
            try:
                value = float(row["val_loss"])
            except (KeyError, TypeError, ValueError) as exc:
                raise SweepError(f"Invalid val_loss row in {metrics_path}") from exc
            if not math.isfinite(value):
                raise SweepError(f"Nonfinite val_loss in {metrics_path}")
            values.append(value)
        if len(values) < LR_POLICY["minimum_metrics_rows"]:
            raise SweepError(
                f"{metrics_path} has {len(values)} validation points; "
                f"need at least {LR_POLICY['minimum_metrics_rows']}"
            )
        first, final, best = values[0], values[-1], min(values)
        changes = [values[index + 1] - values[index] for index in range(len(values) - 1)]
        rise_threshold = max(0.02, 0.02 * abs(first))
        ending_threshold = max(0.05, 0.02 * abs(first))
        rise_fraction = sum(change > rise_threshold for change in changes) / len(changes)
        ending_above_best = final - best > ending_threshold
        unstable = rise_fraction >= LR_POLICY["unstable_rise_fraction"] or ending_above_best
        relative_improvement = (first - best) / max(abs(first), 1e-12)
        slow = not unstable and relative_improvement < LR_POLICY["slow_relative_improvement_below"]
        if unstable:
            factor = LR_POLICY["factors"]["unstable_or_ending_above_best"]
            classification = "unstable_or_ending_above_best"
            rationale = (
                "The default curve rose often or ended materially above its best value, "
                "so the comparison run uses half the default LR."
            )
        elif slow:
            factor = LR_POLICY["factors"]["slow_and_smooth"]
            classification = "slow_and_smooth"
            rationale = (
                "The default curve improved by less than 10% from its first value without "
                "material instability, so the comparison run doubles the LR."
            )
        else:
            factor = LR_POLICY["factors"]["stable_default"]
            classification = "stable_default"
            rationale = (
                "The default curve improved materially without meeting the instability "
                "rule, so the comparison run probes a moderately higher 1.5x LR."
            )
        alternative = float(f"{default_record['learning_rate'] * factor:.12g}")
        return {
            "mode": "adaptive_after_default",
            "classification": classification,
            "rationale": rationale,
            "default_lr": default_record["learning_rate"],
            "factor": factor,
            "selected_lr": alternative,
            "source_metrics": str(metrics_path),
            "evidence": {
                "validation_points": len(values),
                "first_val_loss": first,
                "final_val_loss": final,
                "best_val_loss": best,
                "absolute_improvement_to_best": first - best,
                "relative_improvement_to_best": relative_improvement,
                "rise_threshold": rise_threshold,
                "rise_fraction": rise_fraction,
                "ending_above_best_threshold": ending_threshold,
                "ending_above_best": ending_above_best,
                "step_change_volatility": statistics.pstdev(changes),
            },
        }

    def run_defaults(self):
        for index, (optimizer, lr_field) in enumerate(OPTIMIZERS, 1):
            name = f"{index:02d}-{optimizer}-default"
            lr = self.base_values[lr_field]
            config_path, run_dir = self.make_config(name, optimizer, lr_field, lr)
            record = self.make_record(
                name,
                "default",
                optimizer,
                lr_field,
                lr,
                config_path,
                run_dir,
                {
                    "mode": "provided_default",
                    "rationale": "This is the provided default learning rate for this optimizer.",
                },
            )
            self.run_training(record)

    def run_alternatives(self):
        default_records = list(self.records_by_name.values())
        for index, default_record in enumerate(default_records, 4):
            optimizer = default_record["optimizer"]
            lr_field = default_record["lr_field"]
            selection = self.select_alternative_lr(default_record)
            name = f"{index:02d}-{optimizer}-selected"
            config_path, run_dir = self.make_config(
                name, optimizer, lr_field, selection["selected_lr"]
            )
            record = self.make_record(
                name,
                "selected",
                optimizer,
                lr_field,
                selection["selected_lr"],
                config_path,
                run_dir,
                selection,
            )
            self.run_training(record)

    def postprocess(self):
        self.metadata["status"] = "postprocessing"
        self.save()
        for record in self.metadata["runs"]:
            command = [
                sys.executable,
                "plot.py",
                f"--run-dir={record['run_dir']}",
            ]
            log_path = self.output_root / "logs" / f"{record['name']}-plot.log"
            record["analysis_commands"].append({
                "command": command,
                "shell_command": shell_command(command),
                "log": str(log_path),
            })
            self.save()
            result = self.execute(command, log_path)
            record["analysis_commands"][-1]["returncode"] = result.returncode
            if result.returncode != 0:
                raise SweepError(f"Plotting failed for {record['name']}; see {log_path}")
            self.save()

        # plot.py reads sweep.json through run_records.load_runs, which requires completed.
        self.metadata["status"] = "completed"
        self.save()
        compare_command = [
            sys.executable,
            "plot.py",
            f"--runs-dir={self.output_root}",
        ]
        compare_log = self.output_root / "logs/comparison-plot.log"
        self.metadata["postprocessing"].append({
            "name": "comparison_plot",
            "command": compare_command,
            "shell_command": shell_command(compare_command),
            "log": str(compare_log),
        })
        self.save()
        result = self.execute(compare_command, compare_log)
        self.metadata["postprocessing"][-1]["returncode"] = result.returncode
        if result.returncode != 0:
            raise SweepError(f"Comparison plotting failed; see {compare_log}")

        sample_record = self.records_by_name["01-adamw-default"]
        sample_output = self.output_root / "representative-sample.txt"
        sample_command = [
            sys.executable,
            "sample.py",
            f"--checkpoint={Path(sample_record['run_dir']) / 'last.pt'}",
            "--device=cuda",
            "--tokens=400",
            f"--output={sample_output}",
        ]
        sample_log = self.output_root / "logs/sample.log"
        self.metadata["postprocessing"].append({
            "name": "representative_sample",
            "optimizer": "adamw",
            "learning_rate": sample_record["learning_rate"],
            "command": sample_command,
            "shell_command": shell_command(sample_command),
            "log": str(sample_log),
            "output": str(sample_output),
        })
        self.save()
        result = self.execute(sample_command, sample_log)
        self.metadata["postprocessing"][-1]["returncode"] = result.returncode
        if result.returncode != 0:
            raise SweepError(f"Sample generation failed; see {sample_log}")
        self.write_summary()
        self.metadata["status"] = "completed"
        self.metadata["finished_at_utc"] = utc_now()
        self.save()

    def write_summary(self):
        fields = (
            "name",
            "optimizer",
            "learning_rate",
            "lr_field",
            "endpoint_val_loss",
            "best_val_loss",
            "training_seconds",
            "wall_seconds",
            "tokens",
            "config",
            "run_dir",
            "command",
        )
        summary_path = self.output_root / "sweep-summary.csv"
        temporary = summary_path.with_name(summary_path.name + ".tmp")
        with temporary.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for record in self.metadata["runs"]:
                summary = record["summary"]
                writer.writerow({
                    "name": record["name"],
                    "optimizer": record["optimizer"],
                    "learning_rate": record["learning_rate"],
                    "lr_field": record["lr_field"],
                    "endpoint_val_loss": summary["endpoint_scores"]["loss"],
                    "best_val_loss": summary["best_val_loss"],
                    "training_seconds": summary["train_seconds"],
                    "wall_seconds": summary["wall_seconds"],
                    "tokens": summary["tokens"],
                    "config": record["config"],
                    "run_dir": record["run_dir"],
                    "command": record["shell_command"],
                })
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, summary_path)
        self.metadata["summary_csv"] = str(summary_path)

    def mark_failure(self, exc):
        interrupted = isinstance(exc, (KeyboardInterrupt, RunInterrupted))
        self.metadata["status"] = "interrupted" if interrupted else "failed"
        self.metadata["failure"] = {
            "time_utc": utc_now(),
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        if self.current_record is not None:
            self.current_record["status"] = "interrupted" if interrupted else "failed"
            self.current_record.setdefault("error", f"{type(exc).__name__}: {exc}")
        self.metadata["finished_at_utc"] = utc_now()
        self.save()


def read_literal_config_from_text(text, filename):
    values = {}
    tree = ast.parse(text, filename=filename)
    for statement in tree.body:
        if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
            continue
        target = statement.targets[0]
        if not isinstance(target, ast.Name):
            continue
        try:
            values[target.id] = ast.literal_eval(statement.value)
        except (ValueError, TypeError):
            continue
    return values


def default_output_root():
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    return PROJECT_ROOT / "runs" / f"part1-sweep-{stamp}"


def dry_run(base_config, output_root):
    base_config = Path(base_config).resolve()
    values = read_literal_config(base_config)
    missing = [field for _, field in OPTIMIZERS if field not in values]
    if missing:
        raise SweepError(f"Base config is missing learning-rate fields: {missing}")
    plan = {
        "status": "dry-run",
        "base_config": str(base_config),
        "output_root": str(Path(output_root).resolve()),
        "default_runs": [
            {
                "order": index,
                "optimizer": optimizer,
                "lr_field": field,
                "learning_rate": values[field],
            }
            for index, (optimizer, field) in enumerate(OPTIMIZERS, 1)
        ],
        "adaptive_selected_runs": [
            {
                "order": index,
                "optimizer": optimizer,
                "lr_field": field,
                "selection": "derived from the corresponding default validation curve",
            }
            for index, (optimizer, field) in enumerate(OPTIMIZERS, 4)
        ],
        "lr_selection_policy": LR_POLICY,
        "output_layout": {
            "configs": "<output_root>/configs/*.py",
            "runs": "<output_root>/NN-optimizer-{default,selected}/",
            "metadata": "<output_root>/sweep.json",
            "commands": "<output_root>/commands.log",
            "individual_plots": "<output_root>/NN-optimizer-{default,selected}/loss_vs_*.png",
            "comparison_plots": "<output_root>/comparison_*.png",
            "sample": "<output_root>/representative-sample.txt",
        },
    }
    print(json.dumps(plan, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run six fail-fast Part I experiments and save complete metadata."
    )
    parser.add_argument(
        "--base-config",
        default=str(DEFAULT_CONFIG),
        help="Config copied six times; only optimizer, its LR field, and out_dir are changed.",
    )
    parser.add_argument(
        "--output-root",
        default=str(default_output_root()),
        help="New sweep directory; defaults to runs/part1-sweep-<timestamp>.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the six-run plan and LR policy without creating files or using CUDA.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        if args.dry_run:
            dry_run(args.base_config, args.output_root)
            return 0
        runner = SweepRunner(args.base_config, args.output_root)
        try:
            runner.prepare()
            runner.preflight()
            runner.run_defaults()
            runner.run_alternatives()
            runner.postprocess()
        except BaseException as exc:
            runner.mark_failure(exc)
            if isinstance(exc, (KeyboardInterrupt, RunInterrupted)):
                print(f"Interrupted; metadata saved to {runner.metadata_path}", file=sys.stderr)
                return 128 + getattr(exc, "signum", 2)
            print(f"ERROR: {exc}", file=sys.stderr)
            print(f"Metadata saved to {runner.metadata_path}", file=sys.stderr)
            return 1
        print(json.dumps({
            "status": "completed",
            "output_root": str(runner.output_root),
            "metadata": str(runner.metadata_path),
            "commands": str(runner.commands_path),
            "summary": runner.metadata["summary_csv"],
        }, indent=2))
        return 0
    except (SweepError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
