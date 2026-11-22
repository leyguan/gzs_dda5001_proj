#!/usr/bin/env python3
"""Build the three-page Part I report PDF from a completed experiment sweep."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


A4_PORTRAIT = (8.27, 11.69)
LR_FIELDS = {
    "adamw": "learning_rate",
    "muon": "muon_learning_rate",
    "sgdm": "sgdm_learning_rate",
}


def load_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for key in ("step", "tokens", "train_loss", "val_loss"):
            row[key] = float(row[key])
    return rows


def load_runs(sweep_dir):
    runs = []
    for run_dir in sorted(Path(sweep_dir).iterdir()):
        required = ("config.json", "summary.json", "metrics.csv")
        if not run_dir.is_dir() or not all((run_dir / name).is_file() for name in required):
            continue
        config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        if summary.get("status") != "completed":
            raise ValueError(f"Incomplete run: {run_dir.name}")
        optimizer = config["optimizer_name"]
        lr_field = LR_FIELDS[optimizer]
        runs.append({
            "name": run_dir.name,
            "run_dir": run_dir,
            "config": config,
            "summary": summary,
            "rows": load_rows(run_dir / "metrics.csv"),
            "optimizer": optimizer,
            "learning_rate": float(config[lr_field]),
            "kind": "selected" if run_dir.name.endswith("-selected") else "default",
        })
    if len(runs) != 6:
        raise ValueError(f"Expected six completed runs, found {len(runs)}")
    return runs


def format_minutes(seconds):
    return f"{float(seconds) / 60.0:.1f}"


def wrap_block(title, paragraphs, width=118):
    lines = [title]
    for paragraph in paragraphs:
        lines.extend(textwrap.wrap(paragraph, width=width) or [""])
    return lines


def add_text_block(fig, lines, top, fontsize=8.0, line_height=0.0165):
    y = top
    if lines:
        fig.text(0.07, y, lines[0], fontsize=fontsize + 1.2, weight="bold", va="top")
        y -= line_height * 1.25
    for line in lines[1:]:
        fig.text(0.07, y, line, fontsize=fontsize, va="top")
        y -= line_height
    return y


def pair_analysis(runs):
    pairs = {}
    for run in runs:
        pairs.setdefault(run["optimizer"], {})[run["kind"]] = run
    paragraphs = []
    for optimizer in ("adamw", "muon", "sgdm"):
        default = pairs[optimizer]["default"]
        selected = pairs[optimizer]["selected"]
        low, high = sorted((default, selected), key=lambda run: run["learning_rate"])
        low_loss = low["summary"]["endpoint_scores"]["loss"]
        high_loss = high["summary"]["endpoint_scores"]["loss"]
        better = low if low_loss < high_loss else high
        paragraphs.append(
            f"{optimizer.upper()}: LR {low['learning_rate']:g} ended at {low_loss:.4f}; "
            f"LR {high['learning_rate']:g} ended at {high_loss:.4f}. "
            f"The lower endpoint loss was obtained at LR {better['learning_rate']:g}."
        )
    return paragraphs


def crossover_analysis(runs):
    crossings = []
    for left_index in range(len(runs)):
        for right_index in range(left_index + 1, len(runs)):
            left, right = runs[left_index], runs[right_index]
            left_values = {int(row["step"]): row["val_loss"] for row in left["rows"]}
            right_values = {int(row["step"]): row["val_loss"] for row in right["rows"]}
            steps = sorted(set(left_values) & set(right_values))
            previous_sign = 0
            for step in steps:
                difference = left_values[step] - right_values[step]
                sign = 1 if difference > 1e-12 else -1 if difference < -1e-12 else 0
                if sign and previous_sign and sign != previous_sign:
                    crossings.append((step, left, right))
                if sign:
                    previous_sign = sign
    if not crossings:
        return "No crossover was detected between the recorded validation trajectories."
    examples = "; ".join(
        f"step {step}: {left['optimizer']} LR {left['learning_rate']:g} vs "
        f"{right['optimizer']} LR {right['learning_rate']:g}"
        for step, left, right in crossings[:3]
    )
    return f"{len(crossings)} pairwise validation-curve crossings were detected. First examples: {examples}."


def draw_curves_page(pdf, runs):
    fig, axes = plt.subplots(3, 2, figsize=A4_PORTRAIT, constrained_layout=True)
    fig.suptitle("Appendix A - Training and validation loss for all six settings", fontsize=12)
    for ax, run in zip(axes.flat, runs):
        tokens = [row["tokens"] / 1_000_000 for row in run["rows"]]
        train = [row["train_loss"] for row in run["rows"]]
        val = [row["val_loss"] for row in run["rows"]]
        ax.plot(tokens, train, marker="o", markersize=2.5, label="train")
        ax.plot(tokens, val, marker="o", markersize=2.5, label="validation")
        ax.set_title(
            f"{run['optimizer']} {run['kind']}, LR={run['learning_rate']:g}",
            fontsize=8,
        )
        ax.set_xlabel("Prediction tokens (millions)", fontsize=7)
        ax.set_ylabel("Cross-entropy", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.legend(fontsize=6)
    pdf.savefig(fig)
    plt.close(fig)


def draw_sample_page(pdf, sweep_dir, runs):
    fig = plt.figure(figsize=A4_PORTRAIT)
    sample_run = next(run for run in runs if run["optimizer"] == "adamw")
    sample_path = Path(sweep_dir) / "representative-sample.txt"
    sample = sample_path.read_text(encoding="utf-8") if sample_path.is_file() else "Sample not found."
    fig.text(
        0.07,
        0.95,
        "Appendix B - Representative endpoint sample",
        fontsize=13,
        weight="bold",
    )
    fig.text(
        0.07,
        0.915,
        f"Optimizer: {sample_run['optimizer']} | Learning rate: "
        f"{sample_run['learning_rate']:g} | Checkpoint: last.pt",
        fontsize=9,
    )
    wrapped = textwrap.wrap(sample[:1200].replace("\n", " "), width=104)
    y = 0.86
    for line in wrapped[:35]:
        fig.text(0.07, y, line, fontsize=8, family="monospace", va="top")
        y -= 0.021
    pdf.savefig(fig)
    plt.close(fig)


def build_report(sweep_dir, output, author, contributions, difficulties):
    runs = load_runs(sweep_dir)
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    endpoint_winners = pair_analysis(runs)
    crossover = crossover_analysis(runs)
    principal = max(
        runs,
        key=lambda run: run["summary"]["train_seconds"],
    )
    difficulties = list(difficulties) or [
        "The required fixed token budget repeatedly traverses a small character corpus with "
        "a relatively large GPT-2-medium model. Training loss continued to fall while "
        "validation loss rose for some settings, so overfitting was analyzed using the "
        "actual train/validation curves and endpoint results rather than hidden early stopping.",
        f"The longest run was {principal['optimizer']} at LR "
        f"{principal['learning_rate']:g} ({format_minutes(principal['summary']['train_seconds'])} min). "
        "Identical model, data, scheduler and token budgets were retained for every run.",
    ]
    contributions = list(contributions) or [
        f"{author}: completed the required implementations, ran the six experiments, "
        "verified the exported artifacts and prepared this report."
    ]

    with PdfPages(output) as pdf:
        fig = plt.figure(figsize=A4_PORTRAIT)
        fig.text(0.07, 0.965, "DDA5001 Part I - Pre-training a Small GPT", fontsize=16, weight="bold")
        fig.text(
            0.07,
            0.94,
            "AdamW, Muon and SGDM learning-rate comparison on Shakespeare",
            fontsize=9.5,
        )
        fig.text(0.07, 0.912, f"Author: {author}", fontsize=8.5)
        method = (
            "All six runs start from random weights and use the same GPT-2-medium dimensions, "
            "character data, cosine schedule, 92,160,000-token budget, evaluation windows and seed. "
            "Only optimizer_name, the corresponding learning-rate field and output directory differ."
        )
        lines = wrap_block("Method and controls", [method])
        add_text_block(fig, lines, 0.885)

        table_ax = fig.add_axes((0.07, 0.59, 0.86, 0.245))
        table_ax.axis("off")
        table_rows = [
            [
                f"{run['optimizer']} {run['kind']}",
                f"{run['learning_rate']:g}",
                f"{run['summary']['endpoint_scores']['loss']:.4f}",
                f"{run['summary']['best_val_loss']:.4f}",
                format_minutes(run["summary"]["train_seconds"]),
            ]
            for run in runs
        ]
        table = table_ax.table(
            cellText=table_rows,
            colLabels=("Setting", "LR", "Endpoint val loss", "Best val loss", "Train min"),
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(7)
        table.scale(1, 1.18)
        for cell in table.get_celld().values():
            cell.set_linewidth(0.4)

        lines = wrap_block("Learning-rate effects", endpoint_winners + [crossover])
        add_text_block(fig, lines, 0.57, fontsize=7.3, line_height=0.014)
        lines = wrap_block("Main difficulties and solutions", difficulties)
        add_text_block(fig, lines, 0.355, fontsize=7.3, line_height=0.014)
        lines = wrap_block("Contributions", contributions)
        y = add_text_block(fig, lines, 0.19, fontsize=7.3, line_height=0.014)
        fig.text(
            0.07,
            max(0.04, y - 0.01),
            "Endpoint metrics use last.pt. Appendix A shows train/validation curves; "
            "Appendix B shows a sampled endpoint generation.",
            fontsize=6.5,
        )
        pdf.savefig(fig)
        plt.close(fig)

        draw_curves_page(pdf, runs)
        draw_sample_page(pdf, sweep_dir, runs)
    return output


def parse_args():
    parser = argparse.ArgumentParser(description="Generate the Part I report PDF.")
    parser.add_argument("--runs-dir", required=True)
    parser.add_argument("--output", default="")
    parser.add_argument("--author", default="Leyan Guan")
    parser.add_argument("--contribution", action="append", default=[])
    parser.add_argument("--difficulty", action="append", default=[])
    args = parser.parse_args()
    if not args.output:
        args.output = str(Path(args.runs_dir) / "PartI_Report.pdf")
    return args


def main():
    args = parse_args()
    output = build_report(
        Path(args.runs_dir),
        Path(args.output),
        args.author,
        args.contribution,
        args.difficulty,
    )
    print(output)


if __name__ == "__main__":
    main()
