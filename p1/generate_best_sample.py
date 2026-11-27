"""Print a real endpoint sample from the best completed experiment."""

import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from model import GPT, GPTConfig


RUN_DIR = PROJECT_ROOT / "runs/part1-sweep-20261120T113707/06-sgdm-selected"
CHECKPOINT = RUN_DIR / "last.pt"
DEVICE = "cuda"
PROMPT = """ROMEO:
But soft, what light through yonder window breaks?
JULIET:
"""
MAX_NEW_TOKENS = 400
TEMPERATURE = 0.8
SEED = 1234


def main():
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT}\n"
            "Run this script on the AutoDL machine that has the training checkpoints."
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; run this script on the AutoDL GPU machine.")

    config = json.loads((RUN_DIR / "config.json").read_text(encoding="utf-8"))
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    model = GPT(GPTConfig(**checkpoint["model_args"])).to(DEVICE)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    stoi = checkpoint["tokenizer"]["stoi"]
    itos = checkpoint["tokenizer"]["itos"]
    if any(char not in stoi for char in PROMPT):
        raise ValueError("The built-in prompt contains a character outside the checkpoint vocabulary.")

    torch.manual_seed(SEED)
    input_ids = torch.tensor([[stoi[char] for char in PROMPT]], device=DEVICE)
    with torch.inference_mode():
        output_ids = model.generate(input_ids, MAX_NEW_TOKENS, temperature=TEMPERATURE)[0]
    output = "".join(itos[token_id] for token_id in output_ids.tolist())

    print(f"Run: {RUN_DIR.name}")
    print(f"Optimizer: {config['optimizer_name']}")
    print(f"Learning rate: {config['sgdm_learning_rate']}")
    print(f"Checkpoint: {CHECKPOINT.name} (endpoint, not best.pt)")
    print(f"Prompt: {PROMPT!r}")
    print(f"Generated tokens: {MAX_NEW_TOKENS}")
    print("\n----- generated text -----\n")
    print(output)


if __name__ == "__main__":
    main()
