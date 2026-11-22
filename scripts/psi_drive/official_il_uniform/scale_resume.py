"""Apply the recorded batch-128 to batch-256 LR change to the step 1330 resume."""
import argparse
from pathlib import Path

import torch

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists() or args.source.resolve() == args.output.resolve():
        parser.error("use a new output checkpoint path")
    checkpoint = torch.load(args.source, map_location="cpu", weights_only=False)
    if checkpoint["global_step"] != 1330 or checkpoint["epoch"] != 1:
        raise ValueError("Expected the two completed batch-128 epochs at step 1330")
    (optimizer,) = checkpoint["optimizer_states"]
    (scheduler,) = checkpoint["lr_schedulers"]
    if (scheduler["lr"], scheduler["last_epoch"], scheduler["epochs"], scheduler["warmup_epochs"]) != (1e-4, 2, 200, 3):
        raise ValueError("Unexpected warm-start scheduler state")
    if not optimizer["state"] or not checkpoint["MixedPrecision"]:
        raise ValueError("Optimizer and mixed-precision state are required")
    for group in optimizer["param_groups"]:
        if group["lr"] != 1e-4:
            raise ValueError("Unexpected warm-start optimizer learning rate")
        group["lr"] *= 2
        if "initial_lr" in group:
            group["initial_lr"] *= 2
    for key in ("lr", "min_lr"):
        scheduler[key] *= 2
    for key in ("base_lrs", "_last_lr"):
        scheduler[key] = [lr * 2 for lr in scheduler[key]]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(args.output)

if __name__ == "__main__":
    main()
