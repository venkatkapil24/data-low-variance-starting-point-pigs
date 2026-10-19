#!/usr/bin/env python3
"""Train a force-only MACE potential using the paper's model configuration."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--test", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--name", default="ridge_scaled_grad")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).with_name("mace.yaml")
    )
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    # Paths are resolved before changing the working directory so that this
    # command behaves the same when launched from any folder.
    config.update(
        name=args.name,
        device=args.device,
        max_num_epochs=args.epochs,
        train_file=str(args.train.resolve()),
        valid_file=str(args.validation.resolve()),
        work_dir=str(args.output.resolve()),
    )
    if args.test:
        config["test_file"] = str(args.test.resolve())
    for field, folder in (
        ("log_dir", "logs"),
        ("model_dir", "models"),
        ("checkpoints_dir", "checkpoints"),
        ("results_dir", "results"),
    ):
        config[field] = str((args.output / folder).resolve())
        Path(config[field]).mkdir(parents=True, exist_ok=True)
    args.output.mkdir(parents=True, exist_ok=True)
    effective_config = args.output.resolve() / "training.yaml"
    effective_config.write_text(yaml.safe_dump(config, sort_keys=False))
    command = [
        sys.executable,
        "-m",
        "mace.cli.run_train",
        "--config",
        str(effective_config),
    ]
    (args.output / "command.json").write_text(json.dumps(command, indent=2) + "\n")
    subprocess.run(command, cwd=args.output, check=True)


if __name__ == "__main__":
    main()
