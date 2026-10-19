#!/usr/bin/env python3
"""Apply the paper's existing structure split to newly calculated force labels."""

import argparse
import json
from pathlib import Path

from ase.io import read, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("labels", type=Path)
    parser.add_argument("split_summary", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    structures = read(args.labels, ":")
    by_index = {int(atoms.info["configuration_index"]): atoms for atoms in structures}
    splits = json.loads(args.split_summary.read_text())["split_configuration_indices"]
    indices = [int(i) for name in ("train", "validation", "test") for i in splits[name]]
    if len(by_index) != len(structures) or len(indices) != len(set(indices)):
        raise ValueError("Duplicate configuration indices or overlapping splits")
    if set(indices) != set(by_index):
        raise ValueError("Label configurations do not match the requested splits")
    args.output.mkdir(parents=True, exist_ok=True)
    for name in ("train", "validation", "test"):
        write(
            args.output / f"{name}.extxyz",
            [by_index[int(i)] for i in splits[name]],
            format="extxyz",
        )


if __name__ == "__main__":
    main()
