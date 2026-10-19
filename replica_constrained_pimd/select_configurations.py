#!/usr/bin/env python3

"""Select evenly spaced configurations from an extxyz trajectory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_frames(path):
    frames = []
    with Path(path).open() as handle:
        while True:
            natoms_line = handle.readline()
            if not natoms_line:
                break
            natoms = int(natoms_line)
            lines = [natoms_line, handle.readline()]
            for _ in range(natoms):
                line = handle.readline()
                if not line:
                    raise ValueError("Truncated extxyz frame")
                lines.append(line)
            frames.append(lines)
    return frames


def evenly_spaced_indices(size, count):
    if count < 1:
        raise ValueError("count must be positive")
    if count > size:
        raise ValueError("count cannot exceed the number of input frames")
    if count == 1:
        return [size // 2]
    return [round(index * (size - 1) / (count - 1)) for index in range(count)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trajectory")
    parser.add_argument("output_dir")
    parser.add_argument("--count", type=int, required=True)
    args = parser.parse_args()

    trajectory = Path(args.trajectory).resolve()
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = read_frames(trajectory)
    selected = evenly_spaced_indices(len(frames), args.count)
    anchors = []
    for anchor_index, source_index in enumerate(selected):
        filename = f"anchor_{anchor_index:04d}.extxyz"
        (output_dir / filename).write_text("".join(frames[source_index]))
        anchors.append(
            {
                "anchor_index": anchor_index,
                "source_frame_index": source_index,
                "structure": filename,
                "source_comment": frames[source_index][1].strip(),
            }
        )

    manifest = {
        "source_trajectory": str(trajectory),
        "source_frame_count": len(frames),
        "selection": "evenly_spaced",
        "count": len(anchors),
        "anchors": anchors,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(output_dir / "manifest.json")


if __name__ == "__main__":
    main()
