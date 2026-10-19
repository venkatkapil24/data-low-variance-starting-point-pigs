#!/usr/bin/env python3

"""Run a manifest of independent fixed-bead PIMD calculations."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import subprocess
import sys
from pathlib import Path


def run_anchor(command, log_path):
    with log_path.open("a") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    return result.returncode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    parser.add_argument("output_dir")
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--nbeads", type=int, required=True)
    parser.add_argument("--cell", type=float, nargs=9)
    parser.add_argument("--equilibration-steps", type=int, default=8000)
    parser.add_argument("--production-steps", type=int, default=20000)
    parser.add_argument("--stride", type=int, default=40)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--drivers-per-run", type=int, default=2)
    parser.add_argument(
        "--thermostat", choices=("langevin", "pile_l"), default="langevin"
    )
    parser.add_argument("--thermostat-tau-fs", type=float, default=100.0)
    parser.add_argument("--seed", type=int, default=18885)
    parser.add_argument("--asset", action="append", default=[])
    parser.add_argument("--ipi-command-json", default='["i-pi"]')
    parser.add_argument("--driver-command-json", required=True)
    parser.add_argument("--pythonpath")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--record-all-bead-forces", action="store_true")
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    manifest = json.loads(manifest_path.read_text())
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    runner = Path(__file__).with_name("run_pimd.py")

    jobs = []
    for anchor in manifest["anchors"]:
        anchor_index = int(anchor["anchor_index"])
        structure = (manifest_path.parent / anchor["structure"]).resolve()
        run_dir = output_dir / f"anchor_{anchor_index:04d}"
        command = [
            sys.executable,
            str(runner),
            "--structure",
            str(structure),
            "--run-dir",
            str(run_dir),
            "--temperature",
            str(args.temperature),
            "--nbeads",
            str(args.nbeads),
            "--equilibration-steps",
            str(args.equilibration_steps),
            "--production-steps",
            str(args.production_steps),
            "--stride",
            str(args.stride),
            "--drivers",
            str(args.drivers_per_run),
            "--thermostat",
            args.thermostat,
            "--thermostat-tau-fs",
            str(args.thermostat_tau_fs),
            "--seed",
            str(args.seed + anchor_index),
            "--ipi-command-json",
            args.ipi_command_json,
            "--driver-command-json",
            args.driver_command_json,
        ]
        if args.cell:
            command.extend(["--cell", *[str(value) for value in args.cell]])
        if args.pythonpath:
            command.extend(["--pythonpath", args.pythonpath])
        if args.resume:
            command.append("--resume")
        if args.record_all_bead_forces:
            command.append("--record-all-bead-forces")
        for asset in args.asset:
            command.extend(["--asset", asset])
        jobs.append(
            (anchor_index, command, output_dir / f"anchor_{anchor_index:04d}.log")
        )

    failures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = {
            pool.submit(run_anchor, command, log_path): anchor_index
            for anchor_index, command, log_path in jobs
        }
        for future in concurrent.futures.as_completed(futures):
            anchor_index = futures[future]
            returncode = future.result()
            print(f"anchor {anchor_index:04d}: return code {returncode}", flush=True)
            if returncode != 0:
                failures.append(anchor_index)

    summary = {"count": len(jobs), "failures": failures}
    (output_dir / "batch_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
