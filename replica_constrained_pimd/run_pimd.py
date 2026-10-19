#!/usr/bin/env python3

"""Run one restartable fixed-bead PIMD calculation."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import time
import uuid
from pathlib import Path

LATTICE_RE = re.compile(r'Lattice="([^"]+)"')


def read_structure(path, fallback_cell):
    with Path(path).open() as handle:
        natoms = int(handle.readline())
        comment = handle.readline()
        lattice = LATTICE_RE.search(comment)
        if lattice is not None:
            cell = [float(value) for value in lattice.group(1).split()]
        elif fallback_cell is not None:
            cell = list(fallback_cell)
        else:
            raise ValueError(
                "Structure has no Lattice field and --cell was not supplied"
            )
        if len(cell) != 9:
            raise ValueError("Cell must contain nine values")

        atoms = []
        for _ in range(natoms):
            fields = handle.readline().split()
            if len(fields) < 4:
                raise ValueError("Malformed structure")
            atoms.append((fields[0], [float(value) for value in fields[1:4]]))
    return atoms, cell


def write_xyz(path, atoms):
    lines = [f"{len(atoms)}\n", "fixed-bead PIMD configuration\n"]
    for symbol, position in atoms:
        lines.append(
            symbol + " " + " ".join(f"{value:.16g}" for value in position) + "\n"
        )
    path.write_text("".join(lines))


def write_input(path, args, cell, address):
    total_steps = args.equilibration_steps + args.production_steps
    cell_text = ", ".join(f"{value:.16g}" for value in cell)
    checkpoint_stride = max(1, args.checkpoint_stride)
    all_bead_force_output = ""
    # All replica forces are needed to fit the scaled-grad weights.
    if args.record_all_bead_forces:
        all_bead_force_output = f"""
    <trajectory stride='{args.stride}' filename='physical_all' cell_units='angstrom'>forces{{ev/ang}}</trajectory>
    <trajectory stride='{args.stride}' filename='spring_all' cell_units='angstrom'>forces_spring{{ev/ang}}</trajectory>"""

    xml = f"""<simulation verbosity='low'>
  <ffsocket mode='unix' name='driver' pbc='false'>
    <address>{address}</address>
    <latency>1.0e-3</latency>
    <timeout>120</timeout>
  </ffsocket>
  <total_steps>{total_steps}</total_steps>
  <output prefix='simulation'>
    <trajectory stride='{args.stride}' filename='fixed' bead='{args.fixed_bead}' cell_units='angstrom'>positions{{angstrom}}</trajectory>
    <trajectory stride='{args.stride}' filename='baseline' bead='{args.fixed_bead}' cell_units='angstrom'>forces{{ev/ang}}</trajectory>
    <trajectory stride='{args.stride}' filename='spring' bead='{args.fixed_bead}' cell_units='angstrom'>forces_spring{{ev/ang}}</trajectory>
    <trajectory stride='{args.stride}' filename='centroid' cell_units='angstrom'>f_centroid{{ev/ang}}</trajectory>
{all_bead_force_output}
    <properties stride='{args.stride}'>[ step, time{{femtosecond}}, conserved, temperature{{kelvin}}, potential ]</properties>
    <checkpoint stride='{checkpoint_stride}' filename='restart' overwrite='true'/>
  </output>
  <prng><seed>{args.seed}</seed></prng>
  <system>
    <forces><force forcefield='driver'/></forces>
    <initialize nbeads='{args.nbeads}'>
      <file mode='xyz' units='angstrom'>structure.xyz</file>
      <cell units='angstrom'>[ {cell_text} ]</cell>
    </initialize>
    <ensemble><temperature units='kelvin'>{args.temperature}</temperature></ensemble>
    <motion mode='dynamics'>
      <fixbeads>[{args.fixed_bead}]</fixbeads>
      <dynamics mode='nvt' splitting='baoab'>
        <timestep units='femtosecond'>{args.timestep_fs}</timestep>
        <thermostat mode='{args.thermostat}'>
          <tau units='femtosecond'>{args.thermostat_tau_fs}</tau>
        </thermostat>
      </dynamics>
    </motion>
  </system>
</simulation>
"""
    path.write_text(xml)


def terminate(process, timeout=20):
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def wait_for_socket(socket_path, ipi_process, timeout=60):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if socket_path.exists():
            return
        if ipi_process.poll() is not None:
            raise RuntimeError("i-PI exited before creating its socket")
        time.sleep(0.2)
    raise TimeoutError(f"Timed out waiting for {socket_path}")


def expand_command(command, address):
    return [part.replace("{address}", address) for part in command]


def run(args, run_dir, address):
    socket_path = Path(args.socket_prefix + address)
    if socket_path.exists():
        socket_path.unlink()

    restart = run_dir / "simulation.restart"
    input_name = restart.name if args.resume and restart.exists() else "input.xml"
    ipi_command = json.loads(args.ipi_command_json) + [input_name]
    driver_command = expand_command(json.loads(args.driver_command_json), address)
    environment = os.environ.copy()
    if args.pythonpath:
        environment["PYTHONPATH"] = args.pythonpath

    start = time.monotonic()
    ipi_log = (run_dir / "ipi.log").open("a")
    ipi = subprocess.Popen(
        ipi_command,
        cwd=run_dir,
        env=environment,
        stdout=ipi_log,
        stderr=subprocess.STDOUT,
    )
    drivers = []
    driver_logs = []
    try:
        wait_for_socket(socket_path, ipi)
        for driver_index in range(args.drivers):
            log = (run_dir / f"driver_{driver_index:02d}.log").open("a")
            process = subprocess.Popen(
                driver_command,
                cwd=run_dir,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            drivers.append(process)
            driver_logs.append(log)

        while ipi.poll() is None:
            if drivers and all(process.poll() is not None for process in drivers):
                try:
                    ipi.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    terminate(ipi)
                    raise RuntimeError("All force drivers exited before i-PI")
                break
            time.sleep(2)
    finally:
        terminate(ipi)
        for process in drivers:
            terminate(process)
        ipi_log.close()
        for log in driver_logs:
            log.close()
        if socket_path.exists():
            socket_path.unlink()

    if ipi.returncode != 0:
        raise RuntimeError(f"i-PI exited with return code {ipi.returncode}")
    return time.monotonic() - start


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--structure", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--nbeads", type=int, required=True)
    parser.add_argument("--cell", type=float, nargs=9)
    parser.add_argument("--fixed-bead", type=int, default=0)
    parser.add_argument("--equilibration-steps", type=int, default=8000)
    parser.add_argument("--production-steps", type=int, default=20000)
    parser.add_argument("--stride", type=int, default=40)
    parser.add_argument("--checkpoint-stride", type=int, default=4000)
    parser.add_argument("--timestep-fs", type=float, default=0.25)
    parser.add_argument("--thermostat-tau-fs", type=float, default=100.0)
    parser.add_argument(
        "--thermostat", choices=("langevin", "pile_l"), default="langevin"
    )
    parser.add_argument("--seed", type=int, default=18885)
    parser.add_argument("--drivers", type=int, default=1)
    parser.add_argument("--asset", action="append", default=[])
    parser.add_argument("--ipi-command-json", default='["i-pi"]')
    parser.add_argument("--driver-command-json", required=True)
    parser.add_argument("--pythonpath")
    parser.add_argument("--socket-prefix", default="/tmp/ipi_")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--record-all-bead-forces",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Record physical and spring forces for every bead for scaled-gradient analysis",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    completed = run_dir / "completed.json"
    if completed.exists():
        print(completed)
        return
    run_dir.mkdir(parents=True, exist_ok=True)

    atoms, cell = read_structure(args.structure, args.cell)
    write_xyz(run_dir / "structure.xyz", atoms)
    shutil.copy2(args.structure, run_dir / "structure.extxyz")
    for asset_name in args.asset:
        asset = Path(asset_name).resolve()
        if not asset.exists():
            raise FileNotFoundError(asset)
        target = run_dir / asset.name
        if not target.exists():
            target.symlink_to(asset)

    address = f"fixed-{uuid.uuid4().hex[:12]}"
    write_input(run_dir / "input.xml", args, cell, address)
    metadata = vars(args).copy()
    metadata.update(
        {"structure": str(Path(args.structure).resolve()), "address": address}
    )
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    elapsed = run(args, run_dir, address)
    result = {"status": "complete", "elapsed_seconds": elapsed}
    completed.write_text(json.dumps(result, indent=2) + "\n")
    print(completed)


if __name__ == "__main__":
    main()
