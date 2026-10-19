#!/usr/bin/env python3
"""Run unconstrained one-bead i-PI MD with a learned effective potential.

Reuses only the PIMD launcher's process/socket handling. The XML below has
one moving replica and no fixed-bead constraint. The MACE force provider
evaluates the complete learned effective potential, without a classical prior.
"""

import argparse
import json
from pathlib import Path
import sys
import uuid

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "replica_constrained_pimd")
)
from run_pimd import read_structure, write_xyz, run


def main():
    from ase.io import read, write

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structure", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--ipi-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--cell", type=float, nargs=9)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--timestep-fs", type=float, default=0.25)
    parser.add_argument("--splitting", choices=("obabo", "baoab"), default="baoab")
    parser.add_argument("--thermalize", action="store_true")
    parser.add_argument("--stride", type=int, default=40)
    parser.add_argument("--properties-stride", type=int, default=10)
    parser.add_argument("--seed", type=int, default=18885)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    args.model = args.model.resolve()
    args.ipi_root = args.ipi_root.resolve()
    if not args.model.exists():
        parser.error("Model file does not exist")
    atoms, cell = read_structure(args.structure, args.cell)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    run_dir = args.run_dir.resolve()
    write_xyz(run_dir / "structure.xyz", atoms)
    # Plain XYZ cannot retain periodicity; the force provider uses this extxyz.
    template = read(args.structure, index=0)
    template.set_cell([cell[i : i + 3] for i in (0, 3, 6)])
    write(run_dir / "structure.extxyz", template, format="extxyz")
    address = "effective-md-" + uuid.uuid4().hex[:12]
    velocities = (
        f"<velocities mode='thermal' units='kelvin'>{args.temperature}</velocities>"
        if args.thermalize
        else ""
    )
    xml = f"""<simulation verbosity='low'>
  <ffsocket mode='unix' name='mace' pbc='false'>
    <address>{address}</address><latency>1.0e-3</latency><timeout>120</timeout>
  </ffsocket>
  <total_steps>{args.steps}</total_steps>
  <output prefix='simulation'>
    <trajectory stride='{args.stride}' filename='pos' cell_units='angstrom'>positions{{angstrom}}</trajectory>
    <properties stride='{args.properties_stride}'>[step,time{{femtosecond}},temperature{{kelvin}},potential]</properties>
    <checkpoint stride='4000' filename='restart' overwrite='true'/>
  </output>
  <prng><seed>{args.seed}</seed></prng>
  <system>
    <forces><force forcefield='mace'/></forces>
    <initialize nbeads='1'>
      <file mode='xyz' units='angstrom'>structure.xyz</file>
      <cell units='angstrom'>[{','.join(map(str, cell))}]</cell>{velocities}
    </initialize>
    <ensemble><temperature units='kelvin'>{args.temperature}</temperature></ensemble>
    <motion mode='dynamics'><dynamics mode='nvt' splitting='{args.splitting}'>
      <timestep units='femtosecond'>{args.timestep_fs}</timestep>
      <thermostat mode='langevin'><tau units='femtosecond'>100</tau></thermostat>
    </dynamics></motion>
  </system>
</simulation>"""
    (run_dir / "input.xml").write_text(xml)
    args.ipi_command_json = json.dumps(
        [sys.executable, str(args.ipi_root / "bin/i-pi")]
    )
    args.driver_command_json = json.dumps(
        [
            sys.executable,
            str(Path(__file__).with_name("mace_force_provider.py")),
            "--address",
            "{address}",
            "--structure",
            "structure.extxyz",
            "--model",
            str(args.model),
            "--device",
            args.device,
        ]
    )
    args.drivers = 1
    args.pythonpath = str(args.ipi_root)
    args.socket_prefix = "/tmp/ipi_"
    args.resume = False
    (run_dir / "metadata.json").write_text(
        json.dumps(vars(args), default=str, indent=2) + "\n"
    )
    elapsed = run(args, run_dir, address)
    (run_dir / "completed.json").write_text(
        json.dumps(dict(elapsed_seconds=elapsed)) + "\n"
    )


if __name__ == "__main__":
    main()
