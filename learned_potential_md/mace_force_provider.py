#!/usr/bin/env python3
"""Serve MACE energies and forces to i-PI through ASE's socket client."""

import argparse

from ase.calculators.socketio import SocketClient
from ase.io import read
from mace.calculators import MACECalculator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--address", required=True)
    parser.add_argument("--structure", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    atoms = read(args.structure)
    atoms.calc = MACECalculator(
        model_paths=args.model, device=args.device, default_dtype="float32"
    )
    # ASE handles the i-PI protocol and unit conversions. NVT does not require
    # stress. Periodicity is retained from the extxyz structure template.
    client = SocketClient(unixsocket=args.address)
    try:
        client.run(atoms, use_stress=False)
    finally:
        client.close()


if __name__ == "__main__":
    main()
