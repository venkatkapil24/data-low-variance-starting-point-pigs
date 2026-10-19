#!/usr/bin/env python3
"""Turn constrained-PIMD force trajectories into scaled-grad force labels.

One invocation processes a collection of constrained configurations. For each
configuration, the retained replica stays fixed while the other replicas move.
The output contains one mean-force vector and one component-wise standard error
for that retained structure. Forces throughout this script are in eV/angstrom.

The calculation, in the order a reader can follow below:
  1. Read physical and spring forces, and discard equilibration.
  2. Average the retained production samples into twenty consecutive blocks.
  3. Fit replica weights on ten alternating blocks, then apply them to the other
     ten. Swap the two sets so that every block contributes to the final label.
  4. Inside each fitting set, select ridge regularization using a five/five swap.
  5. Average the twenty evaluated block forces and calculate their standard error.

Every atom and Cartesian component contributes to choosing one common vector of
replica weights for a constrained configuration. Different constrained structures
are fitted separately. No weights are fitted across different structures.

This reproduces the paper's unrestricted ridge-to-zero estimator. No harmonic
weight profile, displaced estimator, or reference frequency is used here.
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
import re

import numpy as np

# These are the same dimensionless candidate strengths used for the paper.
# The actual regularization is alpha * mean(diagonal(free-force covariance)).
RIDGE_GRID = np.logspace(-8.0, 6.0, 29)


def read_xyz(path):
    """Read one replica's force snapshots, including their integration-step IDs."""
    opener = gzip.open if path.suffix == ".gz" else open
    symbols, steps, frames = None, [], []
    with opener(path, "rt") as handle:
        while line := handle.readline():
            natoms = int(line)
            match = re.search(r"Step:\s*(\d+)", handle.readline())
            if match is None:
                raise ValueError(f"Missing Step field in {path}")
            rows = [handle.readline().split() for _ in range(natoms)]
            names = [row[0] for row in rows]
            if symbols is not None and symbols != names:
                raise ValueError(f"Atom order changes in {path}")
            symbols = names
            steps.append(int(match[1]))
            frames.append([[float(x) for x in row[1:4]] for row in rows])
    return symbols, np.asarray(steps), np.asarray(frames)


def load_forces(run, nbeads):
    """Return step IDs and total forces with shape (sample, replica, atom, xyz).

    Read existing compressed archives directly. For plain/gzipped XYZ outputs,
    verify that every physical/spring trajectory describes the same snapshots.
    The spring force is added to the physical force, as required by the estimator.
    """
    archive = run / "forces.h5"
    if archive.exists():
        import h5py

        with h5py.File(archive) as handle:
            if not handle.attrs.get("conversion_complete", False):
                raise ValueError(f"Incomplete archive: {archive}")
            steps = handle["steps"][:]
            total = handle["physical_all"][:] + handle["spring_all"][:]
    else:
        total, steps, symbols = None, None, None
        width = len(str(nbeads))
        for prefix in ("physical_all", "spring_all"):
            replicas = []
            for bead in range(nbeads):
                path = run / f"simulation.{prefix}_{bead:0{width}d}.xyz"
                if not path.exists():
                    path = path.with_suffix(".xyz.gz")
                names, current_steps, values = read_xyz(path)
                if steps is None:
                    steps, symbols = current_steps, names
                elif names != symbols or not np.array_equal(steps, current_steps):
                    raise ValueError(f"Unaligned force trajectory: {path}")
                replicas.append(values)
            values = np.stack(replicas, axis=1)
            total = values if total is None else total + values
    if total.ndim != 4 or total.shape[1] != nbeads or total.shape[-1] != 3:
        raise ValueError(f"Invalid force-array shape in {run}: {total.shape}")
    if not np.isfinite(total).all() or np.any(np.diff(steps) <= 0):
        raise ValueError(f"Non-finite forces or non-increasing steps in {run}")
    return steps, total


def covariance_matrix(blocks):
    """Covariance between replica forces, averaged over atom/xyz components.

    Input axes are (block, replica, atom, xyz). Flattening the last two axes
    only lets us treat all force components equally; it does not merge time
    samples or combine different constrained structures.
    """
    components = blocks.reshape(len(blocks), blocks.shape[1], -1)
    fluctuations = components - components.mean(axis=0, keepdims=True)
    return np.einsum("rId,rJd->IJ", fluctuations, fluctuations) / (
        (len(blocks) - 1) * components.shape[-1]
    )


def fit_weights(covariance, alpha):
    """Solve the ridge-regularized variance minimization, keeping w[0] = 1.

    The retained replica's weight is fixed at one. Its force is the target whose
    fluctuations we cancel using total forces from the free replicas. This gives
      (free_covariance + regularization * identity) @ free_weights = -cross_covariance.
    The common estimator factor 1/P does not affect the minimizing weights.

    Diagonalizing the covariance gives the same solution as a matrix solve and
    also handles directions with effectively zero sampled variance. The cutoff
    below is unchanged from the analysis used for the paper.
    """
    free_covariance = covariance[1:, 1:]
    cross_covariance = covariance[1:, 0]
    regularization = alpha * np.trace(free_covariance) / len(free_covariance)
    eigenvalues, eigenvectors = np.linalg.eigh(free_covariance)
    denominator = eigenvalues + regularization
    cutoff = max(float(eigenvalues[-1]) * 1.0e-12, np.finfo(float).eps)
    inverse = np.zeros_like(denominator)
    keep = denominator > cutoff
    inverse[keep] = 1.0 / denominator[keep]
    # Archived forces may be float32. Keep the right-hand side in float64,
    # matching the paper's ridge solve rather than rounding this product early.
    rhs = -cross_covariance.astype(np.float64)
    free_weights = eigenvectors @ (inverse * (eigenvectors.T @ rhs))
    return np.concatenate(([1.0], free_weights))


def apply_weights(blocks, weights):
    """Sum weighted total forces over replicas and divide by their number P."""
    return np.einsum("I,rIac->rac", weights, blocks) / len(weights)


def select_regularization(fitting_blocks):
    """Choose alpha using only the ten blocks reserved for weight fitting.

    Fit on five alternating blocks and score on the other five, then swap.
    Scoring measures fluctuations about the validation-force mean, not the
    unknown exact mean force. Validation blocks do not enter their fitted weights.
    """
    if len(fitting_blocks) != 10:
        raise ValueError("Regularization selection requires ten fitting blocks")
    even, odd = np.arange(0, 10, 2), np.arange(1, 10, 2)
    losses = [[] for _ in RIDGE_GRID]
    for fit, validation in ((even, odd), (odd, even)):
        covariance = covariance_matrix(fitting_blocks[fit])
        for index, alpha in enumerate(RIDGE_GRID):
            weights = fit_weights(covariance, alpha)
            forces = apply_weights(fitting_blocks[validation], weights)
            fluctuations = forces - forces.mean(axis=0, keepdims=True)
            # One loss per validation block, averaged over all force components.
            losses[index].extend(np.mean(fluctuations**2, axis=(1, 2)))

    losses = np.asarray(losses)
    mean_loss = losses.mean(axis=1)
    loss_sem = losses.std(axis=1, ddof=1) / np.sqrt(losses.shape[1])
    best = int(np.argmin(mean_loss))
    # The one-standard-error rule favors stronger regularization when its score
    # cannot be distinguished from the best score at this estimated precision.
    acceptable = mean_loss <= mean_loss[best] + loss_sem[best]
    selected = int(np.flatnonzero(acceptable)[-1])
    return float(RIDGE_GRID[selected])


def label_configuration(total):
    """Produce a mean force and SEM from twenty evaluated block forces."""
    if len(total) < 20 or len(total) % 20:
        raise ValueError("Production samples must divide into twenty equal blocks")
    # In the paper, 500 snapshots become twenty blocks of 25 snapshots each.
    # At 10 fs output spacing, each block then spans 0.25 ps.
    blocks = total.reshape(20, len(total) // 20, *total.shape[1:]).mean(axis=1)
    evaluated = np.empty((20, *total.shape[2:]))
    even, odd = np.arange(0, 20, 2), np.arange(1, 20, 2)
    fits = []
    for fit, evaluation in ((even, odd), (odd, even)):
        alpha = select_regularization(blocks[fit])
        weights = fit_weights(covariance_matrix(blocks[fit]), alpha)
        # These ten blocks were excluded from both the weight fit and alpha
        # selection used for them. The second pass evaluates the remaining ten.
        evaluated[evaluation] = apply_weights(blocks[evaluation], weights)
        fits.append(
            dict(
                fitting_blocks=fit.tolist(),
                evaluation_blocks=evaluation.tolist(),
                alpha=alpha,
                weights=weights.tolist(),
            )
        )
    mean_force = evaluated.mean(axis=0)
    # This estimates the error of the mean, not the spread of individual forces.
    # Its interpretation still requires sufficiently long averaging blocks.
    sem = evaluated.std(axis=0, ddof=1) / np.sqrt(len(evaluated))
    return dict(
        mean_force_eV_per_A=mean_force.tolist(),
        sem_eV_per_A=sem.tolist(),
        rms_sem_meV_per_A=float(1000 * np.sqrt(np.mean(sem**2))),
        fits=fits,
    )


def main():
    from ase.io import read, write

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, help="Directory containing anchor_XXXX runs")
    parser.add_argument("output", type=Path)
    parser.add_argument("--nbeads", type=int, required=True)
    parser.add_argument("--fixed-bead", type=int, default=0)
    parser.add_argument("--equilibration-step", type=int, default=8000)
    parser.add_argument(
        "--final-step",
        type=int,
        default=28000,
        help="Exclusive production endpoint; avoids an extra final snapshot",
    )
    parser.add_argument("--pattern", default="anchor_*")
    args = parser.parse_args()
    if args.nbeads < 2 or not 0 <= args.fixed_bead < args.nbeads:
        parser.error("Require at least two replicas and a valid fixed-bead index")
    runs = sorted(path for path in args.runs.glob(args.pattern) if path.is_dir())
    if not runs:
        parser.error("No constrained runs matched --pattern")
    args.output.mkdir(parents=True, exist_ok=True)
    records, structures = [], []
    for run in runs:
        steps, total = load_forces(run, args.nbeads)
        keep = (steps >= args.equilibration_step) & (steps < args.final_step)
        # Put the retained replica first so w[0] = 1 applies for any chosen bead.
        total = np.roll(total[keep], -args.fixed_bead, axis=1)
        record = label_configuration(total)
        index = int(run.name.rsplit("_", 1)[1])
        record.update(
            configuration=index,
            source=str(run.resolve()),
            production_samples=len(total),
            samples_per_block=len(total) // 20,
        )
        atoms = read(run / "structure.extxyz")
        force = np.asarray(record["mean_force_eV_per_A"])
        if force.shape != (len(atoms), 3):
            raise ValueError(f"Structure/force shape mismatch in {run}")
        atoms.calc = None
        atoms.arrays["REF_forces"] = force
        atoms.arrays["REF_force_sem"] = np.asarray(record["sem_eV_per_A"])
        atoms.info["configuration_index"] = index
        structures.append(atoms)
        records.append(record)
        print(
            run.name,
            "RMS force-label SEM:",
            record["rms_sem_meV_per_A"],
            "meV/A",
            flush=True,
        )
    write(args.output / "labels.extxyz", structures, format="extxyz")
    (args.output / "analysis.json").write_text(
        json.dumps(
            dict(
                estimator="unrestricted ridge scaled-grad",
                blocks=20,
                configurations=records,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
