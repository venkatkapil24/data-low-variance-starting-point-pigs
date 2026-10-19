# Low-variance force labels for path-integral coarse graining

Scripts, labeled datasets and learned potentials for the paper. The four stages
are replica-constrained PIMD, scaled-grad force labeling, force-only MACE training,
and one-replica MD with the learned potential.

| Folder | Contents |
|---|---|
| `replica_constrained_pimd/` | Select structures and launch one constrained run or a parallel batch |
| `effective_force_labeling/` | Fit the scaled-grad weights, calculate labels/errors, and preserve dataset splits |
| `mace_training/` | Force-only training script and the paper's MACE architecture/optimizer settings |
| `learned_potential_md/` | One-replica i-PI MD and an ASE/MACE socket force provider |
| `datasets/` | The paper's train/validation/test labels and constrained structures |
| `models/` | The three models used for the paper's production trajectories |

## Environment

Use Python 3.10 and the versions in `requirements.txt`. Install them in a
separate environment. Build/install i-PI with the constrained-bead
implementation at commit `7a1dfa9039225605154b59489625a06f70026864` on
`venkatkapil24/i-pi`, branch `feat/centroid-bead-constraints`. The reproduction
scripts are separate from the i-PI source checkout.

```bash
python -m pip install -r requirements.txt
export IPI_ROOT=/absolute/path/to/i-pi
```

The reference force engines are i-PI's Fortran `zundel` and `qtip4pf` drivers.
Zundel also requires `h5o2.pes4B.coeff.dat` and `h5o2.dms4B.coeff.com.dat` for the
HBB reference potential; pass their paths with `--asset`. These force-engine
assets are supplied separately from the labeled datasets and learned models.

## 1. Run replica-constrained PIMD

The supplied `constrained_configurations.extxyz` files contain the paper's
retained structures, sorted by configuration index. Selecting all of them
preserves the indices in the supplied train/validation/test split.

For example, for water:

```bash
python replica_constrained_pimd/select_configurations.py \
  datasets/water_300K/constrained_configurations.extxyz \
  results/water_structures --count 128

python replica_constrained_pimd/run_batch.py \
  results/water_structures/manifest.json results/water_runs \
  --temperature 300 --nbeads 32 --max-workers 4 \
  --ipi-command-json "[\"python\",\"$IPI_ROOT/bin/i-pi\"]" \
  --driver-command-json "[\"$IPI_ROOT/drivers/f90/driver.x\",\"-u\",\"-a\",\"{address}\",\"-m\",\"qtip4pf\"]" \
  --pythonpath "$IPI_ROOT"
```

`run_pimd.py --help` exposes the same controls for a single structure. The
default setup matches the labeling runs: BAOAB, 0.25 fs integration steps,
Cartesian Langevin with 100 fs damping, 8000 equilibration steps and 20000
production steps. Forces are saved every 40 steps (**10 fs**), for every replica.
The first replica is fixed. For Zundel use 64 structures, 96 replicas at 100 K
or 32 replicas at 300 K, the `zundel` driver, and the two coefficient assets.

PILE-L is optional via `--thermostat pile_l`. With fixed beads its damping time
is `min(1 / omega_k, tau_input)` at the specified i-PI commit. PILE-L was not the
thermostat used to generate the supplied paper datasets.

## 2. Calculate scaled-grad force labels

```bash
python effective_force_labeling/label_forces.py \
  results/water_runs results/water_labels --nbeads 32

python effective_force_labeling/split_labels.py \
  results/water_labels/labels.extxyz datasets/water_300K/summary.json \
  results/water_labels/split
```

The labeling script is self-contained and commented in calculation order.
Physical and spring forces are added to obtain the total replica forces. The
retained replica has weight one. All other weights are fitted separately for
each constrained structure by minimizing the block-force variance with ridge
regularization towards zero.

The 5 ps production trajectory gives 500 saved snapshots and twenty consecutive
0.25 ps blocks. Ten alternating blocks fit the weights; the other ten evaluate
them. The sets are then exchanged. Within each fitting set, an inner five/five
swap chooses the ridge strength from the paper's grid using the one-standard-error
rule. The twenty evaluated block forces determine the mean label and component
standard errors. Block independence remains an assumption to check when changing
simulation conditions or duration.

`labels.extxyz` contains `REF_forces` and `REF_force_sem`, both in eV/angstrom.
`analysis.json` records the weights, ridge strengths, block assignments and RMS
force-label SEM in meV/angstrom. Input may be raw XYZ, gzipped XYZ, or the existing
`forces.h5` physical/spring archives. Equilibration is discarded at step 8000;
step 28000 is excluded to obtain exactly 500 production snapshots.

## 3. Train MACE

To train directly from the supplied labels:

```bash
python mace_training/train_model.py \
  --train datasets/water_300K/train.extxyz \
  --validation datasets/water_300K/validation.extxyz \
  --test datasets/water_300K/test.extxyz \
  --output results/water_training --name water_scaled_grad --device cuda
```

The model is trained on forces only; energy labels are not required. The
architecture and optimizer are in `mace_training/mace.yaml`.
The effective YAML and launch command
are saved with the training outputs. The wrapper starts a fresh run rather than
silently resuming an unrelated checkpoint.

The supplied datasets contain 48/8/8 train/validation/test structures for each
Zundel temperature and 112/8/8 for water. The supplied models are the paper's
selected checkpoints: epoch 362 for Zundel 100 K, epoch 122 for Zundel 300 K,
and the preselected epoch 80 for water. Retraining on different hardware or
software does not guarantee identical floating-point weights.

`temperature_K` specifies the constrained-label temperature. Some Zundel
structures also retain a `temperature` field from their source sampling; that
field does not specify the temperature at which the force label was calculated.

## 4. Run learned-potential MD

```bash
python learned_potential_md/run_md.py \
  --structure datasets/water_300K/constrained_configurations.extxyz \
  --model models/water_300K_epoch80.model --ipi-root "$IPI_ROOT" \
  --run-dir results/water_md --temperature 300 --steps 400000 \
  --timestep-fs 0.25 --splitting baoab --thermalize --device cpu
```

This runs 100 ps with one moving replica, no bead constraint, and 100 fs
Cartesian Langevin damping. The ASE socket provider preserves the structure's
periodic boundary conditions and uses the complete learned effective potential.
For Zundel, use its corresponding model/structure, 200000 steps at 0.5 fs,
`--splitting obabo --properties-stride 5`, and omit `--thermalize` to match the
paper's initialization. Water properties are written every ten steps by default.
