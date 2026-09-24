# MLIP workshop: guided demo copy

This folder is a revised copy of `demo/`. The original demos and repository
environment/container files are unchanged. Start here, then open a notebook.
Use an allocated compute session for calculations; follow the instructor's HPRC
launch instructions rather than running workloads on a login node.

## Workshop route

Prerequisites: basic Python/Jupyter use, energies and forces, and introductory
molecular or crystal structure concepts. No model training is required.

| Notebook | Kernel | Outcome |
| --- | --- | --- |
| [Cu equation of state](InorganicCrystals/Cu_Equation_of_State.ipynb) | MACE | Fit and interpret an energy-volume curve |
| [Oxygen adsorption](InorganicCrystals/Oxygen_binding_to_Cu111.ipynb) | MACE | Compare relaxed adsorption sites |
| [Molecular crystals](MolecularCrystals/Molecular_Crystal_Optimization.ipynb) | MACE | Compare fixed-cell and variable-cell relaxation |
| [Conformer sampling](ConformerSampling/ConformerSampling.ipynb) | AIMNet2 | Compare rigid/relaxed torsional surfaces; optional batch job |
| [SN2 reaction](TransitionStates/SN2_Reaction.ipynb) | AIMNet2 + Sella | Search for and assess a saddle point |
| [Diels-Alder planning](TransitionStates/DielsAlderReaction.ipynb) | No execution needed | Design a future reaction calculation |

Begin with Cu for a common introduction. Then select the materials track
(adsorption and crystals) or molecular track (conformers and SN2). Do not require
every participant to finish every notebook. The dense parallel scan and IRC are
extensions. Set session timings after rehearsing on the actual allocated hardware.

## Instructor preparation and participant preflight

1. Provide a tested environment/container and exact launch instructions for the
   target cluster, including account, partition, CPU/GPU allocation, and Jupyter access.
   The repository has `environment-mace.yml` and `environment.yml` (AIMNet2).
   These files are starting points, not frozen or newly validated dependencies.
2. Make the `mace` and `aimnet2` kernels available. Outside the existing container,
   an instructor can register them from their respective environments with
   `python -m ipykernel install --user --name mace --display-name MACE` and
   `python -m ipykernel install --user --name aimnet2 --display-name AIMNet2`.
   Registration is an explicit setup action; opening these notebooks does not install anything.
3. Stage the validated MACE checkpoints under `models/`:
   `2023-12-03-mace-128-L1_epoch-199.model` for the Cu exercises and
   `2023-12-10-mace-128-L0_energy_epoch-249.model` for molecular crystals.
   Alternatively, set `MACE_MODEL` to an absolute checkpoint path before launching
   Jupyter. That override applies to every MACE exercise, so record when it replaces
   the default. No checkpoints are bundled or automatically downloaded here.
4. Confirm AIMNet2 model loading for `aimnet2` and `aimnet2-2025`, and stage its
   cache for the compute environment. Resolve any network/access requirements before class.
5. Open a notebook from the repository or this folder, select its named kernel,
   and run the setup and single-energy/force check. Confirm that the printed
   Python executable is expected and the browser can display an X3D structure.
6. Rehearse **Restart Kernel and Run All** for the required exercises with fresh
   output folders. Record dependency versions, checkpoint hashes, runtimes, and
   representative convergence results. The revised SN2 charge and slab constraints
   require fresh reference calculations; old results cannot validate them.

The default device is CPU. Selecting `cuda` also requires a GPU allocation and
a compatible Python/PyTorch/model environment; changing the notebook string alone
does not establish that setup. UMA is not required for this workshop route.

## Participant workflow

- Read the prediction question before running the next code cell.
- Run in order; a red error means stop and resolve that step before continuing.
- Inspect convergence and geometry before interpreting an energy.
- Save the result plus the model, electronic state, units, and calculation settings.
- Explain one conclusion and one limitation to a partner at each checkpoint.

The setup cell creates a fresh `results/<timestamp>-<id>/` folder within the
exercise folder. Rerunning setup creates another folder. Keep the folder path
printed by the run; outputs are intentionally not written beside input data.

## Supplied reference results

Original generated scans and transition-state artifacts have been preserved under
each exercise's `reference_results/`. These are illustrative, unvalidated legacy
outputs, not guaranteed results for the revised settings. The conformer notebook
can plot the supplied relaxed surface when a batch calculation would take too long.
The SN2 notebook never reads the legacy vibration cache or IRC results.

## Parallel scan and Slurm

The conformer notebook prepares a new input file with convergence metadata. The
batch scanner uses one process per independent phi row and one numerical thread
per process. It defaults to `SLURM_CPUS_PER_TASK` workers, or one outside Slurm.
The Slurm template requests four CPUs; its memory and walltime are starting values
that must be checked on the actual workshop environment.

From `workshop_demo/ConformerSampling`, after adapting the template:

```bash
sbatch submit_scan.slurm results/YOUR_RUN/phi_1D_scan_structures.xyz results/YOUR_RUN/Relaxed_2D_scan.xyz
squeue -u "$USER"
# Replace JOB_ID with the ID returned by sbatch:
tail -f mlip-scan-JOB_ID.out
scancel JOB_ID
```

Only cancel your own job when needed. A queued job has not yet started computing.
For a small local/interactive allocation, invoke `parallel_relaxed_scan.py` directly
with `--workers` matching the allocated CPUs.

Completed rows are stored in `Relaxed_2D_scan.rows/`. If any row fails, the combined
surface is not written. Inspect the error and completed rows; use a new output name
when rerunning after correcting the issue. Automatic resume/merging is not implemented.
The script refuses to overwrite an existing combined output or row directory.

## Troubleshooting

| Symptom | Next action |
| --- | --- |
| Missing `mace`, `aimnet`, or `sella` | Check the selected kernel and printed Python path; ask the instructor about the prepared environment |
| Missing checkpoint | Stage the named file or set the absolute `MACE_MODEL` path |
| Model loading stalls | Check model/cache access in the allocated environment before restarting calculations |
| Optimization hits the step limit | Inspect geometry/logs; do not rank its energy as a converged result |
| SN2 scan has no interior maximum | Inspect the constrained path and electronic state; revise the range/guess |
| Unexpected imaginary modes | Inspect displacement patterns, convergence, and finite-difference settings |
| Blank 3D display | Ask the instructor to check browser support; coordinates and numerical results remain inspectable |

## Revision and validation scope

The notebooks now have learning goals, guided steps, prediction questions, bounded
calculations, interpretation checkpoints, and explicitly separated optional work.
The SN2 model charge is -1 for the stated anionic reaction. Adsorption comparisons
use same-composition relative energies and fixed bottom-layer atoms. The crystal
exercise reports convergence and compares cell-relaxation choices.

Notebook code and Python helper syntax were checked without running MLIP calculations.
Input structures and supplied surface metadata were inspected with ASE. Numerical
MLIP behavior, cluster launch details, browser rendering, and workshop timings still
require rehearsal with the intended environments and checkpoints.

`original_demo_sha256.json` records hashes of the untouched original demo files.
