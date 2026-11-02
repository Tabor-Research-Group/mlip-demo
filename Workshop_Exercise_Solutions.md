# MLIP Workshop — Instructor Solutions

**17 exercise cells across 8 notebooks.** Cell numbers count **all cells, including Markdown, starting at 1**; they are not execution counts. Replace the exercise placeholders with the lines below and retain the surrounding supplied code. Keep loop-body indentation where applicable.

The notebooks are in `mlip_demo_update/demo/`. Complete notebooks are preserved in `mlip_workshop_backup_20261101/demo/`. `AdditionalExamples` contains no exercise blanks.

## 1. Introduction

Notebook: `Introduction/Introduction.ipynb`

### Cell 8 — Initialize the neutral-molecule calculator

```python
calculator = AIMNet2ASE('aimnet2', charge=0)
```

### Cell 13 — Attach the calculator

```python
H2.calc = calculator
```

### Cell 16 — Calculate energy and forces

```python
energy = H2.get_potential_energy()
forces = H2.get_forces()
```

### Cell 19 — Optimize the geometry

```python
opt = optimize_ase(H2, optimizer=BFGS, fmax=0.00001)
```

## 2. Conformer Sampling

Notebook: `ConformerSampling/ConformerSampling.ipynb`

### Cell 21 — Attach the calculator and optimize each conformer

Place these lines inside the supplied loop.

```python
mol.calc = calculator
opt = optimize_ase(
    mol, optimizer=BFGS, fmax=0.05,
    callback=record_dihedrals, interval=1
)
```

### Cell 25 — Evaluate each geometry in the rigid scan

```python
rigid_data.append((phi, psi, mol.get_potential_energy()))
```

## 3. Oxygen Adsorption on Copper

Notebook: `InorganicCrystals/Oxygen_binding_to_Cu111.ipynb`

### Cell 18 — Load the supplied MACE checkpoint

```python
calculator = MACECalculator(
    model_paths=MODEL, device=DEVICE, default_dtype=DEFAULT_DTYPE
)
```

### Cell 21 — Attach the calculator and relax each adsorption structure

Place these lines inside the supplied loop.

```python
systems[site]['optimized'].calc = calculator
dyn = optimize_ase(
    systems[site]['optimized'],
    optimizer=BFGS, fmax=0.05, steps=150
)
```

### Cell 31 — Calculate adsorption energy

```python
systems[site]['E_ads'] = (
    systems[site]['optenergy'] - cu_slab_energy - oxygen_energy
)
```

## 4. Molecular Crystals

Notebook: `MolecularCrystals/Molecular_Crystal_Optimization.ipynb`

### Cell 16 — Optimize with fixed and variable cells

Retain the other supplied lines in this cell. `FrechetCellFilter` allows the second optimization to change the lattice as well as atomic positions.

```python
fixedcell.calc = calculator
variablecell.calc = calculator

cowcas_fixedcell = optimize_ase(
    fixedcell, optimizer=FIRE, fmax=0.05, steps=MAX_STEPS
)
cowcas_variablecell = optimize_ase(
    FrechetCellFilter(variablecell),
    optimizer=FIRE, fmax=0.05, steps=MAX_STEPS
)
```

## 5. Transition States

Notebook: `TransitionStates/SN2_Reaction.ipynb`

### Cell 11 — Specify the anionic system's charge

```python
calculator = AIMNet2ASE('aimnet2-2025', charge=-1)
```

### Cell 18 — Run the Sella transition-state optimization

Retain the supplied constraint-clearing line before optimization.

```python
ts_guess_structure.calc = calculator
opt = Sella(ts_guess_structure, internal=True)
opt.run(1e-4, 1000)
```

## 6. CO₂ Vibrational MD

Notebook: `MolecularDynamics/CO2_Vibrational_MD.ipynb`

### Cell 6 — Attach the calculator

```python
atoms.calc = calc
```

### Cell 9 — Construct and run the NVE integrator

Retain the supplied timing statements around the run.

```python
dyn = VelocityVerlet(
    atoms, DT_FS * units.fs,
    trajectory=str(OUTPUT/'co2.traj'), loginterval=2
)
dyn.run(STEPS)
```

## 7. Alanine Dipeptide MD

Notebook: `MolecularDynamics/Alanine_Dipeptide_MD.ipynb`

### Cell 4 — Initialize AIMNet2

```python
calc = AIMNet2ASE('aimnet2', charge=0)
```

### Cell 9 — Supply the timestep and temperature, then run Langevin MD

Retain the supplied initialization and timing statements.

```python
dyn = Langevin(
    atoms, DT_FS * units.fs, temperature_K=TEMPERATURE_K,
    friction=0.01/units.fs, fixcm=False, rng=rng,
    trajectory=str(OUTPUT/'alanine.traj'),
    logfile=str(OUTPUT/'md.log'), loginterval=SAVE_EVERY
)
dyn.run(STEPS)
```

## 8. Model Internals

Notebook: `ModelInternals/ModelInternals.ipynb`

### Cell 16 — Retrieve predicted charges and the dipole

```python
partial_charges = water.get_charges()
partial_charge_dipole = water.get_dipole_moment()
```
