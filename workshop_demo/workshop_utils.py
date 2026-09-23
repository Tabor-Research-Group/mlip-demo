"""Small, model-independent helpers for the workshop notebooks."""
from pathlib import Path
from datetime import datetime
import os
import sys
import uuid
import numpy as np

ROOT = Path(__file__).resolve().parent


def start_exercise(folder):
    """Create a fresh output directory; never reuse vibration caches."""
    data = ROOT / folder
    output = data / 'results' / (datetime.now().strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:6])
    output.mkdir(parents=True, exist_ok=False)
    print(f'Python: {sys.executable}\nInputs: {data}\nNew results: {output}')
    return data, output


def mace_model(filename):
    """Use an instructor-staged checkpoint, with an explicit override."""
    path = Path(os.environ.get('MACE_MODEL', str(ROOT / 'models' / filename))).expanduser()
    if not path.is_file():
        raise FileNotFoundError(
            f'MACE checkpoint missing: {path}\n'
            'Ask the instructor for the validated checkpoint, place it in '
            f'{ROOT / "models"}, or set MACE_MODEL before launching Jupyter. '
            'Record the model used when comparing results.'
        )
    print(f'MACE checkpoint: {path}')
    return str(path)


def relax(atoms, optimizer, fmax=0.05, steps=150, **kwargs):
    """Bound optimization work and return an explicit convergence flag."""
    opt = optimizer(atoms, **kwargs)
    converged = bool(opt.run(fmax=fmax, steps=steps))
    print(f'Converged: {converged}; steps: {opt.nsteps}; target: {fmax} eV/angstrom')
    if not converged:
        print('Step limit reached: inspect this geometry before interpreting its energy.')
    return converged


def smoke_check(atoms):
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    if not np.isfinite(energy) or not np.isfinite(forces).all():
        raise ValueError('Non-finite energy or forces; stop and check the input and model.')
    print(f'Energy: {energy:.6f} eV; maximum force: {np.linalg.norm(forces, axis=1).max():.4f} eV/angstrom')


def signed_angle(angle):
    return (angle + 180) % 360 - 180
