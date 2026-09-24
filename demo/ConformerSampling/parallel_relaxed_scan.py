"""Run independent constrained torsion rows inside an allocated compute job."""
import os
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '1'
os.environ['TORCH_COMPILE_DISABLE'] = '1'
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

PHI = [6, 4, 3, 1]
PSI = [3, 4, 6, 8]
PSI_ROTATION = [7, 8, 9, 18, 19, 20, 21]


def scan_row(task):
    # Imports and calculator construction happen in the worker, not in argparse.
    from ase.io import write
    from ase.constraints import FixInternals
    from ase.optimize import BFGS
    from aimnet.calculators import AIMNet2ASE
    mol, step, fmax, max_steps, row_path = task
    phi = float(mol.info['target_phi'])
    mol.calc = AIMNet2ASE('aimnet2', charge=0)
    frames = []
    for psi in range(-180, 180, step):
        mol.set_constraint()
        # Set the exact target before imposing the constraint; no shifted first step.
        mol.set_dihedral(*PSI, float(psi), indices=PSI_ROTATION)
        mol.set_constraint(FixInternals(dihedrals_deg=[(phi, PHI), (float(psi), PSI)]))
        opt = BFGS(mol, logfile=None)
        ok = bool(opt.run(fmax=fmax, steps=max_steps))
        mol.info.update(target_phi=phi, target_psi=float(psi),
                        actual_phi=mol.get_dihedral(*PHI), actual_psi=mol.get_dihedral(*PSI),
                        Energy=mol.get_potential_energy(), converged=ok)
        if not ok:
            raise RuntimeError(f'phi={phi}, psi={psi} failed after {max_steps} steps')
        frames.append(mol.copy())
    # Each worker writes a separate row, avoiding concurrent writes to one file.
    write(row_path, frames, format='extxyz')
    return row_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=None)
    parser.add_argument('--step', type=int, default=30)
    parser.add_argument('--fmax', type=float, default=0.05)
    parser.add_argument('--max-steps', type=int, default=150)
    args = parser.parse_args()
    allocated = int(os.environ.get('SLURM_CPUS_PER_TASK', '1'))
    workers = args.workers if args.workers is not None else allocated
    if workers < 1 or args.step <= 0 or 360 % args.step or args.step >= 180:
        parser.error('Use positive workers and a grid step smaller than 180 that divides 360.')
    if args.fmax <= 0 or args.max_steps < 1:
        parser.error('fmax and max-steps must be positive.')
    if 'SLURM_JOB_ID' in os.environ and workers > allocated:
        parser.error('workers exceeds SLURM_CPUS_PER_TASK; request sufficient CPUs.')
    if args.output.exists():
        parser.error('Output exists. Choose a new output path to preserve previous results.')
    from ase.io import read, write
    structures = read(args.input, index=':')
    if not structures:
        parser.error('Input contains no structures.')
    for mol in structures:
        if 'target_phi' not in mol.info or not mol.info.get('converged', False):
            parser.error('Every input row must have target_phi and converged=True metadata.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    row_dir = args.output.with_suffix('.rows')
    row_dir.mkdir(exist_ok=False)
    tasks = [(mol, args.step, args.fmax, args.max_steps, row_dir / f'row_{i:03d}.xyz')
             for i, mol in enumerate(structures)]
    print(f'{len(tasks)} rows, {workers} workers; checkpoints: {row_dir}', flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(scan_row, task) for task in tasks]
        for n, future in enumerate(as_completed(futures), 1):
            print(f'Completed {n}/{len(tasks)}: {future.result()}', flush=True)
    frames = [frame for task in tasks for frame in read(task[-1], index=':')]
    write(args.output, frames, format='extxyz')
    print(f'Wrote {len(frames)} converged structures to {args.output}', flush=True)


if __name__ == '__main__':
    main()
