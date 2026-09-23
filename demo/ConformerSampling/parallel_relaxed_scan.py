import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

from concurrent.futures import ProcessPoolExecutor
from ase.io import read, write
from ase.constraints import FixInternals
from ase.optimize import BFGS
from aimnet.calculators import AIMNet2ASE


phi_indices = [1,3,4,6][::-1]
phi_rotation_indices = [13,0,1,2,10,11,12]
psi_indices = [3,4,6,8]
psi_rotation_indices = [7,8,9,18,19,20,21]

def scan_psi(phi_structure):
    mol = phi_structure.copy()
    mol.set_constraint()

    target_phi = float(mol.info["target_phi"])
    results = [mol.copy()]  # Already optimized at psi = -180

    # Each process needs its own calculator.
    mol.calc = AIMNet2ASE(
        "aimnet2",
        charge=0,
    )

    for target_psi in range(-180, 181, 10):
        mol.set_constraint()

        # Rotate the previously optimized structure.
        mol.rotate_dihedral(
            *psi_indices,
            10,
            indices=psi_rotation_indices,
        )

        mol.set_constraint(
            FixInternals(
                dihedrals_deg=[
                    (target_phi, phi_indices),
                    (target_psi, psi_indices),
                ]
            )
        )

        print(
            f"Starting phi={target_phi:.0f}, psi={target_psi}",
            flush=True,
        )

        dyn = BFGS(mol, logfile=None)
        dyn.run(fmax=0.01)

        mol.info["target_phi"] = target_phi
        mol.info["target_psi"] = target_psi
        mol.info["actual_phi"] = mol.get_dihedral(*phi_indices)
        mol.info["actual_psi"] = mol.get_dihedral(*psi_indices)
        mol.info["Energy"] = mol.get_potential_energy()
        mol.info["converged"] = dyn.converged()

        results.append(mol.copy())

        print(
            f"Finished phi={target_phi:.0f}, psi={target_psi}, "
            f"steps={dyn.nsteps}",
            flush=True,
        )

    return results


if __name__ == "__main__":
    phi_structures = read(
        "phi_1D_scan_structures.xyz",
        index=":",
        format="extxyz",
    )

    n_workers = 4

    print(
        f"Running {len(phi_structures)} phi scans "
        f"with {n_workers} workers",
        flush=True,
    )

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        psi_scans = list(
            executor.map(scan_psi, phi_structures, chunksize=1)
        )

    relaxed_2d_structures = [
        structure
        for scan in psi_scans
        for structure in scan
    ]

    write(
        "Relaxed_2D_scan.xyz",
        relaxed_2d_structures,
        format="extxyz",
    )

    print(
        f"Finished {len(relaxed_2d_structures)} structures",
        flush=True,
    )