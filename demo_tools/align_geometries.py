import numpy as np
from ase import Atoms
from typing import Sequence


def align_atoms_list(
    geometries: Sequence[Atoms],
    atom_indices: Sequence[int],
    reference_index: int = 0,
) -> list[Atoms]:
    """
    Rigidly align a list of ASE Atoms objects using selected atoms.

    Each geometry is aligned to `geometries[reference_index]` by minimizing
    the RMSD of the atoms specified by `atom_indices`.

    Parameters
    ----------
    geometries
        List/sequence of ASE Atoms objects. All geometries must have the
        same number of atoms and identical atom ordering.
    atom_indices
        Atom indices used to determine the alignment.
    reference_index
        Index of the geometry to use as the reference. Default is 0.

    Returns
    -------
    aligned_geometries
        New ASE Atoms objects containing the aligned coordinates.
        The input geometries are not modified.
    """
    if len(geometries) == 0:
        return []

    atom_indices = np.asarray(atom_indices, dtype=int)

    if len(atom_indices) == 0:
        raise ValueError("atom_indices must contain at least one atom.")

    n_atoms = len(geometries[0])

    for i, atoms in enumerate(geometries):
        if len(atoms) != n_atoms:
            raise ValueError(
                f"Geometry {i} has {len(atoms)} atoms; expected {n_atoms}."
            )

    if np.any(atom_indices < 0) or np.any(atom_indices >= n_atoms):
        raise IndexError("atom_indices contains an invalid atom index.")

    ref = geometries[reference_index]
    ref_sel = ref.positions[atom_indices]

    # Center selected reference atoms
    ref_center = ref_sel.mean(axis=0)
    Q = ref_sel - ref_center

    aligned_geometries = []

    for atoms in geometries:
        new_atoms = atoms.copy()

        P_sel = atoms.positions[atom_indices]
        P_center = P_sel.mean(axis=0)
        P = P_sel - P_center

        # Kabsch algorithm
        H = P.T @ Q
        U, S, Vt = np.linalg.svd(H)

        R = U @ Vt

        # Prevent improper rotation / reflection
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = U @ Vt

        # Apply the transformation to ALL atoms
        positions = atoms.positions - P_center
        positions = positions @ R
        positions += ref_center

        new_atoms.set_positions(positions)
        aligned_geometries.append(new_atoms)

    return aligned_geometries