import numpy as np
from ase import Atoms
from typing import Sequence

from rdkit import Chem
from rdkit.Chem import rdDetermineBonds, rdMolAlign

def signed_angle(angle):
    return (angle + 180) % 360 - 180

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


def deduplicate_conformers(conformers, rmsd_threshold=0.1,
                          heavy_atoms_only=True, charge=0):
    """Deduplicate ASE conformers using symmetry-aware, aligned RDKit RMSD.

    Threshold is in angstrom. Returns original ASE objects unchanged.
    """
    conformers = list(conformers)
    if not conformers:
        return []

    if rmsd_threshold < 0:
        raise ValueError("rmsd_threshold must be nonnegative.")

    # Build the molecular connectivity from the first conformer.
    first = conformers[0]
    symbols = first.get_chemical_symbols()
    xyz = f"{len(first)}\n\n" + "".join(
        f"{symbol} {x:.10f} {y:.10f} {z:.10f}\n"
        for symbol, (x, y, z) in zip(symbols, first.positions)
    )
    template = Chem.MolFromXYZBlock(xyz)
    rdDetermineBonds.DetermineBonds(template, charge=charge)

    unique_atoms = []
    unique_rdkit = []

    for atoms in conformers:
        if atoms.get_chemical_symbols() != symbols:
            raise ValueError("Conformers must have identical atom ordering.")

        mol = Chem.Mol(template)
        conf = mol.GetConformer()
        for i, position in enumerate(atoms.positions):
            conf.SetAtomPosition(i, tuple(map(float, position)))

        if heavy_atoms_only:
            mol = Chem.RemoveHs(mol)
        if mol.GetNumAtoms() == 0:
            raise ValueError("No atoms remain for RMSD comparison.")

        # GetBestRMS aligns its probe; use a copy for each comparison.
        duplicate = any(
            rdMolAlign.GetBestRMS(Chem.Mol(mol), reference)
            <= rmsd_threshold
            for reference in unique_rdkit
        )

        if not duplicate:
            unique_atoms.append(atoms)
            unique_rdkit.append(mol)

    return unique_atoms

