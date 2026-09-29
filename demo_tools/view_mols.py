"""View ASE structures with Psience's X3D molecular plotter."""

import numpy as np
from ase import Atoms
from McUtils.Data import UnitsData
from McUtils.ExternalPrograms import ASEMolecule
from Psience.Molecools import Molecule

default_visualization_styles = dict(
    background='white',
    navigation={'headlight': True},
    environment={'reflectionIntensity': 1.0},
    atom_style={'ambientIntensity': 0.2, 'specularColor': (0.38, 0.38, 0.38), 'shininess': 0.32},
    bond_style={'ambientIntensity': 0.2, 'specularColor': (0.38, 0.38, 0.38), 'shininess': 0.32},
    # lighting=[
    #     {'direction': (-0.45, -0.65, -1.0), 'color': (1.0, 0.98, 0.95),
    #      'intensity': 0.2, 'ambientIntensity': 0.16, 'global': True},
    #     {'direction': (0.70, 0.20, -1.0), 'color': (0.90, 0.94, 1.0),
    #      'intensity': 0.2, 'ambientIntensity': 0.05, 'global': True},
    # ],
)


def combine_bonds(mol_list):
    """Offset each molecule's existing bond indices for a combined molecule."""
    offset = 0
    bonds = []
    for mol in mol_list:
        bonds.extend(
            [bond[0] + offset, bond[1] + offset, *bond[2:]]
            for bond in mol.bonds
        )
        offset += len(mol.atoms)
    return bonds


def combine_molecules(mol_list):
    """Combine Psience molecules while reusing their already determined bonds."""
    return Molecule(
        atoms=[atom for mol in mol_list for atom in mol.atoms],
        coords=np.concatenate([np.asarray(mol.coords) for mol in mol_list]),
        bonds=combine_bonds(mol_list),
    )


# Keep the name used by the original demo helper.
combine_mols = combine_molecules


def view(structures, *, combine=False, animate=False, reuse_bonds=False,
         out_file=None, **plot_options):
    """Plot ASE structures through ``Molecule.plot`` on the X3D backend.

    A single ``Atoms`` returns one figure. A list returns one figure per
    structure, unless ``combine=True`` requests one overlaid structure or
    ``animate=True`` requests one trajectory. ``reuse_bonds=True`` avoids
    repeating bond detection for conformers with the same atom ordering.
    """
    def convert(atoms):
        return Molecule.from_ase(ASEMolecule.from_atoms(atoms))

    def finish(figure):
        if out_file is not None:
            figure.savefig(out_file)
        return figure

    options = {**default_visualization_styles, **plot_options,
               "backend": "x3d", "return_objects": False}
    if isinstance(structures, Atoms):
        return finish(convert(structures).plot(**options))

    structures = list(structures)
    if not structures:
        raise ValueError("At least one ASE Atoms object is required")
    if animate:
        if combine:
            raise ValueError("Choose either animate=True or combine=True")
        # The first molecule supplies the bond topology for every frame.
        first = convert(structures[0])
        frames = np.stack([atoms.positions for atoms in structures])
        frames *= UnitsData.convert('Angstroms', 'BohrRadius')
        return finish(first.plot(frames, animate=True, **options))

    molecules = [convert(structures[0])]
    for atoms in structures[1:]:
        mol = convert(atoms)
        if reuse_bonds:
            mol.bonds = molecules[0].bonds
        molecules.append(mol)
    if combine:
        return finish(combine_molecules(molecules).plot(**options))
    if out_file is not None:
        raise ValueError("out_file requires one structure, combine=True, or animate=True")
    return [mol.plot(**options) for mol in molecules]
