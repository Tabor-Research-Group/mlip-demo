import numpy as np
from Psience.Molecools import Molecule

def combine_bonds(mol_list):
    offset = 0
    bonds = []
    for m in mol_list:
        bb = [
            [b[0] + offset, b[1] + offset, b[2]]
            for b in m.bonds
        ]
        bonds.extend(bb)
        offset += len(m.atoms)
    return bonds

def combine_mols(mol_list):
    comb_mol = Molecule(
        atoms=sum((m.atoms for m in mol_list), ()),
        coords=np.concatenate(
            [m.coords for m in mol_list]
        ),
        bonds=combine_bonds(mol_list)
    )
    return comb_mol