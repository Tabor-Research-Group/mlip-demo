"""
Generate conformers for alanine dipeptide (Ace-Ala-NMe) with
`Psience.Molecools.Molecule`, round-trip them through an XYZ file and
`ase`, and visualize a structure with a metallic-looking x3d render.

Notes on how this was put together (so future-me doesn't have to
re-derive it):

- `Molecule.from_string(smi, fmt='smi', num_confs=N, return_multiconf=True)`
  dispatches to `Molecule._from_smiles` -> `RDMolecule.from_smiles`, which
  embeds `N` ETKDG conformers via RDKit. `take_min` (keep only the lowest
  energy structure) defaults to `None`, and `RDMolecule.from_base_mol`
  resolves `None` to `num_confs is None or conf_id is not None` -- since we
  pass an explicit `num_confs` and no `conf_id`, that resolves to `False`,
  so all `N` conformers are kept automatically. `return_multiconf=True`
  then stacks them into a single `Molecule` whose `.coords` has shape
  `(N, n_atoms, 3)` (in Bohr -- `Molecule` always stores Cartesians in
  atomic units internally).

- `Molecule._to_xyz_string` only knows how to format a single geometry
  (it zips `mol.atoms` against `mol.coords` directly), so a multi-conformer
  `Molecule` can't be exported in one call. `write_conformers_xyz` below
  just slices out each conformer with `mol.modify(coords=mol.coords[i])`
  and concatenates the per-frame XYZ blocks -- this is the standard
  "multi-model" XYZ layout (consecutive `natoms / comment / coords` blocks)
  that `ase.io.read(..., index=':')` already knows how to parse.

- `McUtils.ExternalPrograms.ASEMolecule.from_atoms(atoms)` wraps a plain
  `ase.Atoms` object, and `Molecule.from_ase(ase_mol)` reads its
  `.atoms`/`.coords`/`.meta` straight off of that wrapper (converting
  Angstroms -> Bohr), so `to_ase()`/`from_ase()` round-trip through
  `ASEMolecule` on both ends.

- For the "metallic" look: `Molecule.plot(..., backend='x3d')` builds
  X3D `Material`/`PhysicalMaterial` nodes for atoms/bonds. Passing
  `metallic`/`roughness` (optionally `specularity`) as *string* keys in
  `atom_style`/`bond_style` applies them globally (any string key in those
  dicts is layered into the base per-atom/per-bond style), and
  `X3DMaterial` maps `metallic` -> `metallicFactor` and `roughness` ->
  `roughnessFactor`, which routes the atom/bond spheres and cylinders
  through the PBR `PhysicalMaterial` node instead of the plain Phong
  `Material` node -- i.e. an actual metallic BRDF, not just a shininess
  hack. (There's also a top-level `reflectiveness` kwarg on `plot()` that
  derives `shininess`/`specularity` for you, but its scaling is inverted --
  `shininess = 100 * clip(1.1 - reflectiveness, 0, 1)` -- so *smaller*
  `reflectiveness` gives a *more* metallic/shiny result. Setting
  `metallic`/`roughness` directly is far less surprising, so that's what
  `visualize()` uses below.)
"""

import os

from Psience.Molecools import Molecule
from McUtils.ExternalPrograms import ASEMolecule

import ase.io

# N-acetyl-alanine-N'-methylamide ("Ace-Ala-NMe"), i.e. alanine dipeptide,
# the standard minimal model peptide used for phi/psi (Ramachandran)
# conformer studies.
ALANINE_DIPEPTIDE_SMILES = "CC(=O)NC(C)C(=O)NC"

DEFAULT_XYZ_FILE = "alanine_dipeptide_conformers.xyz"


def write_conformers_xyz(mol: Molecule, file: str, units="Angstroms") -> str:
    """
    Write every conformer held by a (multiconfig) `Molecule` to a single
    multi-frame XYZ file (consecutive per-conformer blocks), which `ase`
    can read back with `ase.io.read(file, index=':')`.

    :param mol: a `Molecule` whose `.coords` may hold multiple
        conformers (`mol.multiconfig`); a single-geometry `Molecule` is
        also accepted and written as a one-frame file
    :type mol: Molecule
    :param file: path to write the XYZ file to
    :type file: str
    :param units: units to write the coordinates in
    :type units: str
    :return: the path that was written
    :rtype: str
    """
    n_confs = len(mol)  # 1 if not mol.multiconfig, else mol.coords.shape[0]
    blocks = []
    for i in range(n_confs):
        coords = mol.coords[i] if mol.multiconfig else mol.coords
        frame = mol.modify(coords=coords)
        blocks.append(
            frame.to_string(
                "xyz",
                units=units,
                comment=f"{mol.name} conformer {i}"
            )
        )
    with open(file, "w+") as out:
        out.write("\n".join(blocks) + "\n")
    return file


def generate_conformers(
        n_conformers=100,
        smiles=ALANINE_DIPEPTIDE_SMILES,
        optimize=True,
        confgen_opts=None,
        out_file=DEFAULT_XYZ_FILE,
        **molecule_opts
):
    """
    Generate `n_conformers` conformers of alanine dipeptide with
    `Psience.Molecools.Molecule` (RDKit ETKDG embedding under the hood)
    and save them all to a single multi-frame XYZ file.

    :param n_conformers: number of conformers to embed
    :type n_conformers: int
    :param smiles: the SMILES string to build conformers for (defaults to
        alanine dipeptide, Ace-Ala-NMe)
    :type smiles: str
    :param optimize: force-field (MMFF) optimize each embedded conformer
    :type optimize: bool
    :param confgen_opts: extra options forwarded to the RDKit ETKDG
        embedder (e.g. `{'randomSeed': 42, 'pruneRmsThresh': .1}`)
    :type confgen_opts: dict | None
    :param out_file: path to write the multi-frame XYZ file to
    :type out_file: str
    :param molecule_opts: extra options forwarded to `Molecule.from_string`
    :return: `(mol, out_file)` -- the multiconfig `Molecule` (coords shaped
        `(n_conformers, n_atoms, 3)`) and the path the XYZ file was
        written to
    :rtype: tuple[Molecule, str]
    """
    mol = Molecule.from_string(
        smiles,
        fmt="smi",
        num_confs=n_conformers,
        return_multiconf=True,
        optimize=optimize,
        confgen_opts=confgen_opts,
        **molecule_opts
    )

    write_conformers_xyz(mol, out_file)

    return mol, out_file


def load_mols(file=DEFAULT_XYZ_FILE):
    """
    Load every conformer out of a multi-frame XYZ file (as written by
    `generate_conformers`/`write_conformers_xyz`) as `ase.Atoms` objects.

    :param file: path to the XYZ file to load
    :type file: str
    :return: the loaded structures, one `ase.Atoms` per conformer
    :rtype: list[ase.Atoms]
    """
    return ase.io.read(file, index=":")


def visualize(atoms, out_file=None, backend="x3d", metallic=0.2, roughness=0.25, **plot_opts):
    """
    Convert an `ase.Atoms` object into a `Molecule` (via
    `McUtils.ExternalPrograms.ASEMolecule` -> `Molecule.from_ase`) and plot
    it with a metallic-looking material on the `x3d` backend.

    :param atoms: the structure to visualize
    :type atoms: ase.Atoms
    :param out_file: if given, the figure is saved here (e.g. an `.html`
        path) instead of being shown interactively -- useful outside of a
        notebook
    :type out_file: str | None
    :param backend: the `Molecule.plot` rendering backend
    :type backend: str
    :param metallic: `metallicFactor` (0-1) applied to atoms/bonds; higher
        is more metal-like
    :type metallic: float
    :param roughness: `roughnessFactor` (0-1) applied to atoms/bonds;
        lower is glossier/more mirror-like
    :type roughness: float
    :param plot_opts: extra options forwarded to `Molecule.plot`
    :return: the constructed figure
    """
    ase_mol = ASEMolecule.from_atoms(atoms)
    mol = Molecule.from_ase(ase_mol)

    metal_style = {
        "metallic": metallic,
        "roughness": roughness,
        "specularity": "white",
    }

    figure = mol.plot(
        backend=backend,
        atom_style=metal_style,
        bond_style=metal_style,
        **plot_opts
    )

    if out_file is not None:
        figure.savefig(out_file)
    # else:
    #     figure.show()

    return figure


if __name__ == "__main__":
    mol, xyz_file = generate_conformers(100)
    print(f"wrote {len(mol)} conformers to {os.path.abspath(xyz_file)}")

    mols = load_mols(xyz_file)
    print(f"loaded {len(mols)} structures back in with ase")

    visualize(mols[0])
