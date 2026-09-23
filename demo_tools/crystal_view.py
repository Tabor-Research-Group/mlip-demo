"""Visualization helpers copied from the original molecular crystal demo."""
from ase import Atoms
from ase.data import covalent_radii
from ase.neighborlist import NeighborList
from ase.build.surface import add_vacuum
import numpy as np

from collections import deque, defaultdict

def build_bond_graph(atoms: Atoms, 
                     scale: float = 1.15, 
                     skin: float = 0.0,
                     periodic: bool = True
                     ) -> dict[int, list[tuple[int, np.ndarray]]]:
    """
    Build a bonded graph from distances under PBC.

    atoms : ase.Atoms object
    scale : float
        Bond criterion scale factor on covalent radii sum.
    skin : float
        ASE NeighborList skin.

    Returns
    -------
    graph : dict
        graph[i] = list of (j, S_ij)
        where S_ij is the integer lattice shift such that
        r_j + S_ij @ cell - r_i
        is the bonded minimum-image vector.
    """
    numbers = atoms.get_atomic_numbers()
    cell = atoms.cell.array

    # Pair cutoff = scale * (r_cov_i + r_cov_j)
    # NeighborList expects one radius per atom, and pairs interact if overlaps exist
    cutoffs = [scale * covalent_radii[z] for z in numbers]

    nl = NeighborList(cutoffs, skin=skin, bothways=True, self_interaction=False)
    nl.update(atoms)

    graph = defaultdict(list)

    
    for i in range(len(atoms)):
        indices, offsets = nl.get_neighbors(i)

        for j, S in zip(indices, offsets):
            if j <= i:
                continue

            if periodic:
                S = np.asarray(S, dtype=int)
                rij = atoms.positions[j] + S @ cell - atoms.positions[i]
            elif not periodic:
                rij = atoms.positions[j] - atoms.positions[i]

            dij = np.linalg.norm(rij)

            cutoff_ij = scale * (covalent_radii[numbers[i]] + covalent_radii[numbers[j]])

            if dij <= cutoff_ij:
                graph[i].append((j, S))
                graph[j].append((i, -S))

    return graph


def connected_components(graph: dict, 
                         n_atoms: int,
                         ) -> list[list[int]]:
    """
    graph : from build_bond_graph
    Find connected components in the bond graph.
    """
    seen = set()
    components = []

    for start in range(n_atoms):
        if start in seen:
            continue

        comp = []
        queue = deque([start])
        seen.add(start)

        while queue:
            i = queue.popleft()
            comp.append(i)

            for j, _ in graph.get(i, []):
                if j not in seen:
                    seen.add(j)
                    queue.append(j)

        components.append(comp)

    return components


def unwrap_component(atoms: Atoms, 
                     graph: dict, 
                     component: list[int]
                     ) -> tuple[dict[int, np.ndarray], np.ndarray]:
    """
    Assign integer lattice shifts n_i to atoms in one connected component
    so bonded neighbors become contiguous in Cartesian space.

    If atom i has shift n_i, then neighbor j gets:
        n_j = n_i + S_ij
    """
    cell = atoms.cell.array
    component_set = set(component)

    shifts = {}
    root = component[0]
    shifts[root] = np.zeros(3, dtype=int)

    queue = deque([root])

    while queue:
        i = queue.popleft()

        for j, Sij in graph.get(i, []):
            if j not in component_set:
                continue

            proposed = shifts[i] + Sij

            if j not in shifts:
                shifts[j] = proposed
                queue.append(j)
            else:
                # Ignore inconsistency here; usually means overconnected graph
                pass

    for i in component:
        if i not in shifts:
            shifts[i] = np.zeros(3, dtype=int)

    unwrapped = np.array([
        atoms.positions[i] + shifts[i] @ cell
        for i in component
    ])

    return shifts, unwrapped

def repair_fragmented_molecules(
    atoms: Atoms,
    bond_scale: float = 1.15,
    neighbor_skin: float = 0.0,
    centroid_position_frac: np.ndarray = None,
    # return_components=False,
    ) -> Atoms:
    """
    Repair molecules fragmented across periodic boundaries.

    Parameters
    ----------
    atoms : ase.Atoms
    bond_scale : float
        Bond criterion scale factor on covalent radii sum.
    neighbor_skin : float
        ASE NeighborList skin.
    
    Returns
    -------
    repaired : ase.Atoms
        New Atoms object with repaired positions.
    info : dict, optional
        Extra information if return_components=True
    """
    repaired = atoms.copy()
    
    graph = build_bond_graph(repaired, scale=bond_scale, skin=neighbor_skin)
    components = connected_components(graph, len(repaired))

    new_positions = repaired.positions.copy()
    
    for comp_idx, comp in enumerate(components):
        if len(comp) == 1:
            continue

        atom_shifts, unwrapped = unwrap_component(repaired, graph, comp)

        for k, atom_idx in enumerate(comp):
            new_positions[atom_idx] = unwrapped[k]

    repaired.positions[:] = new_positions

    if centroid_position_frac is not None:
        cell = repaired.cell.array
        centroid_cart = np.mean(repaired.positions, axis=0)
        centroid_frac = np.linalg.solve(cell.T, centroid_cart)
        shift_frac = centroid_position_frac - centroid_frac
        shift_cart = shift_frac @ cell
        repaired.positions += shift_cart

    return repaired

