# %%
from __future__ import annotations
import ase
import pysmiles
import networkx as nx
import numpy as np
import scipy.optimize as sco
import scine_molassembler as sm
import scine_utilities as su
from rdkit import Chem
from rdkit.Chem import rdDistGeom
from xtb.interface import Calculator, Param, XTBException
import xtb.libxtb


class System:
    """Holds a molecular system definition and provides interfaces to various molecular representations."""

    def __init__(self, adj: np.ndarray, elements: list[str]):
        self._adj = adj
        self._elements = elements
        self._Zs = np.array([ase.data.chemical_symbols.index(_) for _ in elements])

    @staticmethod
    def from_smiles(smiles: str) -> System:
        """Generator for a molecular system from a SMILES string.

        Parameters
        ----------
        smiles : str
            SMILES string.

        Returns
        -------
        System
            Parsed molecular system.
        """
        G = pysmiles.read_smiles(
            smiles,
            explicit_hydrogen=True,
            zero_order_bonds=True,
            reinterpret_aromatic=True,
        )  # graph in networkx
        E = nx.get_node_attributes(G, name="element")  # attributes of graph nodes
        A, _ = nx.attr_matrix(
            G,
            edge_attr=None,
            node_attr=None,
            normalized=False,
            rc_order=None,
            dtype=None,
            order=None,
        )  # adjacency matrix
        return System(A.astype(int), [E[i] for i in range(len(A))])

    def to_XYZ_content(self, coords: np.ndarray) -> str:
        """Generates the content of an XYZ file.

        Parameters
        ----------
        coords : np.ndarray
            Molecular geometry.

        Returns
        -------
        str
            File contents.
        """
        content = f"{self.natoms}\n\n"
        for element, pos in zip(self._elements, coords):
            content += f"{element} {' '.join(map(str, pos))}\n"
        return content

    @property
    def radii(self) -> np.ndarray:
        """Property for atomic radii.

        Returns
        -------
        np.ndarray
            Radii.
        """
        return np.array([ase.data.vdw_radii[_] for _ in self._Zs])

    @property
    def natoms(self) -> int:
        """Number of atoms.

        Returns
        -------
        int
            Number of atoms.
        """
        return len(self._elements)

    @property
    def graph(self) -> nx.Graph:
        """NetworkX graph representation.

        Returns
        -------
        nx.Graph
            Molecular graph.
        """
        return nx.from_numpy_array(self._adj, parallel_edges=True)

    @property
    def elements(self) -> list[str]:
        """Elements

        Returns
        -------
        list[str]
            Element labels
        """
        return self._elements

    @property
    def charges(self) -> np.ndarray:
        """Nuclear charges.

        Returns
        -------
        np.ndarray
            Nuclear charges.
        """
        return self._Zs

    @property
    def rdkit(self) -> Chem.Mol:
        """Converts to a RDKit molecule.

        Returns
        -------
        Chem.Mol
            New molecule.
        """
        mol = Chem.RWMol()
        atoms = [mol.AddAtom(Chem.Atom(_)) for _ in self._elements]
        for atom in atoms:
            mol.GetAtomWithIdx(atom).SetNoImplicit(True)
        for i in range(self.natoms):
            for j in range(i + 1, self.natoms):
                bondorder = [
                    None,
                    Chem.rdchem.BondType.SINGLE,
                    Chem.rdchem.BondType.DOUBLE,
                    Chem.rdchem.BondType.TRIPLE,
                    Chem.rdchem.BondType.QUADRUPLE,
                ][self._adj[i, j]]
                if bondorder is not None:
                    mol.AddBond(atoms[i], atoms[j], bondorder)
        mol.UpdatePropertyCache()
        return mol.GetMol()

    @property
    def molassembler(self) -> sm.Molecule:
        """Converts to a Molassembler molecule.

        Returns
        -------
        sm.Molecule
            New Molecule.
        """

        if self.natoms == 0:
            raise ValueError("Cannot create a molecule with no atoms for molassembler")

        # workaround to fix the atom ordering in the molecule to match the adjacency matrix
        mol = sm.Molecule()  # defaults to H2
        mol.remove_atom(0)  # leaves only H
        # set first atom to be of correct element
        mol.set_element_type(0, getattr(su.ElementType, self._elements[0]))
        # create all atoms, and connect them to the first atom (since disconnected graphs are not supported even temporarily)
        for element in self._elements[1:]:
            mol.add_atom(getattr(su.ElementType, element), 0, sm.BondType.Single)

        for i in range(self.natoms):
            for j in range(i + 1, self.natoms):
                bondorder = [
                    None,
                    sm.BondType.Single,
                    sm.BondType.Double,
                    sm.BondType.Triple,
                    sm.BondType.Quadruple,
                ][self._adj[i, j]]
                if bondorder is not None and i != 0:
                    mol.add_bond(i, j, bondorder)

        # remove erroneous bonds to atom 0
        for j in range(1, self.natoms):
            if self._adj[0, j] == 0:
                mol.remove_bond(0, j)
        return mol


class BasePotential: ...


class SimplePotential(BasePotential):
    """Universal simple Lennard-Jones potential.

    Implements https://doi.org/10.25950/962b4967
    """

    db = {
        "H": (4.47789, 0.552357),
        "He": (0.0009421, 0.498903),
        "Li": (1.04969, 2.2807),
        "Be": (0.572942, 1.71053),
        "B": (2.96703, 1.51453),
        "C": (6.36953, 1.35417),
        "N": (9.75379, 1.26508),
        "O": (5.12647, 1.17599),
        "F": (1.60592, 1.01562),
        "Ne": (0.0036471, 1.03344),
        "Na": (0.736745, 2.95778),
        "Mg": (0.0785788, 2.51233),
        "Al": (2.70067, 2.15597),
        "Si": (3.17431, 1.9778),
        "P": (5.0305, 1.90652),
        "S": (4.36927, 1.87089),
        "Cl": (4.48328, 1.81743),
        "Ar": (0.0123529, 1.88871),
        "K": (0.551799, 3.61705),
        "Ca": (0.132679, 3.13596),
        "Sc": (1.6508, 3.02906),
        "Ti": (1.18027, 2.85088),
        "V": (2.75249, 2.72615),
        "Cr": (1.53679, 2.4767),
        "Mn": (0.599888, 2.4767),
        "Fe": (1.18442, 2.35197),
        "Co": (1.27769, 2.24506),
        "Ni": (2.07572, 2.20943),
        "Cu": (2.04463, 2.35197),
        "Zn": (0.191546, 2.17379),
        "Ga": (1.0642, 2.17379),
        "Ge": (2.70171, 2.13816),
        "As": (3.9599, 2.12034),
        "Se": (3.38677, 2.13816),
        "Br": (1.97063, 2.13816),
        "Kr": (0.0173276, 2.06689),
        "Rb": (0.468265, 3.91995),
        "Sr": (0.133923, 3.47451),
        "Y": (2.75975, 3.38542),
        "Zr": (3.05201, 3.11815),
        "Nb": (5.2782, 2.92215),
        "Mo": (4.47499, 2.74397),
        "Tc": (3.38159, 2.61924),
        "Ru": (1.96172, 2.60142),
        "Rh": (2.40582, 2.53015),
        "Pd": (1.37097, 2.4767),
        "Ag": (1.64976, 2.58361),
        "Cd": (0.0377447, 2.56579),
        "In": (0.811314, 2.53015),
        "Sn": (1.90057, 2.4767),
        "Sb": (3.08828, 2.4767),
        "Te": (2.63123, 2.45888),
        "I": (1.53938, 2.4767),
        "Xe": (0.023888, 2.49452),
        "Cs": (0.416642, 4.34759),
        "Ba": (1.9, 3.83086),
        "La": (2.49961, 3.68832),
        "Ce": (2.57008, 3.63487),
        "Pr": (1.29946, 3.61705),
        "Nd": (0.819605, 3.58141),
        "Pm": (3.24134, 3.54578),
        "Sm": (0.521122, 3.52796),
        "Eu": (0.429918, 3.52796),
        "Gd": (2.09956, 3.49232),
        "Tb": (1.39999, 3.45669),
        "Dy": (0.690055, 3.42105),
        "Ho": (0.690055, 3.42105),
        "Er": (0.738766, 3.3676),
        "Tm": (0.521122, 3.38542),
        "Yb": (0.130399, 3.33196),
        "Lu": (1.43315, 3.33196),
        "Hf": (3.36086, 3.11815),
        "Ta": (4.00343, 3.02906),
        "W": (6.86389, 2.88651),
        "Re": (4.43871, 2.69051),
        "Os": (4.26253, 2.56579),
        "Ir": (3.70287, 2.51233),
        "Pt": (3.1401, 2.42324),
        "Au": (2.3058, 2.42324),
        "Hg": (0.045414, 2.35197),
        "Tl": (0.577087, 2.58361),
        "Pb": (0.858988, 2.60142),
        "Bi": (2.07987, 2.63706),
        "Po": (1.89953, 2.49452),
        "At": (1.385442, 2.6727),
        "Rn": (0.0214992, 2.6727),
        "Fr": (0.3749778, 4.63267),
        "Ra": (1.71, 3.93777),
        "Ac": (2.249649, 3.83086),
        "Th": (2.313072, 3.6705),
        "Pa": (1.169514, 3.56359),
        "U": (0.7376445, 3.49232),
        "Np": (2.917206, 3.38542),
        "Pu": (0.4690098, 3.33196),
        "Am": (0.3869262, 3.20724),
        "Cm": (1.889604, 3.01124),
        "Bk": (1.259991, 2.99342),
        "Cf": (0.6210495, 2.99342),
        "Es": (0.6210495, 2.93997),
        "Fm": (0.6648894, 2.9756),
        "Md": (0.4690098, 3.08251),
        "No": (0.1173591, 3.13596),
        "Lr": (1.289835, 2.86869),
        "Rf": (3.024774, 2.79742),
        "Db": (3.603087, 2.65488),
        "Sg": (6.177501, 2.54797),
        "Bh": (3.994839, 2.51233),
        "Hs": (3.836277, 2.38761),
        "Mt": (3.332583, 2.29852),
        "Ds": (2.82609, 2.2807),
        "Rg": (2.07522, 2.15597),
        "Cn": (0.0408726, 2.17379),
        "Nh": (0.5193783, 2.42324),
        "Fl": (0.7730892, 2.54797),
        "Mc": (1.871883, 2.88651),
        "Lv": (1.709577, 3.11815),
        "Ts": (1.2468978, 2.93997),
        "Og": (0.0193493, 2.79742),
    }

    def __init__(self, system: System):
        self._system = System

        # Lorentz-Berthelot
        sigmas = np.array([SimplePotential.db[_][1] for _ in system.elements])
        self._sigmas = (sigmas[:, None] + sigmas) / 2
        np.fill_diagonal(self._sigmas, 0)
        self._sigmas[system._adj == 0] = 0

        epsilons = np.array([SimplePotential.db[_][0] for _ in system.elements])
        self._epsilons = np.sqrt(np.outer(epsilons, epsilons))

    def optimize(self, configuration: np.ndarray):
        res = sco.minimize(
            lambda _: self.single_point(_.reshape(-1, 3)),
            configuration.reshape(-1),
            method="L-BFGS-B",
        )
        return res.x.reshape(-1, 3)

    def single_point(self, configuration: np.ndarray) -> float:
        """Calculates the energy of a molecular geometry.

        Parameters
        ----------
        configuration : np.ndarray
            Molecular geometry.

        Returns
        -------
        float
            Energy.
        """
        ds = np.linalg.norm(configuration[:, np.newaxis] - configuration, axis=-1)

        np.fill_diagonal(ds, 1)
        frac = (self._sigmas / ds) ** 6
        return np.sum(self._epsilons * (frac**2 - frac))


class XTBPotential(BasePotential):
    """Common interface for xTB potentials."""

    def __init__(self, system: System):
        self._system = system

    def single_point(self, configuration: np.ndarray) -> float:
        """Calculates the xTB energy of a molecular geometry.

        Parameters
        ----------
        configuration : np.ndarray
            Geometry.

        Returns
        -------
        float
            Energy.
        """
        calc = Calculator(
            Param.GFN2xTB,
            self._system.charges,
            np.array(configuration) / ase.units.Bohr,
        )
        calc.set_max_iterations(30)
        calc.set_verbosity(xtb.libxtb.VERBOSITY_MUTED)
        try:
            res = calc.singlepoint()
        except XTBException:
            return 1e10
        return res.get_energy()


def guess_random(system: System) -> np.ndarray:
    """Randomly place atoms in a box.

    Parameters
    ----------
    system : System
        A molecule specification.

    Returns
    -------
    np.ndarray
        A molecular geometry.
    """
    atomic_volumes = 4 * np.pi * np.array(system.radii) * 3 / 3
    ball_volume = (
        atomic_volumes.sum() / 0.75
    )  # adjust for packing efficiency of spheres
    box_length = ball_volume ** (1 / 3)
    return np.random.uniform(low=0.0, high=box_length, size=(system.natoms, 3))


def _fruchterman_reingold(system: System, dims: int) -> np.ndarray:
    """Spring-based embedding.

    Parameters
    ----------
    system : System
        A molecule specification.
    dims : int
        Number of dimensions.

    Returns
    -------
    np.ndarray
        Molecular geometry.
    """
    pos = np.zeros((system.natoms, 3))
    for idx, coords in nx.fruchterman_reingold_layout(
        system.graph, k=1.2, dim=dims
    ).items():
        pos[idx, :dims] = coords
    return pos


def fruchterman_reingold_2d(system: System) -> np.ndarray:
    """Wrapper for 2D geometries.

    Parameters
    ----------
    system : System
        Molecule specification

    Returns
    -------
    np.ndarray
        2D geometry.
    """
    return _fruchterman_reingold(system, 2)


def fruchterman_reingold_3d(system: System) -> np.ndarray:
    """Wrapper for 3D geometries.

    Parameters
    ----------
    system : System
        Specification of the molecule

    Returns
    -------
    np.ndarray
        A 3D geometry.
    """
    return _fruchterman_reingold(system, 3)


# Holds a list of tools to generate initial geometries
_tools = (guess_random, fruchterman_reingold_2d, fruchterman_reingold_3d)


def compare_rdkit_etkdg(mol: Chem.Mol) -> np.ndarray:
    """Obtain a conformation from RDKit's ETKDG.

    Parameters
    ----------
    mol : Chem.Mol
        A RDKit molecule data structure.

    Returns
    -------
    np.ndarray
        Molecular geometry.
    """
    ps = rdDistGeom.ETKDGv3()
    conf_id = rdDistGeom.EmbedMolecule(mol, ps)
    return mol.GetConformer(conf_id).GetPositions()


def compare_molassembler(mol: sm.Molecule) -> np.ndarray:
    """Obtain a conformation from the Molassembler library.

    Parameters
    ----------
    mol : sm.Molecule
        A molassembler molecule data structure.

    Returns
    -------
    np.ndarray
        Molecular geometry.
    """
    conformation = sm.dg.generate_conformation(mol, 1)
    if isinstance(conformation, sm.dg.Error):
        return None
    return conformation * ase.units.Bohr


def transfer_potentials(
    pot1: BasePotential, pot2: BasePotential, conf1: np.ndarray, steps: int
) -> np.ndarray:
    """Transfers a geometry from a local minimum of one potential to a local minimum of another potential.

    Parameters
    ----------
    pot1 : BasePotential
        The first potential.
    pot2 : BasePotential
        The second potential (the one to which the geometry is transferred).
    conf1 : np.ndarray
        Molecular geometry.
    steps : int
        The number of steps to take in the interpolation. More steps are slower but more stable.

    Returns
    -------
    np.ndarray
        The new molecular geometry.
    """
    mixings = 1 - np.linspace(0, 1, steps, endpoint=False)[::-1]

    conf = conf1.copy()
    for mixing in mixings:
        res = sco.minimize(
            lambda _: pot1.single_point(_.reshape(-1, 3)) * (1 - mixing)
            + pot2.single_point(_.reshape(-1, 3)) * mixing,
            conf.reshape(-1),
            method="L-BFGS-B",
        )
        conf = res.x.reshape(-1, 3)
    return conf


def graph2geometry(smiles: str, ntries: int):
    """Takes a molecular graph and returns a geometry.

    Parameters
    ----------
    smiles : str
        Graph in SMILES format.
    ntries : int
        How hard to try to find a geometry.
    """
    system = System.from_smiles(smiles)
    robust_potential = SimplePotential(system)
    accurate_potential = XTBPotential(system)

    configurations = [
        compare_rdkit_etkdg(system.rdkit),
        compare_molassembler(system.molassembler),
    ]
    configurations = [
        transfer_potentials(accurate_potential, accurate_potential, _, 1)
        for _ in configurations
    ]
    # TODO: check whether configurations are a minimum: if yes, stop here
    # definition minimum: first order deriv almost 0, second order deriv positive

    # TODO: check whether bond perception yields the same adjacency matrix that was put in
    # otherwise: ignore configuration

    for _ in range(ntries):
        for tool in _tools:
            robust = robust_potential.optimize(tool(system))

            configurations.append(
                transfer_potentials(robust_potential, accurate_potential, robust, 3)
            )

    energies = [accurate_potential.single_point(_) for _ in configurations]
    return energies, configurations


print(graph2geometry("[C-]#[O+]", 1))
# print(graph2geometry("C1C23CC2C=13", 0))
# system = System.from_smiles("[C-]#[O+]")
# print(system.to_XYZ_content(compare_rdkit_etkdg(system.rdkit)))
# print(system.natoms)
# pos = compare_rdkit_etkdg("C1(C2(CC2)C1)1C2(CC2)C1")
# p = XTBPotential()


# %%
