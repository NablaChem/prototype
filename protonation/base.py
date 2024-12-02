# %%
# Free energies of deprotonation in solvent of molecule M
# 1. Get conformers of M with a wide energy range: M_i, e.g. from xTB
# 2. Relax conformers in solvent
# 3. Enumerate protonation states: 2^n for n protons
# 4. For each protonation state, for each conformer: apply protonation state, relax conformer, get free energy
# ??? Choose reference protonation state for alchemy
from pyscf import gto
from pyscf.hessian import thermo

# from pyscf.solvent import pcm
# from pyscf import scf

# First compute nuclear Hessian.
mol = gto.M(
    atom="""O    0.   0.       0
              H    0.   -0.757   0.587
              H    0.    0.757   0.587""",
    basis="631g",
)

mf = mol.RHF().PCM()
mf.run()
hessian = mf.Hessian().kernel()

# Frequency analysis
freq_info = thermo.harmonic_analysis(mf.mol, hessian)
# Thermochemistry analysis at 298.15 K and 1 atmospheric pressure
thermo.thermo(mf, freq_info["freq_au"], 298.15, 101325)["G_tot"]

# %%
import itertools as it
import pyscf
import pyscf.dft
from pyscf.hessian import thermo
import pyscf.solvent.pcm
import rdkit
import rdkit.Chem.AllChem
import rdkit.Chem.rdDistGeom
import py3Dmol
import numpy as np
import scipy.spatial.distance as ssd


class System:
    def __init__(
        self,
        SMILES: str,
        basis: str,
        charge: int,
    ):
        self._basis = basis
        self._smiles = SMILES
        self._charge = charge

    def find_conformers(self):
        """Finds conformers using UFF."""
        mol = rdkit.Chem.MolFromSmiles(self._smiles)
        mol = rdkit.Chem.AddHs(mol)
        rdkit.Chem.rdDistGeom.EmbedMultipleConfs(mol, numConfs=100)

        # optimize all confs
        status = rdkit.Chem.AllChem.MMFFOptimizeMoleculeConfs(mol)

        self._elements = [atom.GetSymbol() for atom in mol.GetAtoms()]
        self._conformers = []
        for confid in range(mol.GetNumConformers()):
            # converged FF opt?
            if status[confid][0] == 0:
                self._conformers.append(
                    np.array(mol.GetConformer(confid).GetPositions())
                )
        self._remove_duplicate_conformers()
        print(f"Found {len(self._conformers)} conformers.")

    def _build_atomspec(self, confid, protonation_state: list[int] = None):
        atomspec = []
        seen_hydrogens = 0
        for i, atom in enumerate(self._elements):
            if protonation_state and atom == "H":
                seen_hydrogens += 1
                if protonation_state[seen_hydrogens - 1] == 0:
                    continue
            atomspec.append((atom, self._conformers[confid][i]))
        return atomspec

    def optimize_conformers(self):
        """Lifts conformer geometries to quantum chemistry level in solvent."""
        for conf in range(len(self._conformers)):
            atomspec = self._build_atomspec(conf)
            mol = pyscf.gto.M(
                atom=atomspec, basis=self._basis, charge=self._charge, verbose=0
            )
            mf = pyscf.dft.RKS(mol, xc="B3LYP").PCM()
            mol_opt = mf.Gradients().optimizer(solver="geomeTRIC").kernel()
            self._conformers[conf] = mol_opt.atom_coords(unit="Ang")
        self._remove_duplicate_conformers()
        print(f"Kept {len(self._conformers)} conformers after solvation.")

    def _remove_duplicate_conformers(self):
        kept = [self._conformers[0]]
        pdists = [ssd.pdist(kept[0])]
        for conf in self._conformers[1:]:
            pdist = ssd.pdist(conf)
            for ref in pdists:
                if np.abs(ref - pdist).max() < 5e-2:
                    break
            else:
                kept.append(conf)
                pdists.append(pdist)
        self._conformers = kept

    @property
    def nconfomers(self):
        return len(self._conformers)

    @property
    def nprotons(self):
        return len([atom for atom in self._elements if atom == "H"])

    def protonation_states(self):
        for i in it.product((0, 1), repeat=self.nprotons):
            yield i

    def canonical_calculation(
        self, confid: int, protonation_state: list[int], relax: bool
    ):
        atomspec = self._build_atomspec(confid, protonation_state)
        nremoved = len(protonation_state) - sum(protonation_state)
        mol = pyscf.gto.M(
            atom=atomspec, basis=self._basis, charge=self._charge - nremoved, verbose=0
        )

        mf = pyscf.dft.RKS(mol, xc="B3LYP").PCM()
        if relax:
            mol = mf.Gradients().optimizer(solver="geomeTRIC").kernel()
            mf = pyscf.dft.RKS(mol, xc="B3LYP").PCM()
        mf.run()
        hessian = mf.Hessian().kernel()

        freq_info = thermo.harmonic_analysis(mf.mol, hessian)
        return thermo.thermo(mf, freq_info["freq_au"], 298.15, 101325)["G_tot"][0]


# s = System("C(C(O)=O)1CC[N+]([H])([H])C(C(=O)O)C1", "6-31G", 1)
s = System("C=C", "6-31G", 0)
s.find_conformers()
s.optimize_conformers()


for pstate in s.protonation_states():
    print(pstate)
    for confid in range(s.nconfomers):
        print(s.canonical_calculation(confid, pstate, True))
#    base_free_energy = compute_free_energy(mol)
#    for conf in conformers:
#        conf = apply_protonation_state(conf, pstate)
#        conf = geo_opt(conf)
#        free_energy = compute_free_energy(conf)
#        delta_free_energy = free_energy - base_free_energy
# eq2 from paper
# save result


# %%
