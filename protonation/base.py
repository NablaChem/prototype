# %%
import itertools as it
import rdkit
import rdkit.Chem.AllChem
import rdkit.Chem.rdDistGeom
import numpy as np
import pandas as pd
import random
import scipy.spatial.distance as ssd
import hmq


@hmq.task
def qm_calculation(
    atomspec: str,
    basis: str,
    charge: int,
    epsilon: float,
    geo_opt: bool,
    free_energy: bool,
    label,
) -> dict:
    import pyscf
    import pyscf.dft
    from pyscf.hessian import thermo
    import pyscf.solvent.pcm

    mol = pyscf.gto.M(atom=atomspec, basis=basis, charge=charge, verbose=0)
    mf = pyscf.dft.RKS(mol, xc="B3LYP").PCM()
    mf.eps = epsilon
    results = {"label": label}

    if geo_opt:
        mol = mf.Gradients().optimizer(solver="geomeTRIC").kernel()
        results["opt_geo"] = mol.atom_coords(unit="Ang")
        mf = pyscf.dft.RKS(mol, xc="B3LYP").PCM()
        mf.eps = epsilon

    if free_energy:
        mf.run()
        hessian = mf.Hessian().kernel()

        freq_info = thermo.harmonic_analysis(mf.mol, hessian)
        results["free_energy"] = thermo.thermo(
            mf, freq_info["freq_au"], 298.15, 101325
        )["G_tot"][0]

    return results


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

    def _build_atomspec(self, confid, protonation_state: list[int]):
        atomspec = []
        seen_hydrogens = 0
        for i, atom in enumerate(self._elements):
            if protonation_state and atom == "H":
                seen_hydrogens += 1
                if protonation_state[seen_hydrogens - 1] == 0:
                    continue
            atomspec.append((atom, self._conformers[confid][i]))
        return atomspec

    def optimize_conformers(self, epsilon: float) -> str:
        """Lifts conformer geometries to quantum chemistry level in solvent."""
        for conf in range(len(self._conformers)):
            atomspec = self._build_atomspec(conf, [1] * self.nprotons)
            qm_calculation(
                atomspec, self._basis, self._charge, epsilon, True, False, conf
            )
        tag = qm_calculation.submit()
        return tag.name

    def load_conformers(self, tag: str):
        tag = hmq.Tag.from_queue(tag)
        tag.pull(blocking=True)
        self._conformers = [None] * len(tag.results)
        for result in tag.results:
            conf = result["label"]
            self._conformers[conf] = result["opt_geo"]

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

    def get_free_energies(
        self, epsilon: float, all_k_body: int = None, rest_random_sample: int = None
    ):
        pstates = []
        # build list of all k-body protonation states
        if all_k_body is not None:
            pstates.append([1] * self.nprotons)

            for k in range(1, all_k_body + 1):
                for skipped in it.combinations(range(self.nprotons), k):
                    state = [1] * self.nprotons
                    for site in skipped:
                        state[site] = 0
                    pstates.append(state)

            # build list of random additional protonation states
            if rest_random_sample is not None:
                total_states = 2**self.nprotons
                included = len(pstates)
                nremaining = total_states - included
                if nremaining <= rest_random_sample:
                    # everything needs to be included
                    pstates = [
                        list(_) for _ in it.product((0, 1), repeat=self.nprotons)
                    ]
                elif nremaining < 10 * rest_random_sample:
                    # probably can be built in memory
                    all_states = list(it.product((0, 1), repeat=self.nprotons))
                    remaining = [_ for _ in all_states if _ not in pstates]
                    selected = random.sample(
                        range(len(remaining)), k=rest_random_sample
                    )
                    pstates += [list(remaining[_]) for _ in selected]
                else:
                    # may not fit in memory
                    fstring = "{0:0" + str(self.nprotons) + "b}"
                    while True:
                        to_choose = included + rest_random_sample - len(pstates)
                        if to_choose == 0:
                            break

                        # random numbers, then convert to binary == protonation state
                        selected = random.sample(range(2**self.nprotons), k=to_choose)
                        selected = [list(map(int, fstring.format(_))) for _ in selected]
                        selected = [_ for _ in selected if _ not in pstates]
                        pstates += selected
        else:
            if rest_random_sample is not None:
                raise NotImplementedError("Random sampling without all k-body states.")

            # build list of all protonation states
            pstates = [list(_) for _ in it.product((0, 1), repeat=self.nprotons)]

        for pstate in pstates:
            for confid in range(self.nconfomers):
                atomspec = self._build_atomspec(confid, pstate)
                nremoved = len(pstate) - sum(pstate)

                qm_calculation(
                    atomspec,
                    self._basis,
                    self._charge - nremoved,
                    epsilon,
                    True,
                    True,
                    (confid, pstate),
                )
        tag = qm_calculation.submit(ncores=4)
        return tag.name

    def load_free_energies(self, tag: str):
        tag = hmq.Tag.from_queue(tag)
        tag.pull()
        results = tag.results
        confids = [result["label"][0] for result in results]
        protonations = [result["label"][1] for result in results]
        geometries = [result["opt_geo"] for result in results]
        free_energies = [result["free_energy"] for result in results]
        kchanged = [len(p) - sum(p) for p in protonations]
        return pd.DataFrame(
            {
                "confid": confids,
                "protonation": protonations,
                "geometry": geometries,
                "free_energy": free_energies,
                "kchanged": kchanged,
            }
        )


s = System("C(C(O)=O)1CC[N+]([H])([H])C(C(=O)O)C1", "6-31G", 1)
# s = System("C=C", "6-31G", 0)
# s = System("Cc1ccc(cc1Nc2nccc(n2)c3cccnc3)NC(=O)c4ccc(cc4)CN5CCN(CC5)C", "6-31G", 0)
s.find_conformers()
#
# s.optimize_conformers(water_eps)
s.load_conformers("qm_calculation_1469f5d3-7a4b-41e8-9b33-ac86a9c984c6")
# results = s.get_free_energies(water_eps)

# %%
# "C(C(O)=O)1CC[N+]([H])([H])C(C(=O)O)C1", 6-31G, 1
# conformers: qm_calculation_1469f5d3-7a4b-41e8-9b33-ac86a9c984c6
# free energies: qm_calculation_10a16bb1-dcdb-4fac-9215-6e0be369c9c3
water_eps = 78.5
s.get_free_energies(water_eps, 2, 100)
# %%
