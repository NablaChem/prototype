# %%
import pyscf
from pyscf.cc.ccd import CCD
import numpy as np


def single_point_dimer(lambdaval):
    mol = pyscf.gto.M(atom="Li 0 0 0; H 0 0 1.6", basis="sto-6g")

    # build perturbation to Hamiltonian
    Hprime = 0
    for i, delta_Z in enumerate([1, -1]):
        mol.set_rinv_orig_(mol.atom_coords()[i])
        Hprime -= delta_Z * mol.intor("int1e_rinv")

    # correct NN energy (optional)
    d = np.linalg.norm((mol.atom_coord(0) - mol.atom_coord(1)))
    dnn = ((3 + lambdaval) * (1 - lambdaval) / d) - mol.energy_nuc()

    # patch Hamiltonian in RHF
    mf = pyscf.scf.RHF(mol)
    hcore = mf.get_hcore()
    mf.get_hcore = lambda *args, **kwargs: hcore + Hprime * lambdaval
    mf.kernel()
    hf_energy = mf.e_tot + dnn

    # patch Hamiltonian in CCSD
    ccsd_energy = mf.CCSD().kernel()[0]
    ccd_energy = CCD(mf).kernel()[0]

    return hf_energy, ccsd_energy, ccd_energy


xs = np.linspace(-1, 1, 20)
ys = [single_point_dimer(x) for x in xs]

# %%
import matplotlib.pyplot as plt

f, axs = plt.subplots(3, 1, figsize=(5, 10), sharex=True)
for i in range(3):
    axs[i].plot(xs, [y[i] for y in ys], label=["HF", "CCSD", "CCD"][i])
    axs[i].legend()
    axs[i].set_xlabel(r"$\lambda$")
    axs[i].set_ylabel("Energy [a.u.]")
