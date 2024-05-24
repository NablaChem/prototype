# %%
import numpy as np


def read_reference(filename):
    with open(filename) as fh:
        lines = fh.readlines()

    # atomspec
    natoms = 15
    atomspec = ";".join([_.strip() for _ in lines[2 : 2 + natoms]])

    # MBS
    started = False
    ref = np.zeros((natoms, natoms))
    for line in lines:
        if "MBS Condensed to atoms (all electrons)" in line:
            started = True
            continue
        if started:
            if " " * 12 in line:
                # header
                section_indices = [int(_) - 1 for _ in line.strip().split()]
                continue
            if len(line.strip()) == 0:
                break
            parts = line.strip().split()
            row_index = int(parts[0]) - 1
            for col_index, value in zip(section_indices, parts[2:]):
                ref[row_index, col_index] = float(value)

    return atomspec, ref


atomspec, refdata = read_reference(
    "/home/ferchault/wrk/prototype/minpop/dsgdb9nsd_001000.txt"
)
# %%
import pyscf
import numpy as np
import pyscf.lo
from pyscf.scf import addons
from pyscf.scf import hf
import pyscf.gto
import scipy.linalg


def minpop(atomspec, basis):
    mollow = pyscf.gto.M(atom=atomspec, basis="STO-3G")
    mflow = pyscf.scf.RHF(mollow)
    # mflow.kernel()

    molhigh = pyscf.gto.M(atom=atomspec, basis=basis)
    mfhigh = pyscf.scf.RHF(molhigh)
    mfhigh.kernel()

    Sbar = pyscf.gto.intor_cross("int1e_ovlp", mollow, molhigh)
    C = mfhigh.mo_coeff[:, mfhigh.mo_occ > 0]
    Sprime = mflow.get_ovlp()

    P = Sbar @ C
    PL = scipy.linalg.sqrtm(np.linalg.inv(Sprime)) @ P
    Sprimeinv = np.linalg.inv(Sprime)
    Cprime = PL @ scipy.linalg.sqrtm(np.linalg.inv(C.T @ Sbar.T @ Sprimeinv @ Sbar @ C))

    # literal expression from DOI 10.1063/1.481224, same result
    # sprimeinv12 = scipy.linalg.sqrtm(np.linalg.inv(sprime))
    # Cprime = sprimeinv12 @ Sbar @ C @ scipy.linalg.sqrtm(np.linalg.inv(C.T@Sbar.T @Sprimeinv @sbar@C))

    pm = pyscf.lo.PM(mollow, Cprime, mflow)
    # pm.pop_method = "mulliken"
    loc_orb = pm.kernel()
    mo_in_ao = loc_orb

    s = hf.get_ovlp(mollow)
    print(mfhigh.mo_occ[mfhigh.mo_occ > 0])
    dm = mflow.make_rdm1(mo_in_ao, mfhigh.mo_occ[mfhigh.mo_occ > 0])
    pop = np.einsum("ij,ji->ij", dm, s).real

    population = np.zeros((mollow.natm, mollow.natm))
    for i, si in enumerate(mollow.ao_labels(fmt=None)):
        for j, sj in enumerate(mollow.ao_labels(fmt=None)):
            population[si[0], sj[0]] += pop[i, j]

    return population


custom = minpop(atomspec, "6-31+G")

# %%
###### do not use: tries to use the high-level functions in pyscf, but that would be
###### inconvenient for (automatic) differentiability
import pyscf
import numpy as np
import pyscf.lo
from pyscf.scf import addons
from pyscf.scf import hf


def populations(atomspec, minimal_basis, high_basis):
    mollow = pyscf.gto.M(atom=atomspec, basis=minimal_basis)
    mflow = pyscf.scf.RHF(mollow)
    mflow.kernel()

    molhigh = pyscf.gto.M(atom=atomspec, basis=high_basis)
    mfhigh = pyscf.scf.RHF(molhigh)
    mfhigh.kernel()

    def getpop(mol, mf):
        pm = pyscf.lo.PM(mol, mf.mo_coeff[:, mf.mo_occ > 0], mf)
        pm.pop_method = "mulliken"
        loc_orb = pm.kernel()
        return loc_orb

    def getpop_on_low(mollow, mflow, molhigh, mfhigh):
        s = hf.get_ovlp(mollow)
        mo = addons.project_mo_nr2nr(
            molhigh, mfhigh.mo_coeff[:, mfhigh.mo_occ > 0], mollow
        )
        norm = np.einsum("pi,pi->i", mo.conj(), s.dot(mo))
        mo /= np.sqrt(norm)
        pm = pyscf.lo.PM(mollow, mo, mflow)
        pm.pop_method = "mulliken"
        loc_orb = pm.kernel()
        return loc_orb

    return getpop(mollow, mflow).T  # , getpop_on_low(mollow, mflow, molhigh, mfhigh).T
