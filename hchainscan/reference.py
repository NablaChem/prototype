#%%
import pyscf.gto as gto
import pyscf.scf as scf
import pyscf.dft as dft
import pyscf.data as data
import numpy as np
import enum as enum
import matplotlib.pyplot as plt
from collections import namedtuple
import scipy.signal as signal

Densityprofile = namedtuple("densityprofile", ["xs", "ys"])


class Method(enum.Enum):
    RHF = enum.auto()
    MP2 = enum.auto()
    CCSD = enum.auto()
    PBE0 = enum.auto()
    B3LYP = enum.auto()
    PBE = enum.auto()
    HSE06 = enum.auto()


class HydrogenChain:
    def __init__(self, atom_count: int, spacing: float, basis: str):
        if atom_count % 2 != 0:
            raise NotImplementedError()

        self.mol = gto.M(
            atom=[[1, (i * spacing, 0, 0)] for i in range(atom_count)], basis=basis
        )

        # build grid
        n_grid = 1000
        self._grid = np.zeros((n_grid, 3))
        self._grid[:, 0] = np.linspace(
            -2, (atom_count - 1) * spacing / data.nist.BOHR + 2, 1000
        )

        # obtain AO along grid
        self._ao_value = dft.numint.eval_ao(self.mol, self._grid, deriv=0)

    def get_density_profile(self, method: Method) -> Densityprofile:
        if method == Method.RHF:
            calc = scf.RHF(self.mol)
        elif method == Method.MP2:
            calc = scf.RHF(self.mol)
            calc.run()
            calc = calc.MP2()
        elif method == Method.CCSD:
            calc = scf.RHF(self.mol)
            calc.run()
            calc = calc.CCSD()
        elif method == Method.PBE0:
            calc = scf.RKS(self.mol)
            calc.xc = "PBE0"
        elif method == Method.B3LYP:
            calc = scf.RKS(self.mol)
            calc.xc = "B3LYP"
        elif method == Method.PBE:
            calc = scf.RKS(self.mol)
            calc.xc = "PBE"
        elif method == Method.HSE06:
            calc = scf.RKS(self.mol)
            calc.xc = "HSE06"
        else:
            raise NotImplementedError()

        calc.kernel()
        dm1_ao = calc.make_rdm1()
        if method == Method.MP2 or method == Method.CCSD:
            dm1_ao = np.einsum(
                "pi,ij,qj->pq", calc.mo_coeff, dm1_ao, calc.mo_coeff.conj()
            )

        rho = dft.numint.eval_rho(self.mol, self._ao_value, dm1_ao, xctype="LDA")
        return Densityprofile(self._grid[:, 0] * data.nist.BOHR, rho)


#%%
n_H = 4
chain = HydrogenChain(n_H, 1.1, "cc-pVDZ")
for method in Method:
    try:
        dp = chain.get_density_profile(method)
    except:
        raise
    plt.plot(dp.xs, dp.ys, label=method.name)
    plt.legend()
# %%
def scan_distance(n_H, spacings, basis, method):
    heights = []
    for spacing in spacings:
        chain = HydrogenChain(n_H, spacing, basis)
        dp = chain.get_density_profile(method)
        dips = signal.find_peaks(-dp.ys)[0]
        peaks = signal.find_peaks(dp.ys)[0]
        heights.append(dp.ys[peaks[n_H // 2 - 1]] - dp.ys[dips[n_H // 2 - 1]])
    return spacings, np.array(heights)


for method in (Method.RHF, Method.MP2, Method.CCSD):
    spacings, heights = scan_distance(4, np.linspace(0.5, 4, 8), "cc-pVDZ", method)
    plt.plot(spacings, heights, label=method.name)
plt.legend()
# %%
