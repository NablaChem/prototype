#%%
import pyscf.scf as scf
import pyscf.gto as gto
import numpy as np

#%%
def get_molecule(distance: float) -> gto.Mole:
    mol = gto.M(atom=f"H 0 0 0; F 0 0 {distance}", basis="ccpvdz")
    return mol


def get_energy(mol: gto.Mole) -> float:
    mol.verbose = 0
    mf = scf.RHF(mol)
    mf.kernel()
    return mf.e_tot


get_energy(get_molecule(1.0))
# %%
# finite differences
def central_finite_differences(distance: float, delta: float) -> float:
    return (
        get_energy(get_molecule(distance + delta))
        - get_energy(get_molecule(distance - delta))
    ) / (2 * delta)


def forward_finite_differences(distance: float, delta: float) -> float:
    return (
        get_energy(get_molecule(distance + delta)) - get_energy(get_molecule(distance))
    ) / (delta)


deltas = 10.0 ** np.linspace(-15, -1, 50)
cfd_values = [central_finite_differences(1.0, delta) for delta in deltas]
ffd_values = [forward_finite_differences(1.0, delta) for delta in deltas]

# %%
mol = get_molecule(1.0)
calc = scf.RHF(mol)
calc.kernel()
calc = calc.nuc_grad_method()
grad = calc.kernel()
refgrad = grad[1, 2] / 0.52917721092
# %%
import matplotlib.pyplot as plt

plt.loglog(deltas, np.abs(cfd_values - refgrad), label="central finite differences")
plt.loglog(deltas, np.abs(ffd_values - refgrad), label="forward finite differences")
plt.legend()
plt.title("H-F finite differences vs reference gradient @ 1.0 Angstrom, RHF/cc-pVDZ")
plt.xlabel("delta")
plt.ylabel("absolute error")

#%%
def LJ_energy(sigma1, sigma2, epsilon1, epsilon2, distance):
    sigma = (sigma1 + sigma2) / 2
    epsilon = jnp.sqrt(epsilon1 * epsilon2)
    sig_dist = (sigma / distance) ** 6
    energy = 4 * epsilon * (sig_dist**2 - sig_dist)
    return energy


import jax
import jax.numpy as jnp

ljg = jax.grad(LJ_energy, argnums=2)
# %%
import numpy as np
import matplotlib.pyplot as plt

sigma1 = 0.1
sigma2 = 0.01

epsilon1 = 2
epsilon2 = 3

r = np.linspace(0.01, 0.5, 250)

plot_gradient = []
for i in r:
    plot_gradient.append(ljg(sigma1, sigma2, epsilon1, epsilon2, i) / 75)

plt.plot(r, LJ_energy(sigma1, sigma2, epsilon1, epsilon2, r), label="LJ-Potential")
plt.plot(r, plot_gradient, label="LJ-Gradient")
plt.ylim(-2, 2)
plt.legend()
# %%
from mpmath import mp
import mpmath
import numpy as np

mp.dps = 2

mp.mpf("1") / mp.mpf("3")
# %%
mp.dps = 15


def testcase(x):
    return mp.sin(x)


def actual_derivative(x):
    return np.cos(x)


at = 0.2
mpmath.diff(testcase, at, n=1) - actual_derivative(at)

# %%
mp.dps = 100
mpmath.quad(lambda x: mp.sin(x), [0, 2 * mp.pi])


# %%
import pyscf.scf as scf
import pyscf.gto as gto
import numpy as np

mol = gto.M(atom="H 0 0 0; F 0 0 1", basis="ccpvdz")
mol.verbose = 0
calc = scf.RHF(mol)
calc.kernel()

from pyscf.hessian.rhf import Hessian

hess = Hessian(calc)
hess.kernel()
# %%
hess.kernel().shape
# %%
def get_E(d: float) -> float:
    mol = gto.M(atom=f"H 0 0 {d}; F 0 0 1", basis="ccpvdz")
    mol.verbose = 0
    calc = scf.RHF(mol)
    calc.kernel()
    return calc.e_tot


A = get_E(0.0)
B = get_E(0.01)
C = get_E(-0.01)


# %%
(C + B - 2 * A) / ((0.01 * 1.88) ** 2)
# %%
hess.kernel()[0, 0, 2, 2]
# %%
