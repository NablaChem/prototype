#%%
import pyscf.gto as gto
import pyscf.scf as scf
import pyscf.dft as dft
import numpy as np
import scipy.signal as signal


def density_difference(
    functional: str, basis: str, hydrogen_count: int, spacing: float
):
    # setup molecule
    atoms = []
    for i in range(hydrogen_count):
        atoms.append(["H", (i * spacing, 0, 0)])
    mol = gto.M(atom=atoms, basis=basis)

    # run DFT
    calc = scf.RKS(mol)
    calc.xc = functional
    calc.kernel()

    # calculate density profile
    # PySCF: dm0 = initial guess of the density matrix
    # PySCF: dm1 = density matrix
    # PySCF: dm2 = two particle density matrix
    dm1_ao = calc.make_rdm1()
    n_pts = 1000
    grid = np.zeros((n_pts, 3))
    grid[:, 0] = np.linspace(-2, (hydrogen_count - 1) * spacing * 1.81 + 2, n_pts)
    ao_value = dft.numint.eval_ao(mol, grid, deriv=0)
    rho = dft.numint.eval_rho(mol, ao_value, dm1_ao, xctype="LDA")

    # find peaks / dips
    peaks = signal.find_peaks(rho)[0]
    dips = signal.find_peaks(-rho)[0]
    peak_value = rho[peaks[hydrogen_count // 2 - 1]]
    dip_value = rho[dips[hydrogen_count // 2 - 1]]
    return peak_value - dip_value


import matplotlib.pyplot as plt

dds = [density_difference("PBE", "STO-3G", 4, _) for _ in np.linspace(0.5, 2, 10)]
plt.plot(dds)
# %%
dds = [density_difference("PBE", "cc-pVDZ", _, 0.6) for _ in (4, 6, 8, 10, 12)]
plt.plot(dds)

# %%
import jax
import jax.numpy as jnp

# %%
def square(x: float) -> float:
    return x**2


grad_square = jax.grad(square)
grad_square(3.0)
#%%
grad_grad_square = jax.grad(jax.grad(square))
grad_grad_square(3.0)
# %%
def max_arg(a, b):
    return max(a, b)


grad_max_arg = jax.grad(max_arg, argnums=(0, 1))
grad_max_arg(3, 4)

# %%
def lowest_eigenvalue(matrix: jnp.ndarray) -> jnp.ndarray:
    return jnp.linalg.eigvalsh(matrix.reshape((2, 2)))[0]


grad_lowest_eigenvalue = jax.grad(lowest_eigenvalue)
grad_lowest_eigenvalue(jnp.array([1.0, 2.0, 3.0, 4.0]))

(
    lowest_eigenvalue(jnp.array([1.0, 2.0, 3.0, 4.0]))
    - lowest_eigenvalue(jnp.array([1.1, 2.0, 3.0, 4.0]))
) / 0.1

jax.hessian(lowest_eigenvalue)(jnp.array([1.0, 2.0, 3.0, 4.0]))
# %%
