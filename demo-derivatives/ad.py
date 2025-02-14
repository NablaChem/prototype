# %%
import pyscfad.gto
import pyscfad.scf
import pyscf.gto
import jax
import numpy as np
import jax.numpy as jnp

mol_co = "O 0 0 -2; C 0 0 0; O 0 0 2.1"


def test(atomspec: str, basis: str):
    mol = pyscfad.gto.Mole()
    mol.atom = atomspec
    mol.basis = basis
    mol.symmetry = False
    mol.build(trace_exp=False, trace_ctr_coeff=False)
    mol_old = pyscf.gto.Mole()
    mol_old.atom = atomspec
    mol_old.basis = basis
    mol_old.symmetry = False
    mol_old.build()

    def change_nuclear_charges(mol, dZ):
        mf = pyscfad.scf.RHF(mol)

        h1 = mf.get_hcore()
        # electronic: extend external potential
        s = 0
        for i, Z in enumerate(dZ):
            mol_old.set_rinv_orig_(mol_old.atom_coords()[i])
            s -= Z * mol_old.intor("int1e_rinv")

        # nuclear: difference to already included NN repulsion
        nn = 0
        for i in range(mol.natm):
            Z_i = mol.atom_charge(i) + dZ[i]

            for j in range(i + 1, mol.natm):
                Z_j = mol.atom_charge(j) + dZ[j]

                if i != j:
                    rij = jnp.linalg.norm(
                        mol.atom_coords().at[i].get() - mol.atom_coords().at[j].get()
                    )
                    missing = Z_i * Z_j - mol.atom_charge(j) * mol.atom_charge(i)
                    nn += missing / rij

        mf.get_hcore = lambda *args, **kwargs: h1 + s
        return mf.kernel() + nn

    nuclear_grad = jax.jacrev(change_nuclear_charges, argnums=1)(
        mol, np.zeros(mol.natm)
    )
    nuclear_hess = jax.jacrev(jax.jacrev(change_nuclear_charges, argnums=1), argnums=1)(
        mol, np.zeros(mol.natm)
    )
    spatial_grad = jax.jacrev(change_nuclear_charges, argnums=0)(
        mol, np.zeros(mol.natm)
    ).coords
    spatial_hess = jax.jacfwd(jax.jacrev(change_nuclear_charges, argnums=0), argnums=0)(
        mol, np.zeros(mol.natm)
    ).coords.coords

    mixed_hess = jax.jacfwd(jax.jacrev(change_nuclear_charges, argnums=1), argnums=0)(
        mol, np.zeros(mol.natm)
    ).coords

    # gradient (dx1, dy1, dz1, dZ1, dx2, ...)
    gradient = np.zeros((mol.natm, 4))
    gradient[:, :3] = spatial_grad
    gradient[:, 3] = nuclear_grad

    hessian = np.zeros((mol.natm, 4, mol.natm, 4))
    hessian[:, :3, :, :3] = spatial_hess
    hessian[:, 3, :, 3] = nuclear_hess
    hessian[:, 3, :, :3] = mixed_hess
    hessian[:, :3, :, 3] = mixed_hess.transpose(1, 2, 0)
    hessian = hessian.reshape(4 * mol.natm, -1)

    return gradient.reshape(-1), hessian


test(mol_co, "sto-3g")

# %%
