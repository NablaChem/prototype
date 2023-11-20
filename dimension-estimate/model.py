# %%
import hmq


@hmq.task
def get_energy(molecule, basis_set):
    import pyscf.gto
    import pyscf.qmmm
    import pyscf.scf
    from pyscf.data.nist import BOHR
    import numpy as np

    # build molecule
    mol = pyscf.gto.Mole()
    mol.atom = molecule
    mol.basis = basis_set
    mol.verbose = 0
    mol.build()

    # displacement
    displacement = np.random.uniform(-0.05, 0.05, (mol.natm, 4))

    def add_qmmm(calc, mol, deltaZ):
        mf = pyscf.qmmm.mm_charge(calc, mol.atom_coords() * BOHR, deltaZ)

        def energy_nuc(self):
            q = mol.atom_charges().astype(float) + deltaZ
            return mol.energy_nuc(q)

        mf.energy_nuc = energy_nuc.__get__(mf, mf.__class__)
        return mf

    mol.set_geom_(mol.atom_coords() * BOHR + displacement[:, :3])
    calc = add_qmmm(pyscf.scf.RHF(mol), mol, displacement[:, 3])
    calc.kernel(verbose=0)
    total_energy = calc.e_tot
    if not calc.converged:
        raise ValueError("SCF not converged")
    return displacement.flatten().tolist(), total_energy


# %%
for i in range(1200):
    get_energy("N 0 0 0; N 0 0 1", "def2-TZVP")
tag = get_energy.submit()
# %%
tag.pull()
# %%
data = tag.results[:]
# %%
import numpy as np

X = np.array([x for x, _ in data])
y = np.array([y for _, y in data])
# %%
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error


def learning_curve_point(X, y, limit):
    # restrict to a random subset
    idx = np.random.choice(len(X), min(int(limit * 1.3 + 10), len(y)), replace=False)
    X = X.copy()[idx]
    y = y.copy()[idx]

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=limit)

    model = KernelRidge(kernel="rbf")
    param_grid = {
        "alpha": [1e-11, 1e-12, 1e-13],
        "gamma": 2.0 ** np.arange(-5, 5),
    }

    n_cv = min(10, len(X_train))
    grid_search = GridSearchCV(model, param_grid, cv=n_cv)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)
    mse_train = mean_squared_error(y_train, y_train_pred)
    mse_test = mean_squared_error(y_test, y_test_pred)
    return (
        np.sqrt(mse_train),
        np.sqrt(mse_test),
        best_model.get_params()["alpha"],
        best_model.get_params()["gamma"],
    )


# %%
import matplotlib.pyplot as plt

null_model = np.sqrt(mean_squared_error(y, y * 0 + y.mean()))
ntrains = 2 ** np.arange(4, 11)
performance = np.array([learning_curve_point(X, y, ntrain) for ntrain in ntrains])

plt.loglog(ntrains, performance[:, 0], "o-", label="train")
plt.loglog(ntrains, performance[:, 1], "o-", label="test")
plt.axhline(null_model, color="grey", label="null model")
plt.legend()
