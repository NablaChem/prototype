# %%
import numpy as np
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from __future__ import annotations


def get_training_data(ndim, npts):
    def testfun(x):
        return np.sin(x.sum(axis=-1))

    X = np.random.uniform(0, 1, size=(npts, ndim))
    y = testfun(X)
    return X, y


# %%
class Kernel:
    @staticmethod
    def _check_shape(arr: np.ndarray, shape: tuple[int]):
        if arr.shape != shape:
            raise ValueError(
                f"Shape mismatch. Please check kernel implementation. Expected {shape}, got {arr.shape}."
            )

        return arr

    def _assert_representation(self, X: np.ndarray):
        if isinstance(X[0], ase.Atoms):
            X = np.array([self._representation.transform(_) for _ in X])
        return X

    def _check_args(self, X: np.ndarray, Xother: np.ndarray, args, kwargs):
        if len(args) != 0:
            raise ValueError("Function requires explicit keyword-arguments.")

        for parameter in kwargs.keys():
            if parameter not in self._parameters:
                raise ValueError(f"Unknown parameter {parameter} passed.")

        X = self._assert_representation(X)

        if Xother is None:
            Xother = X
        else:
            Xother = self._assert_representation(Xother)

        return X, Xother

    def value(self, X: np.ndarray, Xother: np.ndarray = None, *args, **kwargs):
        X, Xother = self._check_args(X, Xother, args, kwargs)

        vals = np.array(self.calculate_value(X, Xother, kwargs))

        return Kernel._check_shape(vals, (X.shape[0], Xother.shape[0]))

    def grad(self, X: np.ndarray, Xother: np.ndarray = None, *args, **kwargs):
        X, Xother = self._check_args(X, Xother, args, kwargs)

        jac = jax.jacfwd(self.calculate_value, argnums=2)(X, Xother, kwargs)
        jacs = []
        for parameter in self._parameters:
            jacs.append(np.array(jac[parameter]))

        return Kernel._check_shape(
            np.array(jacs), (len(self._parameters), X.shape[0], Xother.shape[0])
        )

    def __init__(self, representation: Representation):
        self._representation = representation

    def prepare_for_dataset(self, dataset: list[ase.Atoms]):
        self._representation.prepare_for_dataset(dataset)


class LinearCombination(Kernel):
    def __init__(self, *args: tuple[Kernel]):
        self._parameters = []
        self._kernels = list(args)

        for kernel_index, kernel in enumerate(self._kernels):
            self._parameters.append(f"__{kernel_index}_prefactor")
            self._parameters += [f"_{kernel_index}_{_}" for _ in kernel._parameters]

    def _get_subkwargs(self, kwargs, kernel_index):
        subkwargs = {}
        for arg in kwargs:
            _, idx, argname = arg.split("_", 2)
            if idx == str(kernel_index):
                subkwargs[argname] = kwargs[arg]
        return subkwargs

    def value(self, X: np.ndarray, Xother: np.ndarray = None, *args, **kwargs):
        X, Xother = self._check_args(X, Xother, args, kwargs)

        vals = 0
        for kernel_index, kernel in enumerate(self._kernels):
            subkwargs = self._get_subkwargs(kwargs, kernel_index)
            vals += kwargs[f"__{kernel_index}_prefactor"] * kernel.value(
                X, Xother, *args, **subkwargs
            )

        return Kernel._check_shape(vals, (X.shape[0], Xother.shape[0]))

    def grad(self, X: np.ndarray, Xother: np.ndarray = None, *args, **kwargs):
        X, Xother = self._check_args(X, Xother, args, kwargs)

        grad = {}
        for kernel_index, kernel in enumerate(self._kernels):
            subkwargs = self._get_subkwargs(kwargs, kernel_index)
            grad[f"__{kernel_index}_prefactor"] = kernel.value(
                X, Xother, *args, **subkwargs
            )
            kgrad = kernel.grad(X, Xother, *args, **subkwargs)
            for parameter, gradelement in zip(kernel._parameters, kgrad):
                grad[f"{kernel_index}_{parameter}"] = gradelement
        jacs = [grad[_] for _ in self._parameters]

        return Kernel._check_shape(
            np.array(jacs), (len(self._parameters), X.shape[0], Xother.shape[0])
        )


class RBF(Kernel):
    _parameters = ["sigma"]

    def calculate_value(self, X, Xother, kwargs):
        distances = jnp.linalg.norm((Xother - X[:, None]), axis=-1)
        return jnp.exp(-(distances**2) / (2 * kwargs["sigma"] ** 2))


def test_gaussian_finite_diff():
    def one_run(kernel, Xtrain, ytrain, Xtest, ytest, sigma, lval):
        npts, _ = Xtrain.shape
        K = kernel.value(Xtrain, Xtrain, sigma=sigma)
        alpha = np.linalg.inv(K.T + lval * np.identity(npts)) @ ytrain

        K = kernel.value(Xtrain, Xtest, sigma=sigma)
        pred = np.sum(K.T * alpha, axis=1)
        return np.average((pred - ytest) ** 2)

    kernel = RBF()
    ndim, npts, sigma, lval = 2, 10, 1e-1, 1e-10

    ntest = 100
    train = get_training_data(ndim, npts)
    test = get_training_data(ndim, ntest)

    # finite diff
    delta = 0.001
    down = one_run(kernel, *train, *test, sigma + delta, lval)
    up = one_run(kernel, *train, *test, sigma - delta, lval)
    deriv = (up - down) / (2 * delta)

    # analytical
    K = kernel.value(train[0], train[0], sigma=sigma)
    beta = np.linalg.inv(K + lval * np.identity(npts))
    alpha = beta @ train[1]

    gradK = kernel.grad(train[0], train[0], sigma=sigma)[0]
    gradbeta = -np.linalg.inv(K) @ gradK @ np.linalg.inv(K)
    K = kernel.value(train[0], test[0], sigma=sigma)
    gradK = kernel.grad(train[0], test[0], sigma=sigma)[0]
    pred = np.sum(K.T * alpha, axis=1)

    gradhat = gradbeta @ train[1] @ K + beta @ train[1] @ gradK
    deriv_analytical = -(2 * (pred - test[1]) * gradhat).sum() / ntest

    assert abs(deriv - deriv_analytical) < 1e-3


# %%
X, y = get_training_data(2, 10)
kern = LinearCombination(RBF(), RBF())
print(kern._parameters)
kern.value(X, X, __0_prefactor=1, _0_sigma=1, __1_prefactor=1, _1_sigma=1)

# %%
import ase


class Representation:
    def prepare_for_dataset(self, dataset: list[ase.Atoms]):
        raise NotImplementedError("Representations need to implement this function.")

    def transform(self, mol: ase.Atoms):
        raise NotImplementedError("Representations need to implement this function.")


import dscribe.descriptors


class CoulombMatrix(Representation):
    def prepare_for_dataset(self, dataset: list[ase.Atoms]):
        maxatoms = max([len(_) for _ in dataset])
        self._cm = dscribe.descriptors.CoulombMatrix(n_atoms_max=maxatoms)

    def transform(self, mol: ase.Atoms):
        return self._cm.create(mol)


class OptKRR:
    def __init__(
        self, kernel: Kernel, molecules: np.ndarray[ase.Atoms], labels: np.ndarray
    ):
        self._kernel = kernel
        self._kernel.prepare_for_dataset(molecules)
        self._split_dataset(molecules, labels)

    def _split_dataset(self, molecules: list[ase.Atoms], labels: np.ndarray):
        ntotal = len(molecules)
        print(f"Received dataset of {ntotal} molecules.")
        molecules = np.array(molecules, dtype=object)
        ntrain = int(0.8 * ntotal)
        nholdout = ntotal - ntrain
        if nholdout < 100:
            print("WARNING: Holdout set has only few data points.")

        train, holdout = OptKRR._random_indices(ntrain, ntotal)
        self._training_data = (molecules[train], labels[train])
        self._holdout_data = (molecules[holdout], labels[holdout])

    @staticmethod
    def _random_indices(take: int, total: int):
        idx = np.arange(total)
        np.random.shuffle(idx)
        return idx[:take], idx[take:]

    def _guess_parameters(self):
        return {_: 1 for _ in self._kernel._parameters}

    def _get_rmse(self, train: np.ndarray, test: np.ndarray, parameters: dict):
        Xtrain = self._training_data[0][train]
        ytrain = self._training_data[1][train]
        Xtest = self._training_data[0][test]
        ytest = self._training_data[1][test]

        npts = len(train)
        lval = 1e-7  # TODO
        K = kernel.value(Xtrain, Xtrain, **parameters)
        alpha = np.linalg.inv(K.T + lval * np.identity(npts)) @ ytrain

        K = kernel.value(Xtrain, Xtest, **parameters)
        pred = np.sum(K.T * alpha, axis=1)
        return np.average((pred - ytest) ** 2)

    @property
    def ntrain(self):
        return len(self._training_data[0])

    def run(self, ntrain: int, nsubsplits: int = 5):
        if ntrain > self.ntrain:
            raise ValueError(
                f"Fewer than the requested {ntrain} data points available."
            )

        subsplits = [
            OptKRR._random_indices(ntrain, self.ntrain) for _ in range(nsubsplits)
        ]

        start = self._guess_parameters()
        rmses = [self._get_rmse(*_, start) for _ in subsplits]
        print(np.array(rmses).mean())


kernel = RBF(CoulombMatrix())
opt = OptKRR(kernel, a[:500], l[:500])
opt.run(199)

# %%
import io
import glob
import ase.io


def get_local_QMspin_database():
    basedir = "/home/ferchault/data/QMspin/QMspin_Part1/geometries_singlet"
    atoms = []
    labels = []
    for filename in glob.glob(basedir + "/*.xyz"):
        with open(filename) as fh:
            content = fh.read()

        fh = io.StringIO(content.replace(",", " "))
        try:
            mol = ase.io.read(fh, format="xyz")
        except:
            continue
        atoms.append(mol)
        labels.append(float(content.split("\n")[1].split(";")[3].split()[3]))
    return np.array(atoms, dtype=object), np.array(labels)


a, l = get_local_QMspin_database()

# %%
import requests
import tarfile


def database_qm9(random_limit=1000):
    """Reads the QM9 database from network, http://www.nature.com/articles/sdata201422."""
    # exclusion list
    res = requests.get("https://ndownloader.figshare.com/files/3195404")
    exclusion_ids = [
        _.strip().split()[0] for _ in res.content.decode("ascii").split("\n")[9:-2]
    ]

    # geometries and energies
    res = requests.get("https://ndownloader.figshare.com/files/3195389")
    webfh = io.BytesIO(res.content)
    t = tarfile.open(fileobj=webfh)
    energies = []
    contents = []
    for fo in t:
        lines = t.extractfile(fo).read().decode("ascii").split("\n")
        natoms = int(lines[0])
        lines = lines[: 2 + natoms]
        lines = [_.replace("*^", "e") for _ in lines]
        molid = lines[1].strip().split()[0]
        if molid in exclusion_ids:
            continue
        energies.append(float(lines[1].strip().split()[12]))
        contents.append(lines)

    # random subset for development purposes
    idx = np.arange(len(energies))
    np.random.shuffle(idx)
    subset = idx[:random_limit]

    energies = [energies[_] for _ in subset]
    compounds = [ase.io.read(io.StringIO(contents[_]), format="xyz") for _ in subset]
    return np.array(compounds, dtype=object), np.array(energies)


# %%
