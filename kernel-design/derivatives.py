# %%
from __future__ import annotations
import numpy as np
import jax
import functools
import time
import ase


jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp


# %%
class Kernel:
    def _check_shape(
        self,
        value: np.ndarray = None,
        gradient: np.ndarray = None,
        training: bool = True,
        npts: int = 1,
    ):
        if training:
            len_other = npts
        else:
            len_other = self._dataset.ntrain - npts

        gradient_shape = (len(self._parameters), npts, len_other)
        value_shape = (npts, len_other)

        if value is not None:
            got = value.shape
            expected = value_shape
        if gradient is not None:
            got = gradient.shape
            expected = gradient_shape
        if got != expected:
            raise ValueError(
                f"Shape mismatch. Please check kernel implementation. Expected {expected}, got {got}."
            )

        if value is not None:
            return value
        return gradient

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

    def _get_representation(
        self, split_idx: int, npts: int, training: bool, rep_params: dict
    ):
        X = self._dataset.split_points(split_idx, npts, training)
        if isinstance(X[0], ase.Atoms):
            X = np.array([self._representation.transform(_, rep_params) for _ in X])
        return X

    def _split_params(parameters: dict):
        rep_params, kernel_params = {}, {}
        for k, v in parameters.items():
            if k.startswith("_"):
                rep_params[k[1:]] = v
            else:
                kernel_params[k] = v
        return rep_params, kernel_params

    def value(self, split_idx: int, npts: int, training: bool, **parameters):
        rep_params, kernel_params = Kernel._split_params(parameters)

        X = self._get_representation(split_idx, npts, True, rep_params)
        Xother = self._get_representation(split_idx, npts, training, rep_params)

        return np.array(self.kernel_matrix(X, Xother, kernel_params))

    def grad(self, split_idx: int, npts: int, training: bool, **parameters):
        rep_params, kernel_params = Kernel._split_params(parameters)

        X = self._get_representation(split_idx, npts, True, rep_params)
        Xother = self._get_representation(split_idx, npts, training, rep_params)

        jac = jax.jacfwd(self.kernel_matrix, argnums=2)(X, Xother, kernel_params)
        jacs = []
        for parameter in sorted(self._parameters):
            jacs.append(np.array(jac[parameter]))

        return np.array(jacs)

    def __init__(self, representation: Representation):
        self._representation = representation

        for parameter in self._parameters:
            if parameter.startswith("_"):
                raise ValueError("Parameters must not have leading underscores.")

    def prepare_for_dataset(self, dataset: list[ase.Atoms]):
        self._cache = {}
        self._dataset = dataset
        try:
            self._representation.prepare_for_dataset(dataset)
        except:
            pass


class LinearCombination(Kernel):
    def __init__(self, *args: tuple[Kernel]):
        self._parameters = []
        self._kernels = list(args)

        for kernel_index, kernel in enumerate(self._kernels):
            self._parameters.append(f"__{kernel_index}_prefactor")
            self._parameters += [f"_{kernel_index}_{_}" for _ in kernel._parameters]

    def prepare_for_dataset(self, dataset: Dataset):
        super().prepare_for_dataset(dataset)
        for kernel in self._kernels:
            kernel.prepare_for_dataset(dataset)

    def _get_subkwargs(self, kwargs, kernel_index):
        subkwargs = {}
        for arg in kwargs:
            _, idx, argname = arg.split("_", 2)
            if idx == str(kernel_index):
                subkwargs[argname] = kwargs[arg]
        return subkwargs

    def value(self, split_idx: int, npts: int, training: bool, **kwargs):
        vals = 0
        for kernel_index, kernel in enumerate(self._kernels):
            subkwargs = self._get_subkwargs(kwargs, kernel_index)
            vals = kernel.value(split_idx, npts, training, **subkwargs)
            vals += kwargs[f"__{kernel_index}_prefactor"] * vals

        return self._check_shape(value=vals, training=training, npts=npts)

    def grad(self, split_idx: int, npts: int, training: bool, **kwargs):
        grad = {}
        for kernel_index, kernel in enumerate(self._kernels):
            subkwargs = self._get_subkwargs(kwargs, kernel_index)
            grad[f"__{kernel_index}_prefactor"] = kernel.value(
                split_idx, npts, training, **subkwargs
            )
            kgrad = kernel.grad(split_idx, npts, training, **subkwargs)
            for parameter, gradelement in zip(kernel._parameters, kgrad):
                grad[f"_{kernel_index}_{parameter}"] = gradelement
        jacs = [grad[_] for _ in sorted(self._parameters)]

        return self._check_shape(gradient=np.array(jacs), training=training, npts=npts)


class RBF(Kernel):
    _parameters = ["sigma"]

    def kernel_matrix(self, X: np.ndarray, Xother: np.ndarray, kernel_params: dict):
        distances = jnp.linalg.norm((Xother - X[:, None]), axis=-1)
        return jnp.exp(-(distances**2) / (2 * kernel_params["sigma"] ** 2))


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


class Representation:
    def prepare_for_dataset(self, dataset: list[ase.Atoms]):
        raise NotImplementedError("Representations need to implement this function.")

    def transform(self, mol: ase.Atoms, parameters: dict):
        raise NotImplementedError("Representations need to implement this function.")


import dscribe.descriptors


class CoulombMatrix(Representation):
    def prepare_for_dataset(self, dataset: list[ase.Atoms]):
        maxatoms = max([len(_) for _ in dataset])
        self._cm = dscribe.descriptors.CoulombMatrix(n_atoms_max=maxatoms)

    def transform(self, mol: ase.Atoms, parameters: dict):
        return self._cm.create(mol)


class Identity(Representation):
    def prepare_for_dataset(self, dataset):
        pass

    def transform(self, point: np.ndarray, parameters: dict):
        return point


class Dataset:
    def __init__(
        self,
        data_points: np.ndarray[ase.Atoms] | np.ndarray[float],
        labels: np.ndarray,
        nsplits: int,
    ):
        if isinstance(data_points[0], ase.Atoms):
            self._data_points = np.array(data_points, dtype=object)
        else:
            self._data_points = data_points

        self._labels = labels
        self._split(nsplits)

    @property
    def npoints(self) -> int:
        return len(self._data_points)

    @property
    def ntrain(self) -> int:
        return int(0.8 * self.npoints)

    @property
    def nholdout(self) -> int:
        return self.npoints - self.ntrain

    @property
    def nsplits(self) -> int:
        return len(self._splits)

    def _split(self, nsplits: int):
        # training - holdout
        idx = np.arange(self.npoints)
        np.random.shuffle(idx)
        self._training_indices = idx[: self.ntrain]
        self._holdout_indices = idx[self.ntrain :]

        # splits for cross validation
        idx = np.arange(self.ntrain)
        self._splits = []
        for i in range(nsplits):
            np.random.shuffle(idx)
            self._splits.append(idx.copy())

    def _split_idx(self, split_idx, npts, training):
        idx = self._splits[split_idx]
        if training:
            idx = idx[:npts]
        else:
            idx = idx[npts:]
        return idx

    def split_labels(self, split_idx, npts, training):
        return self._labels[self._split_idx(split_idx, npts, training)]

    def split_points(self, split_idx, npts, training):
        return self._data_points[self._split_idx(split_idx, npts, training)]


def get_training_data(ndim, npts):
    def testfun(x):
        return np.sin(x.sum(axis=-1))

    X = np.random.uniform(0, 1, size=(npts, ndim))
    y = testfun(X)
    return Dataset(X, y, 5)


class OptKRR:
    def __init__(self, kernel: Kernel, dataset: Dataset):
        self._kernel = kernel
        self._dataset = dataset
        self._kernel.prepare_for_dataset(dataset)
        print(f"Received dataset of {dataset.npoints} points.")

        if self._dataset.nholdout < 100:
            print("WARNING: Holdout set has only few data points.")

    def _guess_parameters(self) -> dict[str, float]:
        return {_: 1.0 for _ in self._kernel._parameters}

    def _score(self, npts: int, parameters: dict, include_grad=False):
        avgs = []
        derivs = []
        lval = 1e-7  # TODO
        for split_idx in range(self._dataset.nsplits):
            ytrain = self._dataset.split_labels(split_idx, npts, True)
            ytest = self._dataset.split_labels(split_idx, npts, False)

            Ktraintrain = self._kernel.value(split_idx, npts, True, **parameters)
            Ktraintrain_inv = np.linalg.inv(Ktraintrain)
            beta = np.linalg.inv(Ktraintrain + lval * np.identity(npts))
            alpha = beta @ ytrain

            if include_grad:
                gradKs_traintrain = self._kernel.grad(
                    split_idx, npts, True, **parameters
                )
                gradKs_traintest = self._kernel.grad(
                    split_idx, npts, False, **parameters
                )
            deriv_analytical = []
            Ktraintest = self._kernel.value(split_idx, npts, False, **parameters)
            pred = np.sum(Ktraintest.T * alpha, axis=1)
            if include_grad:
                for parameter_idx in range(gradKs_traintrain.shape[0]):
                    gradbeta = (
                        -Ktraintrain_inv
                        @ gradKs_traintrain[parameter_idx]
                        @ Ktraintrain_inv
                    )
                    gradhat = (
                        gradbeta @ ytrain @ Ktraintest
                        + alpha @ gradKs_traintest[parameter_idx]
                    )

                    deriv_analytical.append(
                        -(2 * (pred - ytest) * gradhat).sum() / len(ytest)
                    )

            avgs.append(np.average((pred - ytest) ** 2))
            derivs.append(np.array(deriv_analytical))

        if include_grad:
            derivs = np.median(derivs, axis=0)
        else:
            derivs = None
        return np.average(avgs), derivs

    @property
    def ntrain(self):
        return len(self._training_data[0])

    def run(self, ntrain: int):
        if ntrain > self._dataset.ntrain:
            raise ValueError(
                f"Fewer than the requested {ntrain} data points available."
            )

        start = self._guess_parameters()

        print(f"Optimizing {len(start)} parameters for the given kernel.")
        print("step | RMSE    | improvement | time | parameters")
        firstscore = None
        starttime = time.time()
        for i in range(20):
            score, grad = self._score(ntrain, start, include_grad=True)
            if firstscore is None:
                firstscore = score

            print(
                f"{i:4} | {score:5.1e} | {firstscore / score:11.1f} | {time.time()-starttime:4.1f} | {start}"
            )

            direction = grad / np.linalg.norm(grad)
            # direction should not depend on param scale

            # line scan
            best_scale = None
            best_score = score
            for scale in 2.0 ** np.arange(-7, 3):
                for pidx, parameter in enumerate(sorted(self._kernel._parameters)):
                    start[parameter] += direction[pidx] * scale
                score, _ = self._score(ntrain, start, include_grad=False)
                if best_score is None or score < best_score:
                    best_scale = scale
                    best_score = score
                for pidx, parameter in enumerate(sorted(self._kernel._parameters)):
                    start[parameter] -= direction[pidx] * scale
            if best_scale is None:
                break

            for pidx, parameter in enumerate(sorted(self._kernel._parameters)):
                start[parameter] += direction[pidx] * best_scale


# dataset = get_training_data()
# kernel = RBF(Identity())
# opt = OptKRR(kernel, a[:500], l[:500])
# opt.run(199)
if __name__ == "__main__":
    ds = get_training_data(3, 500)
    kernel = RBF(Identity())
    opt = OptKRR(kernel, ds)
    opt.run(100)


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


# a, l = get_local_QMspin_database()

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
