# %%
from nablachem.alchemy import MultiTaylor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as sco
import inspect
import functools


@functools.cache
def get_mt(molname):
    df = pd.read_csv(
        f"/mnt/c/Users/guido/Downloads/taylor_{molname}.csv",
    )
    natoms = (len(df.columns) - 1) // 4
    columnnames = sum(
        [[l + str(_) for l in "xyzZ"] for _ in range(1, natoms + 1)], []
    ) + ["e"]
    df.columns = columnnames
    mt = MultiTaylor(df, "e")
    center = {_: 0 for _ in set(list(df.columns)) - set(["e"])}
    mt.reset_center(**center)
    mt.build_model(2)
    return mt


# %%
def get_hessian(func, nargs):
    displacement = 0.001
    h = np.zeros((nargs, nargs))

    def eval_func(*args):
        pos = [0 for _ in range(nargs)]
        for k, v in args:
            pos[k] = v * displacement
        return func(*pos)

    center = eval_func()
    for i in range(nargs):
        h[i, i] = (eval_func((i, 1)) - 2 * center + eval_func((i, -1))) / (
            displacement**2
        )
        for j in range(i + 1, nargs):
            h[i, j] = (
                eval_func((i, 1), (j, 1))
                - eval_func((i, 1), (j, -1))
                - eval_func((i, -1), (j, 1))
                + eval_func((i, -1), (j, -1))
            ) / (displacement**2 * 4)
            h[j, i] = h[i, j]
    return h


def get_grad(func, nargs):
    grad = np.zeros(nargs)
    displacement = 0.001

    def eval_func(*args):
        pos = [0 for _ in range(nargs)]
        for k, v in args:
            pos[k] = v * displacement
        return func(*pos)

    for i in range(nargs):
        grad[i] = (eval_func((i, 1)) - eval_func((i, -1))) / (2 * displacement)
    return grad


def build_grad_and_hess(mt: MultiTaylor, output: str):
    args = list(sorted(mt._center.keys()))
    nargs = len(args)
    grad = np.zeros(nargs)
    hess = np.zeros((nargs, nargs))
    for monomial in mt._monomials[output]:
        coef = monomial._prefactor
        order = sum(monomial._powers.values())
        if order == 1:
            (a,) = monomial._powers.keys()
            idx = args.index(a)
            grad[idx] = coef
        elif order == 2:
            try:
                a, b = monomial._powers.keys()
            except:
                (a,) = monomial._powers.keys()
                b = a
            idx1 = args.index(a)
            idx2 = args.index(b)
            hess[idx1, idx2] = coef / 2
            hess[idx2, idx1] = coef / 2
    return grad, hess


def count_nonzero_distinct(values, threshold=1e-4):
    values = np.copy(values)
    mask = abs(values) > threshold
    values = values[mask]
    if len(values) == 0:
        return 0
    values = np.sort(values)

    last_value = values[0]
    count = 1
    for value in values[1:]:
        if value - threshold > last_value:
            count += 1
            last_value = value
    return count


class PivotEstimator:
    def __init__(self, g, h, explain=False):
        self._g = np.array(g)
        self._h = np.array(h)
        self._s = np.ones(g.shape)
        self._n = self._g.shape[0]
        self._explain = explain
        self._span_threshold = 1e-5
        self._unique_threshold = 1e-5

    def estimate(self):
        self._fuzzy()
        if sum(self._mask) > 0:
            self._make_degenerate()
        ndims = self._detect_redundancy()
        return min(ndims, self._n)

    def _eigen(self, s=None):
        if s is None:
            s = self._s
        hmod = np.outer(s, s) * self._h
        return np.linalg.eigh(hmod)

    def _fuzzy(self, ntries=5000):
        evs = np.random.uniform(size=(ntries, self._n), low=0.1, high=10)
        for _ in range(ntries):
            evs[_] = self._eigen(self._s * evs[_]).eigenvalues

        spans = np.amax(evs, axis=0) - np.amin(evs, axis=0)
        if max(spans) > 0:
            spans /= max(spans)
        if self._explain:
            print("Took initial g, H for fuzzying.")
            plt.semilogy(spans, "o-")
            plt.axhline(self._span_threshold)
            plt.show()

        if max(spans) > self._span_threshold:
            spans /= np.amax(spans)
            mask = spans > self._span_threshold
        else:
            mask = spans < 0

        if self._explain:
            print("Kept these entries:")
            print(mask)
        self._mask = mask

    def _make_degenerate(self):
        def _target(smod):
            evs = self._eigen(smod).eigenvalues

            n_modes = sum(abs(evs) / max(abs(evs)) > self._span_threshold)
            mask_penalty = 1e3
            score = mask_penalty * (n_modes - sum(self._mask))

            evs = evs[self._mask]
            evs /= max(abs(evs))
            evs = evs[abs(evs) > self._span_threshold]
            score += count_nonzero_distinct(evs, self._unique_threshold)
            return score

        baseline = _target(self._s)
        res = sco.differential_evolution(
            _target, [[0.1, 10]] * self._n, popsize=50, tol=1e-3
        )
        if res.fun < baseline:
            self._s = res.x

        if self._explain:
            print("scaling", self._s)
            origs = abs(self._eigen(np.ones(self._n)).eigenvalues)
            nows = abs(self._eigen().eigenvalues)
            plt.semilogy(origs / max(origs), "o-", label="original")
            plt.semilogy(nows / max(nows), "o-", label="optimized")
            print("new above threshold", nows / max(nows) > self._span_threshold)
            plt.legend()
            plt.axhline(self._span_threshold)
            plt.show()

    def _detect_redundancy(self):
        gmod = self._g * self._s
        res = self._eigen()

        if sum(self._mask) > 0:
            evs = res.eigenvalues[self._mask]
            evs /= max(abs(evs))
            evs = evs[abs(evs) > self._span_threshold]
            ndims = count_nonzero_distinct(evs, self._unique_threshold)
        else:
            ndims = 0

        # non-zero gradient: one dimension
        if np.linalg.norm(gmod) > 1e-4:
            # ... unless any selected eigenvector is almost the same as the gradient
            kept_vectors = res.eigenvectors[:, self._mask]
            if kept_vectors.shape[1] == 0:
                is_redundant = False
            else:
                is_redundant = (
                    np.linalg.norm((gmod / np.linalg.norm(gmod) @ kept_vectors)) > 0.1
                )
            if not is_redundant:
                ndims += 1
        return ndims


allcases = {
    1: {
        lambda x1: 2: 0,
        lambda x1: x1: 1,
        lambda x1: x1**2: 1,
        lambda x1: x1**2 + x1: 1,
        lambda x1: x1**2 + x1**3: 1,
        lambda x1: x1**3: 0,
    },
    2: {
        lambda x1, x2: x1 + x2: 1,
        lambda x1, x2: x1 + x1**2: 1,
        lambda x1, x2: x1**3 + x1**2: 1,
        lambda x1, x2: x1**3: 0,
        lambda x1, x2: x1 + x2 * x2: 2,
        lambda x1, x2: x1**2 + x2 * x2: 1,
    },
    3: {
        lambda x1, x2, x3: x1 + x2 + x3: 1,
        lambda x1, x2, x3: x1**2 + x2 * x2 + x3 * x3: 1,
        lambda x1, x2, x3: x1**2 + x2 * x2 + 2 * x3 * x3: 2,
        lambda x1, x2, x3: x1**2 + 2 * x2 * x2 + 2 * x3 * x3: 2,
        lambda x1, x2, x3: 2 * x1**2 + 2 * x2 * x2 + 2 * x3 * x3: 1,
    },
}

for nargs, cases in allcases.items():
    for func, expected in cases.items():
        g = get_grad(func, nargs)
        h = get_hessian(func, nargs)
        e = PivotEstimator(g, h, explain=False)
        print(inspect.getsource(func))
        estimated = e.estimate()
        print(estimated, expected)
        assert estimated == expected

# %%

for molname in "N2 CO BF CO2 H2O".split():
    mt = get_mt(molname)
    mt = build_grad_and_hess(mt, "e")
    e = PivotEstimator(*mt, explain=False)
    print(molname, e.estimate())
# %%
