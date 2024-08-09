# %%
from nablachem.alchemy import MultiTaylor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %%
import functools


@functools.cache
def get_mt(molname):
    df = pd.read_csv(
        f"/mnt/c/Users/guido/Downloads/taylor_{molname}.csv",
        names="x1 y1 z1 Z1 x2 y2 z2 Z2 e".split(),
    )
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


# %%
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
    values = np.sort(values)
    if len(values) == 0:
        return 0, mask

    last_value = values[0]
    count = 1
    for value in values[1:]:
        if value - threshold > last_value:
            count += 1
            last_value = value
    return count, mask


def get_local_id(g, h, t):
    # Hessian: one dim per non-zero distinct eigenvalue
    res = np.linalg.eigh(h)
    # span = res.eigenvalues.max() - res.eigenvalues.min()
    # if span == 0:
    #    span = 1
    # ndims, mask = count_nonzero_distinct(res.eigenvalues, t * span)
    # find the nullspace entries by fuzzing: no scaling can make them non-zero
    evs = []
    for i in range(5000):
        s = np.random.uniform(size=g.shape, low=0.1, high=10)
        gmod = g * s
        hmod = np.outer(s, s) * h
        evs.append(np.linalg.eigvalsh(hmod))
    evs = np.array(evs)
    spans = np.amax(evs, axis=0) - np.amin(evs, axis=0)
    if max(spans) > 1e-4:
        spans /= np.amax(spans)
        mask = spans > 1e-4
    else:
        mask = spans < 0
    ndims = sum(mask)

    # non-zero gradient: one dimension
    if np.linalg.norm(g) > 1e-4:
        # ... unless any selected eigenvector is almost the same as the gradient
        kept_vectors = res.eigenvectors[:, mask]
        if kept_vectors.shape[1] == 0:
            is_redundant = False
        else:
            is_redundant = np.linalg.norm((g / np.linalg.norm(g) @ kept_vectors)) > 0.1
        if not is_redundant:
            ndims += 1

    return min(ndims, len(g))


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
    print(nargs)
    for func, expected in cases.items():
        g = get_grad(func, nargs)
        h = get_hessian(func, nargs)
        t = 1e-3
        print(".", get_local_id(g, h, t))
        assert get_local_id(g, h, t) == expected

import scipy.optimize as sco

# %%


def get_local_id_scaled(g, h, t):
    def target(s):
        ssq = s**2
        gmod = g * ssq
        hmod = np.outer(ssq, ssq) * h
        return get_local_id(gmod, hmod, t)

    res = sco.differential_evolution(target, [[1, 500]] * h.shape[0])
    return res.fun


ts = 10.0 ** np.arange(-15, 0)
for molidx, molname in enumerate("CO".split()):
    mt = get_mt(molname)
    mt = build_grad_and_hess(mt, "e")
    ids = [get_local_id(*mt, t) for t in ts]
    plt.semilogx(ts, ids, label=molname, color=f"C{molidx}")
    ids = [get_local_id_scaled(*mt, t) for t in ts]
    plt.semilogx(ts, ids, label=molname, color=f"C{molidx}", ls="--")
plt.legend()
plt.xlabel("Threshold")
plt.ylabel("Local intrinsic dimension")


# %%
mt = get_mt("CO")
g, h = build_grad_and_hess(mt, "e")


# %%
def target(s):
    ssq = s**2
    gmod = g * ssq
    hmod = np.outer(ssq, ssq) * h
    return get_local_id(gmod, hmod, t)


t = 1e-4
res = sco.differential_evolution(target, [[1, 5]] * h.shape[0])
res
# %%
print(res.x)
ts = 10.0 ** np.arange(-15, -2)
ssq = res.x**2
gmod = g * ssq
hmod = np.outer(ssq, ssq) * h
ids = [get_local_id(gmod, hmod, _) for _ in ts]
plt.semilogx(ts, ids)


# %%
# eigenvalue 0: nullspace
# degenerate eigenvalues: symmetries
def func1(x1, x2):
    return x1 * x2 + x1**2 + 2 * x2**2


np.linalg.eigvalsh(get_hessian(func1, 2))
# %%
evs = []
for scaling in np.linspace(0, 5, 500):

    def func1(x1, x2):
        x1 = x1 * scaling
        return x1**2 + 2 * x2**2
        return x1 * x2 + x1**2 + 2 * x2**2

    evs.append(np.linalg.eigvalsh(get_hessian(func1, 2)))
evs = np.array(evs)
plt.plot(evs)
# %%
s = np.ones(g.shape)
s = np.random.uniform(size=g.shape)
pos = 2
evs = []
for scaling in np.linspace(0.1, 1, 500):
    s[pos] = scaling
    gmod = g * s
    hmod = np.outer(s, s) * h
    evs.append(np.linalg.eigvalsh(hmod))
evs = np.array(evs)
plt.plot(evs)
get_local_id(gmod, hmod, 1e-7)
# %%
