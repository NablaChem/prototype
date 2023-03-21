#%%
BASEDIR = "/home/ferchault/wrk/prototype/alchemy-model-pecd"
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import findiff
import math
import functools
import scipy.interpolate as sci


def get_data(kind: str, reduced=False):
    df = pd.read_csv(f"{BASEDIR}/{kind}.txt", index_col=0, delim_whitespace=True)
    df.columns = [float(_.split("=")[1]) for _ in df.columns]
    if reduced:
        df = df.iloc[:, 1:-1]
    return df


# get_data("charge", reduced=True)
# df_pos = get_data("position")
# %%
# compare to KRR/GPR
# uncertainty from distance in dq/dr
# discuss error as function of displacement

# %%
def central_finite_difference(displacement: float, values: np.ndarray):
    assert len(values) % 2 == 1
    order = 0
    pcoeffs = [values[len(values) // 2]]
    while True:
        order += 1
        for acc in (4, 2):
            coeffs = findiff.coefficients(deriv=order, acc=acc)["center"][
                "coefficients"
            ]
            if len(coeffs) == len(values):
                break
        else:
            break
        pcoeffs.append(
            np.sum(coeffs * values) / displacement**order / math.factorial(order)
        )
    return np.poly1d(pcoeffs[::-1])


def alchemical_pecd(energy: float, dr: float, dq: float, reduced=False):
    line = get_data("charge", reduced=reduced).loc[energy].to_list()
    taylor_charge = central_finite_difference(0.1, np.array(line))
    line = get_data("position", reduced=reduced).loc[energy].to_list()
    taylor_pos = central_finite_difference(0.1, np.array(line))
    return (
        line[len(line) // 2]
        + taylor_charge(dq)
        - taylor_charge(0)
        + taylor_pos(dr)
        - taylor_pos(0)
    ) * 100


#%%
def estimated_uncertainty(energy, dr, dq):
    return abs(
        alchemical_pecd(energy, dr, dq, reduced=False)
        - alchemical_pecd(energy, dr, dq, reduced=True)
    )


for energy in range(4, 11):
    target = functools.partial(alchemical_pecd, energy)
    drs = np.linspace(-0.5, 0.5, 20)
    dqs = np.linspace(-0.5, 0.5, 20)
    X, Y = np.meshgrid(drs, dqs)
    Z = np.array([(target(dr, dq)) for dr, dq in zip(X.ravel(), Y.ravel())]).reshape(
        X.shape
    )
    print(np.amax(Z), np.amin(Z))
    levels = range(-10, 11)
    plt.contourf(X, Y, Z, levels=levels, cmap="RdBu")
    plt.colorbar()

    target2 = functools.partial(estimated_uncertainty, energy)
    Z2 = np.array([(target2(dr, dq)) for dr, dq in zip(X.ravel(), Y.ravel())]).reshape(
        X.shape
    )
    cs = plt.contour(
        X,
        Y,
        Z2,
        levels=[
            1,
        ],
        colors="white",
        linewidths=0.8,
    )
    maxval = 0
    pos = None
    for item in cs.collections:
        for i in item.get_paths():
            v = i.vertices
            xs = v[:, 0]
            ys = v[:, 1]
            for x, y in zip(xs, ys):
                value = abs(target(x, y))
                if value > maxval:
                    maxval = value
                    pos = (x, y)
    print(pos, maxval)
    plt.scatter(
        (pos[0],), (pos[1],), color="yellow", s=50, edgecolors="grey", zorder=100
    )

    plt.xlabel("dr [a.u.]")
    plt.ylabel("dq [a.u.]")
    plt.title("$\\beta_1$@E={}".format(energy))
    plt.scatter(
        (0, 0, 0, 0, 0),
        (-0.2, -0.1, 0, 0.1, 0.2),
        color="white",
        edgecolors="grey",
        zorder=100,
    )
    plt.scatter(
        (-0.2, -0.1, 0, 0.1, 0.2),
        (0, 0, 0, 0, 0),
        color="white",
        edgecolors="grey",
        zorder=100,
    )
    plt.show()


# %%
# what if lower orders only?
def compare_to_lower_order(dr: bool):
    if dr:
        plt.title("+- 0.2 from (+- 0.1, 0) for geometry")
        dr, dq = 0.2, 0.0
    else:
        plt.title("+- 0.2 from (+- 0.1, 0) for charges")
        dr, dq = 0.0, 0.2
    Es = range(4, 11)
    betas = [alchemical_pecd(_, dr, dq, reduced=False) for _ in Es]
    plt.plot(Es, betas, color="C0", label="reduced")
    betas = [alchemical_pecd(_, dr, dq, reduced=True) for _ in Es]
    plt.plot(Es, betas, color="C1", label="full")

    betas = [alchemical_pecd(_, -dr, -dq, reduced=False) for _ in Es]
    plt.plot(Es, betas, color="C0")
    betas = [alchemical_pecd(_, -dr, -dq, reduced=True) for _ in Es]
    plt.plot(Es, betas, color="C1")
    plt.legend()
    plt.show()


compare_to_lower_order(True)
compare_to_lower_order(False)
# %%
def through_energy(dr, dq):
    Es = range(4, 11)
    betas = [alchemical_pecd(_, dr, dq, reduced=False) for _ in Es]
    plt.plot(Es, betas, color="C0", label="reduced")
    betas = [alchemical_pecd(_, dr, dq, reduced=True) for _ in Es]
    plt.plot(Es, betas, color="C1", label="full")
    plt.legend()
    plt.show()


through_energy(0.5, 0.3)
# %%
