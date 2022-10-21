#%%
import numpy as np
import itertools as it
import matplotlib.pyplot as plt

timings = {"HF": 0.016328125, "MP2": 0.039101563, "CCSD": 0.235273438}
nullmodels = {"HF": 35.7, "MP2": 6.16, "CCSD": 2.76}

# %%
def cost_and_error(N_HF: int, N_MP2: int, N_CCSD: int) -> tuple[float, float]:
    cost = N_HF * timings["HF"] + N_MP2 * timings["MP2"] + N_CCSD * timings["CCSD"]
    eA = nullmodels["HF"] * np.exp(-0.5 * (np.log(N_HF) - 1))
    eB = nullmodels["MP2"] * np.exp(-0.5 * (np.log(N_MP2) - 1))
    eC = nullmodels["CCSD"] * np.exp(-0.5 * (np.log(N_CCSD) - 1))
    return cost, np.sqrt(eA**2 + eB**2 + eC**2)


def optimal_for_error(threshold: float) -> tuple[int, int, int]:
    levels = [int(_) for _ in 2 ** np.linspace(0, 15)]

    mincost = 1e100
    opt = None
    for NHF, NMP2, NCCSD in it.product(levels, levels, levels):
        cost, error = cost_and_error(NHF, NMP2, NCCSD)
        if cost < mincost and error < threshold:
            mincost = cost
            opt = (NHF, NMP2, NCCSD)
    return opt


# %%
errors = (32, 16, 8, 4, 2, 1)
values = [optimal_for_error(_) for _ in errors]
values = np.array(values)
plt.plot(errors, values[:, 0] / np.sum(values, axis=1), label="HF")
plt.plot(errors, values[:, 1] / np.sum(values, axis=1), label="MP2")
plt.plot(errors, values[:, 2] / np.sum(values, axis=1), label="CCSD")
plt.xlabel("desired MAE")
plt.ylabel("Fraction")
plt.legend()
print("MAE, N_HF, N_MP2, N_CCSD")
print(np.hstack((np.array(errors)[:, np.newaxis], values)))
