import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erf

xs = np.linspace(0, 6, 100)


def get_qbar(Q: float, r: float, gbar: float):
    a = Q - r * gbar
    b = Q + r * gbar

    A = np.exp(-(a**2) / 2) - np.exp(-(b**2) / 2)
    B = erf(b / np.sqrt(2)) - erf(a / np.sqrt(2))

    return np.sqrt(2 / np.pi) * A / B


f, axs = plt.subplots(1, 2, figsize=(12, 4))

for Q in (-2, -1, 0, 1, 2):
    f = get_qbar(Q, xs, 1)
    axs[0].plot(xs, f, label=f"Q={Q}")


axs[0].set_xlabel("r")
axs[0].set_ylabel(r"$\bar{Q}(r)$")
axs[0].legend()


def get_slope_estimate(Q, gbar):
    initial_value = get_qbar(Q, 1e-3, gbar)
    current = 2e-3
    while True:
        current_value = get_qbar(Q, current, gbar)
        if current_value / initial_value < 0.000001:
            return (current_value - initial_value) / (current - 1e-3)
        current *= 1.01


for gbar in (1, 10, 100):
    Qs = np.linspace(-1, 1, 100)
    slopes = [get_slope_estimate(_, gbar) for _ in Qs]
    axs[1].plot(Qs, slopes, label=r"$\bar{g}$=" + f"{gbar}")
axs[1].set_xlabel("label of reference molecule")
axs[1].set_ylabel(r"m")
axs[1].legend()
