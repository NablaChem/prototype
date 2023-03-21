#%%
from sympy import solveset
from sympy.abc import x, y
from sympy import S, N, exp, Abs
import numpy as np
import matplotlib.pyplot as plt

# %%
solveset(exp(-Abs(x - 5)) + exp(-Abs(x - 10)) - 0.4, x)
# %%
def underlying(pos):
    return np.exp(-((pos - 5) ** 2)) + np.exp(-((pos + 5) ** 2))


xs = np.linspace(-20, 10, 100)
ys = underlying(xs)
plt.plot(xs, ys)
# %%
