#%%
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import polynomial_kernel
import scipy.linalg as scl
import sympy as sp


def f_unknown(x):
    return x[0] ** 2 / 2 + x[1] ** 2 - 1


def generate_points(n):
    xs = np.random.random((n, 2)) * 3 - 1.5
    y = f_unknown(xs.T)
    return xs, y


def build_analytical_expression(
    degree: int, alphas: np.ndarray, coef0: float, gamma: float, xs: np.ndarray
) -> sp.polys.polytools.Poly:
    """
    Builds an analytical expression for a polynomial kernel given the degree, dual coefficients, intercept,
    gamma parameter, and input points.

    Args:
    degree (int): The degree of the polynomial kernel.
    alphas (np.ndarray): An array of dual coefficients.
    coef0 (float): The intercept of the kernel function.
    gamma (float): The gamma parameter of the kernel function.
    xs (np.ndarray): An array of input points.

    Returns:
    kernel (sympy.polys.polytools.Poly): An analytical expression for the polynomial kernel.
    """
    x = sp.Symbol("x")
    y = sp.Symbol("y")
    kernel = 0

    for k in range(len(xs)):
        kernel += alphas[k] * gamma * (x * xs[k, 0] + y * xs[k, 1] + coef0) ** degree

    return sp.Poly(kernel.simplify())


def prune_small_terms(
    poly: sp.polys.polytools.Poly, threshold: float
) -> sp.polys.polytools.Poly:
    """
    Prunes small terms from a polynomial expression by setting the coefficients below the threshold to zero.

    Args:
    poly (sympy.polys.polytools.Poly): The polynomial expression to be pruned.
    threshold (float): The threshold value below which coefficients will be set to zero.

    Returns:
    poly (sympy.polys.polytools.Poly): The pruned polynomial expression.
    """
    to_remove = [abs(i) for i in poly.coeffs() if abs(i) < threshold]
    for i in to_remove:
        poly = poly.subs(i, 0)
    return poly


def test_run(degree, npoints, coef0=1, gamma=1):
    xs, y = generate_points(npoints)

    plt.scatter(xs[:, 0], xs[:, 1], c=y)
    plt.colorbar()

    K = polynomial_kernel(xs, xs, degree=degree, coef0=coef0, gamma=gamma)
    alphas = scl.solve(K, y)

    poly = build_analytical_expression(degree, alphas, coef0, gamma, xs)
    print(prune_small_terms(poly, 1e-10))
    plt.show()


test_run(3, 10)
test_run(5, 100)
test_run(10, 100)

# %%
