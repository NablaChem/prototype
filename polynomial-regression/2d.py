#%%
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import polynomial_kernel
import scipy.linalg as scl
import sympy as sp
import scipy.optimize as sco
from typing import Tuple, Callable


def f_unknown(x):
    return x[0] ** 2 / 2 + x[1] ** 2 - 1


def generate_random_points(n):
    xs = np.random.random((n, 2)) * 3 - 1.5
    y = f_unknown(xs.T)
    return xs, y


def generate_points_on_curve(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates a set of n points on an unknown curve.

    Args:
    n (int): The number of points to generate.

    Returns:
    points (np.ndarray): An array of shape (n, 2) containing the generated points.
    ys (np.ndarray): An array of shape (n,) containing the corresponding y-values of the points.
    """
    points = []
    ys = []
    for i in range(n):
        xy = np.random.random(2) * 3 - 1.5

        def target(yval):
            return abs(f_unknown((xy[0], yval)))

        xy[1] = sco.minimize_scalar(target, options={"xtol": 1e-14, "maxiter": 10000}).x
        points.append(xy)
        ys.append(f_unknown(xy))
    ys = np.array(ys)
    # workaround: otherwise the KRR model has enough curvature to find a noisy but accurate result
    ys *= 0
    return np.array(points), ys


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

    poly = kernel.simplify().as_poly(x, y)
    poly = poly.set_domain(sp.RR)
    return poly


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

    if poly == 0:
        return 0
    else:
        return poly


def test_run(
    generator: Callable[[int], Tuple[np.ndarray, np.ndarray]], degree: int, npoints: int
) -> None:
    """
    Runs a test of kernel ridge regression with a polynomial kernel on a generated dataset.

    Args:
    generator (Callable[[int], Tuple[np.ndarray, np.ndarray]]): A function that generates a dataset.
    degree (int): The degree of the polynomial kernel to use.
    npoints (int): The number of points to generate in the dataset.
    """
    xs, y = generator(npoints)

    # show data
    # plt.scatter(xs[:, 0], xs[:, 1], c=y)
    # plt.colorbar()
    # plt.show()

    # build kernel and solve
    coef0 = 1
    gamma = 1
    K = polynomial_kernel(xs, xs, degree=degree, coef0=coef0, gamma=gamma)
    alphas = scl.solve(K, y)

    # convert to simple analytical expression
    poly = build_analytical_expression(degree, alphas, coef0, gamma, xs)
    print("Found expression", prune_small_terms(poly, 1e-10))


test_run(generate_random_points, 3, 10)

test_run(generate_points_on_curve, 3, 10)
