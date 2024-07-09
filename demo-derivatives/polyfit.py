# %%
import numpy as np


def A(x):
    return np.sin(x)


def B(x):
    return 1 + 2 * x + 3 * x**2 + 4 * x**3 + 5 * x**4


def fit_error(func, h):
    xs = np.linspace(-2 * h, 2 * h, 5)
    polycoeffs = np.polyfit(xs, func(xs), 4)
    poly = np.poly1d(polycoeffs)
    return abs(poly(xs) - func(xs)).mean()


for func in (A, B):
    for i in range(1, 10):
        print(func.__name__, fit_error(func, 10**-i))

# %%
import findiff


def finite_difference_error(func, h):
    def do_diff(func, h, deriv, acc):
        stencil = findiff.coefficients(deriv, acc=acc)["center"]
        result = 0
        for x, w in zip(stencil["offsets"], stencil["coefficients"]):
            result += w * func(x * h)
        return result / h**deriv

    zeroth = func(0)
    first = do_diff(func, h, 1, 4)
    second = do_diff(func, h, 2, 4) / 2
    third = do_diff(func, h, 3, 2) / 2 / 3
    fourth = do_diff(func, h, 4, 2) / 2 / 3 / 4
    polycoeffs = [zeroth, first, second, third, fourth][::-1]
    poly = np.poly1d(polycoeffs)
    xs = np.linspace(-2 * h, 2 * h, 5)
    return abs(poly(xs) - func(xs)).mean()


for func in (A, B):
    for i in range(1, 10):
        print(
            func.__name__,
            fit_error(func, 10**-i),
            finite_difference_error(func, 10**-i),
        )

# %%
output = """A 1.2277657352051296e-16 3.8857805861880476e-17
A 1.015586448078255e-17 3.469446951953614e-18
A 5.387244195577897e-19 1.3010426069826053e-19
A 2.9988162581621696e-20 3.794707603699265e-20
A 2.0112902071963098e-21 3.726944967918922e-21
A 5.646285716090796e-22 2.541098841762901e-22
A 5.39340949170381e-23 3.970466940254533e-23
A 7.905730710257794e-24 3.639594695233322e-24
A 1.4607547993884565e-24 2.481541837659083e-25
B 9.325873406851315e-16 2.220446049250313e-16
B 4.662936703425658e-16 6.661338147750939e-17
B 1.9984014443252818e-16 4.4408920985006264e-17
B 1.7763568394002506e-16 1.1102230246251565e-16
B 6.661338147750939e-16 8.881784197001253e-17
B 9.992007221626409e-16 6.661338147750939e-17
B 2.220446049250313e-16 1.7763568394002506e-16
B 9.103828801926283e-16 1.1102230246251565e-16
B 5.329070518200751e-16 1.554312234475219e-16"""
