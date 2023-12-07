# %%
import pandas as pd
import findiff
import numpy as np
import collections
import math
import itertools as it
from typing import Union


class Monomial:
    def __init__(self, prefactor, powers={}):
        self._powers = powers
        self._prefactor = prefactor

    def __repr__(self):
        return f"Monomial({self._prefactor}, {self._powers})"

    def prefactor(self):
        return self._prefactor / np.prod(
            [math.factorial(_) for _ in self._powers.values()]
        )

    def distance(self, pos, center):
        ret = []
        for column, power in self._powers.items():
            ret.append((pos[column] - center[column]) ** power)
        return np.prod(ret)


class MultiTaylor:
    def __init__(self, dataframe, outputs):
        self._dataframe = dataframe
        self._outputs = outputs
        self._filtered = dataframe
        self._filter = {}

    def reset_center(self, **kwargs):
        self._center = kwargs

    def reset_filter(self, **kwargs):
        self._filtered = self._dict_filter(self._dataframe, kwargs)
        self._filter = kwargs

    def _dict_filter(self, df, filter):
        return df.loc[(df[list(filter)] == pd.Series(filter)).all(axis=1)]

    def _check_uniqueness(self, df, terms):
        copy = df.copy()

        # remove variable column
        for term in terms:
            del copy[term]

        # remove output columns: they are arbitrary
        for output in self._outputs:
            del copy[output]

        copy.drop_duplicates(inplace=True)
        if len(copy.index) > 1 and len(copy.columns) > 0:
            print(copy)
            print(copy.index)
            print(copy.columns)
            raise ValueError(f"Terms {terms} are not unique. Is a filter missing?")

    def _split_term(self, term, order):
        parts = term.split("_")
        if len(parts) == 1:
            parts = [parts[0]] * order
        if len(parts) != order:
            raise ValueError(f"Term {term} has the wrong order.")

        return dict(collections.Counter(parts))

    def _offsets_from_df(self, df, variable_columns):
        offsets = np.zeros((len(variable_columns), len(df)), dtype=float)
        spacings = dict()

        for column in variable_columns:
            unique_values = np.sort(df[column].unique())
            spacing = np.diff(unique_values)
            if not np.allclose(spacing, spacing.mean()):
                raise ValueError(f"Variable {column} is not evenly spaced.")
            offsets[variable_columns.index(column)] = (
                df[column].values - self._center[column]
            ) / spacing.mean()
            spacings[column] = spacing.mean()

        return [tuple(_) for _ in offsets.T], [spacings[_] for _ in variable_columns]

    def _build_monomials(self, df, term, order):
        variable_columns = list(set(term.split("_")))
        offsets, spacings = self._offsets_from_df(df, variable_columns)
        assert len(offsets) == len(set(offsets))
        terms = self._split_term(term, order)
        partials = tuple([terms[_] for _ in variable_columns])

        try:
            stencil = findiff.stencils.Stencil(
                offsets, partials={partials: 1}, spacings=spacings
            )
        except:
            raise ValueError(f"Could not build stencil for term {term}.")
        if len(stencil.values) == 0:
            print(df)
            print(order)
            print(offsets)
            print(variable_columns)
            print(partials)
            raise ValueError(f"Not enough points for term {term}.")

        for output in self._outputs:
            weights = [stencil.values[_] if _ in stencil.values else 0 for _ in offsets]
            values = df[output].values

            self._monomials[output].append(
                Monomial(
                    prefactor=np.dot(weights, values),
                    powers=terms,
                )
            )

    def _all_terms_up_to(self, order):
        terms = {}
        data_columns = [_ for _ in self._filtered.columns if _ not in self._outputs]
        for order in range(1, order + 1):
            terms[order] = [
                "_".join(_)
                for _ in it.combinations_with_replacement(data_columns, order)
            ]
        return terms

    def build_model(self, orders: Union[int, dict[int, list[str]]]):
        # check center: there can be only one
        center_rows = self._dict_filter(self._dataframe, self._center)
        center_row = self._dict_filter(center_rows, self._filter)
        if len(center_row) == 0:
            raise NotImplementedError(f"Center is not in the dataframe.")
        if len(center_row) > 1:
            raise ValueError(f"Center is not unique.")

        # setup constant term
        self._monomials = {k: [Monomial(center_row.iloc[0][k])] for k in self._outputs}

        # accept integer as placeholder
        if isinstance(orders, int):
            orders = self._all_terms_up_to(orders)

        # setup other terms
        for order, terms in orders.items():
            for term in terms:
                if order == 1:
                    other_fields = {k: v for k, v in self._center.items() if k != term}
                else:
                    other_fields = {
                        k: v for k, v in self._center.items() if k not in term
                    }
                s = self._dict_filter(self._filtered, other_fields)
                self._check_uniqueness(s, self._split_term(term, order).keys())
                self._build_monomials(s, term, order)

    def query(self, **kwargs):
        ret = {}
        for output in self._outputs:
            ret[output] = 0
            for monomial in self._monomials[output]:
                prefactor = monomial.prefactor()
                ret[output] += prefactor * monomial.distance(kwargs, self._center)
        return ret


# %%
def test_1d():
    for i in range(100):
        polynomial = np.poly1d(np.random.random(5))
        x = 4
        dx = 0.1
        xs = (np.arange(5) - 2) * dx + x
        ys = polynomial(xs)
        df = pd.DataFrame({"x": xs, "y": ys})
        mt = MultiTaylor(df, outputs=["y"])
        mt.reset_center(x=4)
        mt.build_model({1: ["x"], 2: ["x"], 3: ["x"], 4: ["x"]})
        assert abs(mt.query(x=5)["y"] - polynomial(5)) < 1e-4


def test_2d():
    polynomial = lambda A, B: A * B * B * B + B * B + A
    A, B = 4, 5
    dx = 0.1
    As = (np.arange(5) - 2) * dx + A
    Bs = (np.arange(5) - 2) * dx + B
    As, Bs = np.meshgrid(As, Bs)
    As = As.flatten()
    Bs = Bs.flatten()
    ys = polynomial(As, Bs)
    df = pd.DataFrame({"A": As, "B": Bs, "y": ys})
    mt = MultiTaylor(df, outputs=["y"])
    mt.reset_center(A=4, B=5)
    mt.build_model(4)
    assert abs(mt.query(A=5, B=6)["y"] - polynomial(5, 6)) < 1e-4


def test_3d():
    polynomial = lambda A, B, C: A * B * C + B * B * C + A * C
    A, B, C = 4, 5, 6
    dx = 0.1
    As = (np.arange(5) - 2) * dx + A
    Bs = (np.arange(5) - 2) * dx + B
    Cs = (np.arange(5) - 2) * dx + C
    As, Bs, Cs = np.meshgrid(As, Bs, Cs)
    As = As.flatten()
    Bs = Bs.flatten()
    Cs = Cs.flatten()
    ys = polynomial(As, Bs, Cs)
    df = pd.DataFrame({"A": As, "B": Bs, "C": Cs, "y": ys})
    mt = MultiTaylor(df, outputs=["y"])
    mt.reset_center(A=A, B=B, C=C)
    mt.build_model(3)
    assert (
        abs(mt.query(A=A + 1, B=B + 1, C=C + 1)["y"] - polynomial(A + 1, B + 1, C + 1))
        < 1e-4
    )
