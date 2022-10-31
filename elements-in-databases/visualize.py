# needs https://raw.githubusercontent.com/arosen93/ptable_trends/master/ptable_trends.py
#%%
import ptable_trends as pt
import pandas as pd
import collections
import materialsproject as mp
import numpy as np
import itertools as it
import matplotlib.pyplot as plt

# %%
df = pd.read_csv(
    "ani1x.csv", header=None, names=["db", "method", "element1", "element2", "count"]
)
# %%
elements = collections.Counter()
for row in df.itertuples():
    if row.element2 == "single":
        elements[row.element1] += row.count
# %%
df2 = pd.read_csv(
    "materialsproject.csv",
    header=None,
    names=["db", "method", "element1", "element2", "count"],
)
df
# %%
for row in df2.itertuples():
    if row.element2 == "single":
        elements[row.element1] += row.count
# %%
logged = [(_[0], np.log10(_[1])) for _ in elements.items() if _[1] != 0]
pd.DataFrame(logged, columns=["element", "count"]).to_csv(
    "elements.csv", index=False, header=False
)
# %%
pt.ptable_plotter(
    "elements.csv", output_filename="elements.html", show=True, cmap="viridis"
)
#%%
df = pd.concat([df, df2])
# %%
elements = mp.get_elements()
matrix = np.zeros((len(elements), len(elements)))
for row in df.query("method == 'DFT' and element2 != 'single'").itertuples():
    matrix[elements.index(row.element1), elements.index(row.element2)] += row.count
    assert elements.index(row.element1) < elements.index(row.element2)

for row in df.query("method == 'CCSD' and element2 != 'single'").itertuples():
    matrix[elements.index(row.element2), elements.index(row.element1)] += row.count
    assert elements.index(row.element1) < elements.index(row.element2)
# %%
plt.imshow(np.log10(matrix), interpolation="none")
for sep in (0, 2, 10, 18, 36, 54, 86, 100):
    plt.axvline(sep, color="red", lw=0.5)
    plt.axhline(sep, color="red", lw=0.5)
plt.colorbar()
plt.savefig("correlation.svg")
# %%
matrix.shape
# %%
