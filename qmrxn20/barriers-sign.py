#%%
import requests
import io
import gzip
import pandas as pd


def get_barriers():
    # url = "https://archive.materialscloud.org/record/file?file_id=dfb1493e-ca0e-4b1b-9f27-b4ba576e527f&filename=barriers.txt.gz&record_id=414"
    res = requests.get(url)
    webfh = io.BytesIO(res.content)
    with gzip.GzipFile(fileobj=webfh) as fh:
        return pd.read_csv(fh, index_col=0).reset_index()


df = get_barriers()
# %%
import matplotlib.pyplot as plt

# %%
df.plot.scatter("activation", "activation")
df
# %%
# keep only entries with explicit reactant geometries
df = df[~df.filename_r.isna()]
# %%
df.query("activation< 0")
# %%
df.groupby("reactant").count()
# %%
len(df.label.unique())
# %%
