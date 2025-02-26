# %%
import requests
import io
import tarfile
import numpy as np


def database_qm9(random_limit=3000):
    """Reads the QM9 database from network, http://www.nature.com/articles/sdata201422."""
    # exclusion list
    res = requests.get("https://ndownloader.figshare.com/files/3195404")
    exclusion_ids = [
        _.strip().split()[0] for _ in res.content.decode("ascii").split("\n")[9:-2]
    ]

    # geometries and energies
    res = requests.get("https://ndownloader.figshare.com/files/3195389")
    webfh = io.BytesIO(res.content)
    t = tarfile.open(fileobj=webfh)
    energies = []
    contents = []
    for fo in t:
        lines = t.extractfile(fo).read().decode("ascii").split("\n")
        natoms = int(lines[0])
        lines = lines[: 2 + natoms]
        lines = [_.replace("*^", "e") for _ in lines]
        molid = lines[1].strip().split()[0]
        if molid in exclusion_ids:
            continue
        energies.append(float(lines[1].strip().split()[12]))
        contents.append(lines)

    # random subset for development purposes
    idx = np.arange(len(energies))
    np.random.shuffle(idx)
    subset = idx[:random_limit]

    energies = [energies[_] for _ in subset]
    compounds = [contents[_] for _ in subset]
    return compounds, np.array(energies)


db = database_qm9(None)


# %%
def parse_qm9_xyz(lines):
    """Parses the QM9 xyz file format."""
    natoms = int(lines[0])
    elements = [_.split()[0] for _ in lines[2 : 2 + natoms]]
    xyz = np.array(
        [_.split()[1:4] for _ in lines[2 : 2 + natoms]], dtype=float
    ).tolist()
    return [elements, xyz]


# %%
import msgpack


def build_split():
    nmols = len(db[0])
    selected = np.random.choice(nmols, 5000, replace=False)
    payload = [parse_qm9_xyz(db[0][_]) + [db[1][_]] for _ in selected]
    return msgpack.packb(payload)


for s in range(20):
    with open(f"qm9-S{s:02d}.msgpack", "wb") as f:
        f.write(build_split())
# %%
import ase
import msgpack


def read_split(filename: str):
    with open(filename, "rb") as f:
        data = msgpack.unpackb(f.read())
    molecules = []
    labels = []
    for d in data:
        atoms = ase.Atoms(d[0], d[1])
        molecules.append(atoms)
        labels.append(d[2])
    return molecules, labels


q = read_split("qm9-S00.msgpack")
# %%
q[1]
# %%
