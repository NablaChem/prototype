#%%
import pyscf.gto
import pyscf.scf
import numpy as np
import pyscf.dft
import requests
import io
import gzip
import tarfile
import itertools as it

#%%
mol = pyscf.gto.Mole(atom="""H 0 0 0; F 0 0 1.1""", basis="ccpvdz").build()
calc = pyscf.scf.RHF(mol)
calc.kernel()
# %%


def database_qmrxn20(kind: str, random_subsample=1000):
    """Reads transitition state geometries from network, https://iopscience.iop.org/article/10.1088/2632-2153/aba822."""
    # energies
    energiesurl = "https://archive.materialscloud.org/record/file?file_id=0eaa6011-b9d7-4c30-b424-2097dd90c77c&filename=energies.txt.gz&record_id=414"
    res = requests.get(energiesurl)
    webfh = io.BytesIO(res.content)
    with gzip.GzipFile(fileobj=webfh) as fh:
        lines = [_.decode("ascii") for _ in fh.readlines()]
    relevant = np.array([_ for _ in lines if f"{kind}/" in _ and ".xyz" in _])
    np.random.shuffle(relevant)
    relevant = relevant[:random_subsample]
    filenames = [line.strip().split(",")[4] for line in relevant]
    energies = np.array([float(line.strip().split(",")[-2]) for line in relevant])

    # geometries
    geometriesurl = "https://archive.materialscloud.org/record/file?file_id=4905b29e-a989-48a3-8429-32e1db989972&filename=geometries.tgz&record_id=414"
    res = requests.get(geometriesurl)
    webfh = io.BytesIO(res.content)
    t = tarfile.open(fileobj=webfh)
    mols = {}
    for fo in t:
        if fo.name in filenames:
            lines = t.extractfile(fo).readlines()
            lines = [_.decode("ascii") for _ in lines]
            mols[fo.name] = lines
    cs = [mols[_] for _ in filenames]
    return cs, energies, filenames


def lines2mol(lines: list[str], basisset: str) -> pyscf.gto.Mole:
    atomlines = ";".join([_.strip() for _ in lines[2:]])
    mol = pyscf.gto.Mole(atom=atomlines, basis=basisset).build()
    return mol


# %%
if __name__ == "__main__":
    db = database_qmrxn20("reactant-conformers", 10)

    for lines, energy, filename in zip(*db):
        mol = lines2mol(lines, "cc-pvdz")

        calcs = dict()
        for method in "SVWN PBE PBE0 B3LYP SCAN".split():
            calcs[method] = pyscf.dft.RKS(mol, xc=method).density_fit().run()
            if not calcs[method].converged:
                break

        calchf = pyscf.scf.RHF(mol).density_fit().run()
        if not calchf.converged:
            break
        calccc = pyscf.cc.CCSD(calchf).density_fit().run()
        if not calccc.converged:
            break

        grid = pyscf.dft.gen_grid.Grids(mol)
        grid.level = 3
        grid.build()
        ao_value = pyscf.dft.numint.eval_ao(mol, grid.coords, deriv=0)
        rhos = dict()
        for key in calcs.keys():
            rhos[key] = pyscf.dft.numint.eval_rho(
                mol, ao_value, calcs[key].make_rdm1(), xctype="LDA"
            )

        rhos["HF"] = pyscf.dft.numint.eval_rho(
            mol, ao_value, calchf.make_rdm1(), xctype="LDA"
        )

        dm1 = calccc.make_rdm1()
        dm1_ao = np.einsum("pi,ij,qj->pq", calchf.mo_coeff, dm1, calchf.mo_coeff.conj())
        rhos["CCSD"] = pyscf.dft.numint.eval_rho(mol, ao_value, dm1_ao, xctype="LDA")

        for key1, key2 in it.combinations(rhos.keys(), 2):
            nelectrons = sum(rhos[key1] * grid.weights)
            absolute_density_difference = np.abs(rhos[key1] - rhos[key2])
            print(
                key1,
                key2,
                np.sum(absolute_density_difference * grid.weights) / nelectrons,
            )

# %%
