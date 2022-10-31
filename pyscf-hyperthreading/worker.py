#%%
import leruli as lrl
import pyscf.scf
import pyscf.cc
import pyscf.gto
import sys
import resource
import time


def get_mol(molname: str) -> pyscf.gto.Mole:
    lines = lrl.graph_to_geometry(lrl.name_to_graph(molname)["graph"], "XYZ")[
        "geometry"
    ]
    lines = ";".join(lines.split("\n")[2:])
    mol = pyscf.gto.Mole(atom=lines, basis="cc-pvdz")
    mol.verbose = 0
    mol.build()
    return mol


# %%
# cases: 1,2,4,8,16,32 cores with HT on and off for HF, PBE, CCSD. Output: time, memory, molname
if __name__ == "__main__":
    molname = sys.argv[1]
    method = sys.argv[2]

    mol = get_mol(molname)
    start = time.time()
    if method == "HF":
        calc = pyscf.scf.RHF(mol)
        calc.kernel()
    if method == "PBE":
        calc = pyscf.scf.RKS(mol)
        calc.xc = "pbe"
        calc.kernel()
    if method == "CCSD":
        calc = pyscf.cc.RCCSD(pyscf.scf.RHF(mol).run())
        calc.kernel()

    memory_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    end = time.time()
    print(f"{molname},{method},{end-start},{memory_mb}")

# %%
