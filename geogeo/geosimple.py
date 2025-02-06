# %%
import rdkit
import rdkit.Chem
import sympy as sp
import scipy.optimize as sco
import numpy as np

from sympy.printing.pycode import pycode


def build_equations(smiles: str):
    print("#" * 100)
    print(f"case {smiles}")
    mol = rdkit.Chem.MolFromSmiles(smiles)
    mol = rdkit.Chem.AddHs(mol)
    adj = rdkit.Chem.GetAdjacencyMatrix(mol)
    natoms = mol.GetNumAtoms()
    elements = [a.GetSymbol() for a in mol.GetAtoms()]
    charges = [a.GetAtomicNum() for a in mol.GetAtoms()]
    print("elements=", elements)
    print("adj=", adj)

    bond_lengths = {("C", "C"): 1.54, ("C", "H"): 1.09, ("H", "O"): 0.96}

    ndims = 3 * natoms
    x = list(sp.symbols(f"x:{ndims}"))
    aval = 1
    bval = 0.2

    a, b = sp.symbols("a b")
    total_energy = 0
    for i in range(natoms):
        xi, yi, zi = x[i * 3], x[i * 3 + 1], x[i * 3 + 2]
        for j in range(i + 1, natoms):
            xj, yj, zj = x[j * 3], x[j * 3 + 1], x[j * 3 + 2]
            rsquared = (xi - xj) ** 2 + (yi - yj) ** 2 + (zi - zj) ** 2

            if adj[i, j]:
                bond_length = bond_lengths[tuple(sorted((elements[i], elements[j])))]
                total_energy += a * ((bond_length**2 - rsquared) ** 2)
            else:
                total_energy += b * (charges[i] * charges[j] / rsquared)

    skip = (0, 1, 2, 3, 4, 6)
    for zero_out in sorted(skip)[::-1]:
        total_energy = total_energy.subs(x[zero_out], 0)
        del x[zero_out]

    callable = sp.lambdify(x, total_energy.subs({a: aval, b: bval}))
    res = sco.minimize(lambda x: callable(*x), 3 * np.random.normal(0, 1, len(x)))
    if res.success:
        print("Minimum", res.x)

    coords = np.zeros(ndims)
    coords[5] = res.x[0]
    coords[7:] = res.x[1:]
    coords = coords.reshape(-1, 3)
    with open("/tmp/coords.xyz", "w") as f:
        f.write(f"{natoms}\n\n")
        for element, coord in zip(elements, coords):
            f.write(f"{element} {' '.join(map(str, coord))}\n")

    print("############ Julia code")
    print("a=", aval)
    print("b=", bval)
    print(f"variables = [{', '.join(map(str, x))}]")
    print("potential=", pycode(total_energy).replace("**", "^"))

    return coords


examples = ["CC", "C=C", "C#C", "C", "O"]
for example in examples:
    build_equations(example)

# %%
