#%%
import matplotlib.pyplot as plt
import numpy as np

# %%
def read_data():
    with open("densitycompare.log") as fh:
        lines = fh.readlines()

    lines = [_.strip() for _ in lines if "RCCSD" not in _ and "converged" not in _ and "HF" not in _]

    methods = []
    for line in lines:
        method1, method2, densitydifference = line.split()
        methods.append(method1)
        methods.append(method2)
    
    methods = sorted(set(methods))
    matrix = np.zeros((len(methods), len(methods)))
    methods = "SVWN PBE PBE0 B3LYP SCAN CCSD".split()
    for line in lines:
        method1, method2, densitydifference = line.split()
        i = methods.index(method1)
        j = methods.index(method2)
        matrix[i, j] = float(densitydifference)
        matrix[j, i] = float(densitydifference)

    return methods, matrix  
methods, matrix = read_data()
# %%
plt.imshow(matrix*1000, cmap="viridis")
plt.xticks(np.arange(len(methods)), methods)
plt.yticks(np.arange(len(methods)), methods)
plt.colorbar()
# %%
methods
# %%
