#%%
import h5py
import numpy as np
import materialsproject as mp
import itertools as it
import collections

# https://springernature.figshare.com/ndownloader/files/18112775
if __name__ == "__main__":
    f = h5py.File(
        "/home/ferchault/wrk/prototype/elements-in-databases/ani1xrelease.h5", "r"
    )

    keys_of_interest = {"CCSD": "ccsd(t)_cbs.energy", "DFT": "wb97x_dz.energy"}
    stats_ccsd = collections.Counter()
    stats_dft = collections.Counter()
    stats_single_ccsd = collections.Counter()
    stats_single_dft = collections.Counter()
    for grp in f.values():
        Nc = grp["coordinates"].shape[0]
        mask = np.ones(Nc, dtype=np.bool_)
        data = dict((k, grp[k][()]) for k in keys_of_interest.values())
        for k in keys_of_interest.values():
            v = data[k].reshape(Nc, -1)
            mask = mask & ~np.isnan(v).any(axis=1)
        if not np.sum(mask):
            continue
        atomic_numbers = grp["atomic_numbers"][()]
        num_ccsd = sum(~np.isnan(data[keys_of_interest["CCSD"]]))
        num_dft = sum(~np.isnan(data[keys_of_interest["DFT"]]))

        for Z_1, Z_2 in it.combinations(sorted(set(atomic_numbers)), 2):
            stats_ccsd[(Z_1, Z_2)] += num_ccsd
            stats_dft[(Z_1, Z_2)] += num_dft
            stats_single_ccsd[Z_1] += num_ccsd
            stats_single_ccsd[Z_2] += num_ccsd
            stats_single_dft[Z_1] += num_dft
            stats_single_dft[Z_2] += num_dft

    elements = mp.get_elements()
    for k, v in stats_dft.items():
        Z_1, Z_2 = k
        print(f"ANI1x,DFT,{elements[Z_1-1]},{elements[Z_2-1]},{v}")
    for k, v in stats_ccsd.items():
        Z_1, Z_2 = k
        print(f"ANI1x,CCSD,{elements[Z_1-1]},{elements[Z_2-1]},{v}")
    for k, v in stats_single_ccsd.items():
        print(f"ANI1x,CCSD,{elements[k-1]},single,{v}")
    for k, v in stats_single_dft.items():
        print(f"ANI1x,DFT,{elements[k-1]},single,{v}")
