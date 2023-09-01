# %%
import numpy as np
import scipy.spatial.distance as dist


def get_adj(positions, cutoff=1.6):
    distances = dist.squareform(dist.pdist(positions))
    adj = distances < cutoff
    adj[np.diag_indices_from(adj)] = False
    return adj


def lookup_energy(adj_matrix, nuclear_charges, lookup):
    """
    determine local environment and add the corresponding energy
    """
    energy = 0

    for i in range(len(adj_matrix)):
        central = nuclear_charges[i]
        environment = tuple(nuclear_charges[adj_matrix[i]])
        energy += lookup2[central][environment]
    return energy


# read the lookup table
# with open("/data/sahre/projects/finite_differences/QM9/results/insights/alchemy_mean_energy_lookup_tuple.pkl", "rb") as file:
#     lookup = pickle.load(file)

lookup = {
    (1, 6): -63.86822604784631,
    (1, 8): -46.320151912450804,
    (6, 1, 1, 1, 6): -299.26107752205,
    (6, 1, 1, 6, 8): -257.4238825397997,
    (6, 6, 6, 6, 8): -222.81480998003656,
    (8, 1, 6): -266.47536025444725,
}
lookup2 = {}
for k, v in lookup.items():
    central = k[0]
    environment = k[1:]
    if central not in lookup2:
        lookup2[central] = {}
    lookup2[central][environment] = v

# get the molecule
# compound = sys.argv[1]
# mol = aio.read(f'/data/sahre/projects/finite_differences/QM9/compounds_vsc5/{compound}.xyz')


def do_one():
    nuclear_charges = np.array([6, 6, 6, 8, 6, 8, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    positions = np.array(
        [
            [9.57650777, 11.45371265, 10.09740365],
            [9.63272567, 9.92157225, 10.09756663],
            [10.34795212, 9.38951448, 11.34495818],
            [8.31347336, 9.37256364, 10.00512278],
            [10.315151, 9.40693106, 8.82402372],
            [11.64986651, 9.88971925, 8.79989747],
            [10.58194422, 11.87794121, 10.04107691],
            [8.99798968, 11.80828289, 9.23913793],
            [9.09731447, 11.82387826, 11.01123605],
            [9.86648994, 9.76295008, 12.25626299],
            [10.3114137, 8.29616554, 11.36130634],
            [11.39412513, 9.70488515, 11.35557034],
            [7.82370643, 9.66037286, 10.7822227],
            [10.27991158, 8.30750253, 8.8301218],
            [9.73956543, 9.75739502, 7.95461865],
            [12.07186299, 9.56661314, 7.99947388],
        ]
    )

    order = np.argsort(nuclear_charges)
    nuclear_charges = nuclear_charges[order]
    positions = positions[order]
    adj_matrix = get_adj(positions)

    # find local environments and add up their energies
    energy = lookup_energy(adj_matrix, nuclear_charges, lookup)
    # print(f'Estimated energy = {energy}')
    return energy


do_one()
# %%
import timeit

N = 100000
print(timeit.timeit(do_one, number=N) / N * 1000, "ms")

# %%
