# %%
import numpy as np


def read_reference(lines):
    # atomspec
    natoms = 15
    atomspec = ";".join([_.strip() for _ in lines[2 : 2 + natoms]])

    # MBS
    started = False
    ref = np.zeros((natoms, natoms))
    for line in lines:
        if "MBS Condensed to atoms (all electrons)" in line:
            started = True
            continue
        if started:
            if " " * 12 in line:
                # header
                section_indices = [int(_) - 1 for _ in line.strip().split()]
                continue
            if len(line.strip()) == 0:
                break
            parts = line.strip().split()
            row_index = int(parts[0]) - 1
            for col_index, value in zip(section_indices, parts[2:]):
                ref[row_index, col_index] = float(value)

    return atomspec, ref


# atomspec, refdata = read_reference(
#    "/home/ferchault/wrk/prototype/minpop/dsgdb9nsd_001000.txt"
# )
atomspec = (
    r"""C,0,0.0234395981,1.445343
 0312,0.0421084617\N,0,0.0060615134,-0.0040415917,0.0263854155\C,0,-0.0
 316410872,-0.8101470708,-1.0814610232\N,0,-0.0384525996,-2.0830828391,
 -0.7654487215\C,0,-0.0036644281,-2.112212563,0.6106291669\C,0,0.024514
 5326,-0.840819084,1.1314103328\C,0,0.0662206077,-0.3342545716,2.531594
 1013\H,0,0.9288495405,1.8213133306,0.5246315853\H,0,0.0015391082,1.807
 7786327,-0.9849482601\H,0,-0.8471947544,1.8421269737,0.5698487241\H,0,
 -0.052384671,-0.4013270272,-2.0807335516\H,0,-0.0006359715,-3.04364252
 81,1.1559702974\H,0,0.9618887011,0.265076309,2.7282689421\H,0,-0.80381
 86086,0.2859969856,2.7732882023\H,0,0.0738803686,-1.1769220771,3.22360
 0787""".replace(
        "\n ", ""
    )
    .replace("\\", "; ")
    .replace(",", " ")
    .replace(" 0 ", " ")
)
refdata = """          MBS Condensed to atoms (all electrons):
               1          2          3          4          5          6
     1  C    4.666698   0.332831  -0.025363   0.000847   0.001426  -0.025277
     2  N    0.332831   6.382460   0.412191  -0.045319  -0.064112   0.399528
     3  C   -0.025363   0.412191   4.737045   0.486736  -0.082275  -0.075576
     4  N    0.000847  -0.045319   0.486736   6.649470   0.405788  -0.044960
     5  C    0.001426  -0.064112  -0.082275   0.405788   4.855945   0.551238
     6  C   -0.025277   0.399528  -0.075576  -0.044960   0.551238   4.820390
     7  C   -0.001729  -0.021131   0.001257   0.000543  -0.018330   0.390352
     8  H    0.380144  -0.022893   0.000783  -0.000016   0.000023  -0.001521
     9  H    0.382593  -0.020888  -0.004412   0.000029  -0.000039   0.001475
    10  H    0.380143  -0.022893   0.000783  -0.000016   0.000023  -0.001521
    11  H   -0.001466  -0.023704   0.393141  -0.019962   0.002646   0.001643
    12  H   -0.000018   0.001113   0.002679  -0.020485   0.394889  -0.023422
    13  H   -0.000021  -0.001108   0.000025  -0.000014   0.000407  -0.023361
    14  H   -0.000021  -0.001108   0.000025  -0.000014   0.000408  -0.023360
    15  H    0.000015   0.000943  -0.000030   0.000019  -0.003017  -0.021734
               7          8          9         10         11         12
     1  C   -0.001729   0.380144   0.382593   0.380143  -0.001466  -0.000018
     2  N   -0.021131  -0.022893  -0.020888  -0.022893  -0.023704   0.001113
     3  C    0.001257   0.000783  -0.004412   0.000783   0.393141   0.002679
     4  N    0.000543  -0.000016   0.000029  -0.000016  -0.019962  -0.020485
     5  C   -0.018330   0.000023  -0.000039   0.000023   0.002646   0.394889
     6  C    0.390352  -0.001521   0.001475  -0.001521   0.001643  -0.023422
     7  C    4.744194   0.000012   0.000018   0.000012  -0.000015  -0.000861
     8  H    0.000012   0.586007  -0.021643  -0.025897  -0.000007  -0.000000
     9  H    0.000018  -0.021643   0.575109  -0.021644   0.000940   0.000001
    10  H    0.000012  -0.025897  -0.021644   0.586009  -0.000007  -0.000000
    11  H   -0.000015  -0.000007   0.000940  -0.000007   0.513859  -0.000035
    12  H   -0.000861  -0.000000   0.000001  -0.000000  -0.000035   0.528068
    13  H    0.376445   0.000240  -0.000001  -0.000109  -0.000001   0.000003
    14  H    0.376446  -0.000109  -0.000001   0.000240  -0.000001   0.000003
    15  H    0.380683  -0.000002  -0.000000  -0.000002   0.000001   0.000456
              13         14         15
     1  C   -0.000021  -0.000021   0.000015
     2  N   -0.001108  -0.001108   0.000943
     3  C    0.000025   0.000025  -0.000030
     4  N   -0.000014  -0.000014   0.000019
     5  C    0.000407   0.000408  -0.003017
     6  C   -0.023361  -0.023360  -0.021734
     7  C    0.376445   0.376446   0.380683
     8  H    0.000240  -0.000109  -0.000002
     9  H   -0.000001  -0.000001  -0.000000
    10  H   -0.000109   0.000240  -0.000002
    11  H   -0.000001  -0.000001   0.000001
    12  H    0.000003   0.000003   0.000456
    13  H    0.603073  -0.024567  -0.021300
    14  H   -0.024567   0.603073  -0.021299
    15  H   -0.021300  -0.021299   0.581459"""
refdata = read_reference(refdata.split("\n"))
refdata[1]
# %%
import pyscf
import numpy as np
import pyscf.lo
from pyscf.scf import addons
from pyscf.scf import hf
import pyscf.gto
import scipy.linalg


def minpop(calculation: pyscf.scf.hf.RHF):
    minimal = pyscf.gto.M(atom=calculation.mol.atom, basis="STO-3G")

    # A complete basis set model chemistry. VII. Use of the minimum population localization method
    # J. A. Montgomery, Jr., M. J. Frisch, and J. W. Ochterski, G. A. Petersson
    # DOI 10.1063/1.481224, eqn 3
    Sbar = pyscf.gto.intor_cross("int1e_ovlp", minimal, calculation.mol)
    C = calculation.mo_coeff[:, calculation.mo_occ > 0]
    Sprime = pyscf.gto.intor_cross("int1e_ovlp", minimal, minimal)
    P = Sbar @ C
    PL = scipy.linalg.sqrtm(np.linalg.inv(Sprime)) @ P
    Sprimeinv = np.linalg.inv(Sprime)
    Cprime = PL @ scipy.linalg.sqrtm(np.linalg.inv(C.T @ Sbar.T @ Sprimeinv @ P))
    # correction: typo in eq 3
    Cprime = scipy.linalg.sqrtm(np.linalg.inv(Sprime)) @ Cprime

    pm = pyscf.lo.PM(minimal, Cprime)
    pm.pop_method = "mulliken"
    loc_orb = pm.kernel()

    s = hf.get_ovlp(minimal)
    O = calculation.mo_occ[calculation.mo_occ > 0]
    dm = (loc_orb * O) @ loc_orb.T
    pop = np.einsum("ij,ji->ij", dm, s).real

    population = np.zeros((minimal.natm, minimal.natm))
    for i, si in enumerate(minimal.ao_labels(fmt=None)):
        for j, sj in enumerate(minimal.ao_labels(fmt=None)):
            population[si[0], sj[0]] += pop[i, j]

    return population


calculation = pyscf.scf.RHF(pyscf.gto.M(atom=atomspec, basis="6-31+G"))
calculation.kernel()
custom = minpop(calculation)


# %%
def minpop(calculation: pyscf.scf.hf.RHF):
    mollow = pyscf.gto.M(atom=calculation.mol.atom, basis="STO-3G")
    # mflow = pyscf.scf.RHF(mollow)

    molhigh = pyscf.gto.M(atom=atomspec, basis=basis)
    mfhigh = pyscf.scf.RHF(molhigh)
    mfhigh.kernel()

    # A complete basis set model chemistry. VII. Use of the minimum population localization method
    # J. A. Montgomery, Jr., M. J. Frisch, and J. W. Ochterski, G. A. Petersson
    # DOI 10.1063/1.481224, eqn 3
    Sbar = pyscf.gto.intor_cross("int1e_ovlp", mollow, molhigh)
    C = mfhigh.mo_coeff[:, mfhigh.mo_occ > 0]
    Sprime = mflow.get_ovlp()
    P = Sbar @ C
    PL = scipy.linalg.sqrtm(np.linalg.inv(Sprime)) @ P
    Sprimeinv = np.linalg.inv(Sprime)
    Cprime = PL @ scipy.linalg.sqrtm(np.linalg.inv(C.T @ Sbar.T @ Sprimeinv @ Sbar @ C))
    Cprime = scipy.linalg.sqrtm(np.linalg.inv(mflow.get_ovlp())) @ Cprime
    # literal expression from , same result

    pm = pyscf.lo.PM(mollow, Cprime, mflow)
    pm.pop_method = "mulliken"
    loc_orb = pm.kernel()
    mo_in_ao = loc_orb

    s = hf.get_ovlp(mollow)
    dm = mflow.make_rdm1(mo_in_ao, mfhigh.mo_occ[mfhigh.mo_occ > 0])
    pop = np.einsum("ij,ji->ij", dm, s).real

    population = np.zeros((mollow.natm, mollow.natm))
    for i, si in enumerate(mollow.ao_labels(fmt=None)):
        for j, sj in enumerate(mollow.ao_labels(fmt=None)):
            population[si[0], sj[0]] += pop[i, j]

    return population


custom = minpop(atomspec, "6-31+G")


# %%
mollow = pyscf.gto.M(atom=atomspec, basis="STO-3G")
mflow = pyscf.scf.RHF(mollow)
mflow.kernel()

# %%

molhigh = pyscf.gto.M(atom=atomspec, basis="6-31+G")
mfhigh = pyscf.scf.RHF(molhigh)
mfhigh.kernel()

Sbar = pyscf.gto.intor_cross("int1e_ovlp", mollow, molhigh)
C = mfhigh.mo_coeff[:, mfhigh.mo_occ > 0]
Sprime = mflow.get_ovlp()

P = Sbar @ C
PL = scipy.linalg.sqrtm(np.linalg.inv(Sprime)) @ P
Sprimeinv = np.linalg.inv(Sprime)
Cprime = PL @ scipy.linalg.sqrtm(np.linalg.inv(C.T @ Sbar.T @ Sprimeinv @ Sbar @ C))
# %%
print("nelec low", np.trace(mflow.make_rdm1() @ mflow.get_ovlp()))
print("nelec high", np.trace(mfhigh.make_rdm1() @ mfhigh.get_ovlp()))
dm = mflow.make_rdm1(
    scipy.linalg.sqrtm(np.linalg.inv(mflow.get_ovlp())) @ Cprime,
    mfhigh.mo_occ[mfhigh.mo_occ > 0],
)
print("nelec loc", np.trace(dm @ mflow.get_ovlp()))
# %%

# %%
###### do not use: tries to use the high-level functions in pyscf, but that would be
###### inconvenient for (automatic) differentiability
import pyscf
import numpy as np
import pyscf.lo
from pyscf.scf import addons
from pyscf.scf import hf


def populations(atomspec, minimal_basis, high_basis):
    mollow = pyscf.gto.M(atom=atomspec, basis=minimal_basis)
    mflow = pyscf.scf.RHF(mollow)
    mflow.kernel()

    molhigh = pyscf.gto.M(atom=atomspec, basis=high_basis)
    mfhigh = pyscf.scf.RHF(molhigh)
    mfhigh.kernel()

    def getpop(mol, mf):
        pm = pyscf.lo.PM(mol, mf.mo_coeff[:, mf.mo_occ > 0], mf)
        pm.pop_method = "mulliken"
        loc_orb = pm.kernel()
        return loc_orb

    def getpop_on_low(mollow, mflow, molhigh, mfhigh):
        s = hf.get_ovlp(mollow)
        mo = addons.project_mo_nr2nr(
            molhigh, mfhigh.mo_coeff[:, mfhigh.mo_occ > 0], mollow
        )
        norm = np.einsum("pi,pi->i", mo.conj(), s.dot(mo))
        mo /= np.sqrt(norm)
        pm = pyscf.lo.PM(mollow, mo, mflow)
        pm.pop_method = "mulliken"
        loc_orb = pm.kernel()
        return loc_orb

    return getpop(mollow, mflow).T  # , getpop_on_low(mollow, mflow, molhigh, mfhigh).T


# %%
from pyscf.gto.basis import parse_gaussian

CBSB3 = {
    "H": parse_gaussian.parse(
        """ S   3 1.00       0.000000000000
      0.3386500000D+02  0.2549381454D-01
      0.5094790000D+01  0.1903731086D+00
      0.1158790000D+01  0.8521614860D+00
 S   1 1.00       0.000000000000
      0.3258400000D+00  0.1000000000D+01
 S   1 1.00       0.000000000000
      0.1027410000D+00  0.1000000000D+01
 S   1 1.00       0.000000000000
      0.3600000000D-01  0.1000000000D+01
 P   1 1.00       0.000000000000
      0.1500000000D+01  0.1000000000D+01
 P   1 1.00       0.000000000000
      0.3750000000D+00  0.1000000000D+01"""
    ),
    "C": parse_gaussian.parse(
        """C 0
 S   6 1.00       0.000000000000
      0.4563240000D+04  0.1966650249D-02
      0.6820240000D+03  0.1523060193D-01
      0.1549730000D+03  0.7612690966D-01
      0.4445530000D+02  0.2608010331D+00
      0.1302900000D+02  0.6164620782D+00
      0.1827730000D+01  0.2210060280D+00
 SP   3 1.00       0.000000000000
      0.2096420000D+02  0.1146600807D+00  0.4024869267D-01
      0.4803310000D+01  0.9199996477D+00  0.2375939567D+00
      0.1459330000D+01 -0.3030682134D-02  0.8158538515D+00
 SP   1 1.00       0.000000000000
      0.4834560000D+00  0.1000000000D+01  0.1000000000D+01
 SP   1 1.00       0.000000000000
      0.1455850000D+00  0.1000000000D+01  0.1000000000D+01
 SP   1 1.00       0.000000000000
      0.4380000000D-01  0.1000000000D+01  0.1000000000D+01
 D   1 1.00       0.000000000000
      0.1252000000D+01  0.1000000000D+01
 D   1 1.00       0.000000000000
      0.3130000000D+00  0.1000000000D+01
 F   1 1.00       0.000000000000
      0.8000000000D+00  0.1000000000D+01"""
    ),
    "N": parse_gaussian.parse(
        """N 0
 S   6 1.00       0.000000000000
      0.6293480000D+04  0.1969788147D-02
      0.9490440000D+03  0.1496128592D-01
      0.2187760000D+03  0.7350053084D-01
      0.6369160000D+02  0.2489367658D+00
      0.1882820000D+02  0.6024594331D+00
      0.2720230000D+01  0.2562017589D+00
 SP   3 1.00       0.000000000000
      0.3063310000D+02  0.1119060795D+00  0.3831191864D-01
      0.7026140000D+01  0.9216666549D+00  0.2374031155D+00
      0.2112050000D+01 -0.2569191826D-02  0.8175923978D+00
 SP   1 1.00       0.000000000000
      0.6840090000D+00  0.1000000000D+01  0.1000000000D+01
 SP   1 1.00       0.000000000000
      0.2008780000D+00  0.1000000000D+01  0.1000000000D+01
 SP   1 1.00       0.000000000000
      0.6390000000D-01  0.1000000000D+01  0.1000000000D+01
 D   1 1.00       0.000000000000
      0.1826000000D+01  0.1000000000D+01
 D   1 1.00       0.000000000000
      0.4565000000D+00  0.1000000000D+01
 F   1 1.00       0.000000000000
      0.1000000000D+01  0.1000000000D+01"""
    ),
    "O": parse_gaussian.parse(
        """O 0
 S   6 1.00       0.000000000000
      0.8588500000D+04  0.1895150083D-02
      0.1297230000D+04  0.1438590063D-01
      0.2992960000D+03  0.7073200310D-01
      0.8737710000D+02  0.2400010105D+00
      0.2567890000D+02  0.5947970261D+00
      0.3740040000D+01  0.2808020123D+00
 SP   3 1.00       0.000000000000
      0.4211750000D+02  0.1138890124D+00  0.3651139738D-01
      0.9628370000D+01  0.9208111006D+00  0.2371529830D+00
      0.2853320000D+01 -0.3274470358D-02  0.8197019412D+00
 SP   1 1.00       0.000000000000
      0.9056610000D+00  0.1000000000D+01  0.1000000000D+01
 SP   1 1.00       0.000000000000
      0.2556110000D+00  0.1000000000D+01  0.1000000000D+01
 SP   1 1.00       0.000000000000
      0.8450000000D-01  0.1000000000D+01  0.1000000000D+01
 D   1 1.00       0.000000000000
      0.2584000000D+01  0.1000000000D+01
 D   1 1.00       0.000000000000
      0.6460000000D+00  0.1000000000D+01
 F   1 1.00       0.000000000000
      0.1400000000D+01  0.1000000000D+01"""
    ),
    "F": parse_gaussian.parse(
        """F 0
 S   6 1.00       0.000000000000
      0.1142710000D+05  0.1800930156D-02
      0.1722350000D+04  0.1374190119D-01
      0.3957460000D+03  0.6813340591D-01
      0.1151390000D+03  0.2333250202D+00
      0.3360260000D+02  0.5890860511D+00
      0.4919010000D+01  0.2995050260D+00
 SP   3 1.00       0.000000000000
      0.5544410000D+02  0.1145360155D+00  0.3546088738D-01
      0.1263230000D+02  0.9205121249D+00  0.2374509155D+00
      0.3717560000D+01 -0.3378040458D-02  0.8204577080D+00
 SP   1 1.00       0.000000000000
      0.1165450000D+01  0.1000000000D+01  0.1000000000D+01
 SP   1 1.00       0.000000000000
      0.3218920000D+00  0.1000000000D+01  0.1000000000D+01
 SP   1 1.00       0.000000000000
      0.1076000000D+00  0.1000000000D+01  0.1000000000D+01
 D   1 1.00       0.000000000000
      0.3500000000D+01  0.1000000000D+01
 D   1 1.00       0.000000000000
      0.8750000000D+00  0.1000000000D+01
 F   1 1.00       0.000000000000
      0.1850000000D+01  0.1000000000D+01
 """
    ),
}
# %%
