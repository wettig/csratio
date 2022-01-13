#!/usr/bin/env python3

"""
This script computes analytical results for the coefficients c_mn, a_n and b_n
as well as for the moments. It tests them against reference results for N=3 and N=6.
Input: N (the matrix dimension, set by hand below)
Output available in python interpreter: csr.an, csr.bn, csr.cmn, csr.moments
"""

import sys
import pickle

sys.path.append("..")
from csratio.analytical.csratio_analytical import csratio_analytical

# test against reference results

reference_file = "analytical.pickle"  # reference results for N=3 and 6
with open(reference_file, "rb") as fp:
    pickle.load(fp)  # discard comment in reference file
    for N in [3, 6]:
        print(f"\nComputing for N={N} ...")
        csr = csratio_analytical(N)
        print(f"\nTesting for N={N} ... ", end="")
        y = pickle.load(fp)
        for a in vars(csr).keys():
            assert getattr(csr, a) == getattr(
                y, a
            ), f"Test failed for N={N}: attribute {a}."
        print("done")

# print the numerical values of the coefficients and moments for N=6

print(f"\nResults for N={N}:")
print("\nCoefficients of P(r):\nn\ta_n")
for idx, val in enumerate(csr.an):
    print(f"{idx}\t{float(val)}")
print("\nCoefficients of P(theta):\nn\tb_n")
for idx, val in enumerate(csr.bn):
    print(f"{idx}\t{float(val)}")
print("\nMoments:")
for key, val in csr.moments.items():
    print(f"<{key}> = {float(val)}")
