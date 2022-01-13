#!/usr/bin/env python3

"""
This script computes numerical results for P(r), P(theta) and the moments
and compares them to reference results for N=3 and N=6.
Input: Nvalues (the matrix dimensions, set by hand below)
Output available in python interpreter: csr.Pr(r), csr.Ptheta(theta), csr.moments
(the latter after calling csr.compute_moments(which_moments="all"))
"""

import numpy as np
import sys

sys.path.append("..")
from csratio.numerical.csratio_numerical import csratio_numerical

Nvalues = [3, 6]
eps = 1e-8
nquad_options = {"epsabs": eps, "epsrel": eps}

# the reference files contain results for N=3 and N=6
Pr = np.loadtxt("Pr.dat").T
Ptheta = np.loadtxt("Ptheta.dat").T
moments = np.loadtxt("moments_numerical.dat")
moments[:, 0] = 2 * np.pi  # replace N by normalization, to be checked below

for N in Nvalues:
    print(f"\nComputing moments for N={N} ... ", end="")
    csr = csratio_numerical(N, nquad_options=nquad_options)
    mom = csr.compute_moments(which_moments="all")
    print("done")
    print(f"Testing for N={N} ... ", end="")
    for idx, val in enumerate(mom.items()):
        ref = moments[N // 3 - 1, idx]
        assert abs(val[1] - ref) < eps, f"Test failed for N={N}: moment {val[0]}."
    for Pstring in ["Pr", "Ptheta"]:
        P = locals()[Pstring]
        for idx, x in enumerate(P[0]):
            val = getattr(csr, Pstring)(x)
            ref = P[N // 3, idx]
            assert abs(val - ref) < eps, f"Test failed for N={N}: {Pstring}({x})."
    print("done")
