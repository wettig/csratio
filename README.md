### Approximation formula for complex spacing ratios in the Ginibre ensemble

This code accompanies the paper [arXiv:2201.05104](https://arxiv.org/abs/2201.05104).
The directory structure is as follows:

* csratio/analytical: Analytical results from computer algebra code (sympy).
* csratio/numerical: Numerical integration as an alternative. The results are identical to the analytical results within the precision of the numerical integration.
* csratio/rmt: Moments computed from random matrix simulations. They have statistical erros.
* csratio/timing: Timing routine.
* tests: Test scripts for verification.

To install, simply clone this repository.
The code has been tested with python 3.10, sympy 1.9, numpy 1.21.5 and scipy 1.7.3.

To compute analytical results, start a python interpreter and do something like

```
from csratio.analytical.csratio_analytical import csratio_analytical
csr = csratio_analytical(N=6)
```

The coefficients and moments are then available as attributes of csr, such as csr.an, csr.bn, csr.cmn and csr.moments. Similarly for the numerical and RMT code, see code and test scripts.

No serious attempts at performance optimization have been made since the run time is acceptable for the interesting range of matrix dimensions.
