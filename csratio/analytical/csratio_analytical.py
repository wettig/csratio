import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from csratio.timing.timing import timeit

w = sp.symbols("w")
a, b, cost, m, s, t, x, y = sp.symbols("a b cost m s t x y", real=True)
r, R, u = sp.symbols("r, R u", positive=True)


class csratio_analytical:
    def __init__(self, N):
        self.N = N  # this is the real N, not N-3
        assert N > 2, "N must be at least 3"
        self.Mmn = timeit(self.compute_Mmn)
        self.ck = timeit(self.compute_ck)
        self.In = timeit(self.compute_In)
        self.cmn = timeit(self.compute_cmn)
        self.an = timeit(self.compute_an)
        self.bn = timeit(self.compute_bn)
        self.moments = timeit(self.compute_moments)

    def compute_Mmn(self):
        """
        Compute the coefficient functions M_mn.
        """
        lhs = (
            (a ** 2 + b ** 2)
            * ((a - s) ** 2 + (b - t) ** 2)
            * ((a - s * x + t * y) ** 2 + (b - t * x - s * y) ** 2)
        )
        lhs = lhs.subs(
            {a: (w + w.conjugate()) / 2, b: (w - w.conjugate()) / 2 / sp.I}
        ).expand()
        Mmn = sp.Matrix(
            3,
            3,
            lambda m, n: lhs.coeff(w, m + 1).coeff(w.conjugate(), n + 1)
            # .factor()
            # don't use factor() here since it gives an incorrect result for M_12
            # .factor(domain="ZZ") gives a correct result but does not factorize fully
            .subs({s: (w + sp.conjugate(w)) / 2, t: (w - sp.conjugate(w)) / 2 / sp.I})
            .factor()
            .subs(sp.conjugate(w), R / w),
        )
        return Mmn

    def compute_A_complex(self):
        """
        Compute the matrix A without the exponential factor using complex w.
        (Only used for checking.)
        """
        N = self.N - 3  # this is N-3
        A = sp.Matrix(N, N, sp.zeros(N))
        for i in range(N):
            for j in range(min(3, N, N - i)):  # min catches the corner cases
                A[i, i + j] = (
                    sp.Sum(
                        self.Mmn[m, m - j] * sp.uppergamma(m + i + 2, R),
                        (m, j, 2),
                    ).doit()
                    * sp.exp(R)
                ).expand()
                if j > 0:
                    A[i + j, i] = A[i, i + j].conjugate().subs(w.conjugate(), R / w)
        return A

    def compute_A_real(self):
        """
        Compute the matrix A without the exponential factor using real w=sqrt(R).
        """
        M = self.Mmn.subs(w, sp.sqrt(R))
        N = self.N - 3  # this is N-3
        A = sp.Matrix(N, N, sp.zeros(N))
        for i in range(N):
            for j in range(min(3, N, N - i)):  # min catches the corner cases
                A[i, i + j] = (
                    sp.Sum(
                        M[m, m - j] * sp.uppergamma(m + i + 2, R),
                        (m, j, 2),
                    ).doit()
                    * sp.exp(R)
                ).expand()
                if j > 0:
                    A[i + j, i] = A[i, i + j].conjugate()
        return A

    def compute_ck(self):
        """
        Compute the coefficients c_k(x,y) as ck(r,theta).
        """
        if self.N == 3:
            return None
        # first compute the matrix A without the exponential factor
        A = self.compute_A_real()
        # the default method for det() takes unacceptably long
        detA = A.det(method="berkowitz")
        # extract the coefficients c_k
        detA = sp.Poly(detA, R)  # convert to polynomial (performance bottleneck!)
        kmax = sp.LM(detA).args[1]  # largest power of R
        assert kmax == (self.N - 3) * (self.N + 2) // 2, "max power of k wrong"
        ck = [
            detA.nth(k).subs(y ** 2, r ** 2 - x ** 2).expand().subs(x, r * cost)
            for k in range(kmax + 1)
        ]
        return ck

    def compute_In(self):
        """
        Compute the integrals over r from two recursion relations.
        Note that the output is actually u^p I_{n,p}(u) for p=pmax.
        """
        # The integrals are I(n,p) = \int_0^1 dr r^n/(u+r^2)^p.
        # Using the recursion relations is a lot faster than
        # computing the integrals explicitly.
        N = self.N  # N is the real N, not N-3
        nmax = N * (N - 1) + 1  # two powers more than necessary (for <r> and <r^2>)
        pmax = N * (N - 1) // 2 + 2
        # put the I(n,p) in a matrix
        Inp = sp.Matrix(nmax + 1, pmax + 1, sp.zeros(nmax + 1, pmax + 1))
        # initialize p=0 (all n) and p=1 (n=0,1 only)
        for n in range(nmax + 1):
            Inp[n, 0] = sp.S(1) / (n + 1)
        Inp[0, 1] = sp.atan(1 / sp.sqrt(u)) / sp.sqrt(u)
        Inp[1, 1] = sp.log(u + 1) / 2 - sp.log(u) / 2
        # compute I(0,p) and I(1,p) from one recursion relation
        for n in range(2):
            for p in range(1, pmax):
                Inp[n, p + 1] = -sp.diff(Inp[n, p], u) / p
            for p in range(pmax + 1):
                Inp[n, p] = Inp[n, p].subs(u, N - 2)
        # now compute I(n,p) from the other recursion relation
        for n in range(nmax - 1):
            for p in range(1, pmax + 1):
                Inp[n + 2, p] = Inp[n, p - 1] - (N - 2) * Inp[n, p]
        return (N - 2) ** pmax * Inp[:, pmax]

    def ave_cos(self, n):
        return 2 * sp.pi * sp.factorial2(n - 1) / sp.factorial2(n)

    def normalization(self, cmn):
        """
        Compute the integral of the unnormalized P(r, theta) over r and theta.
        """
        N = self.N  # this is the real N, not N-3
        mmax = (N + 2) * (N - 3)
        nmax = N - 3
        norm = sum(
            sum(cmn[m, n] * self.ave_cos(n) for n in range(0, nmax + 1, 2))
            * (self.In[m + 5] + self.In[m + 3])
            for m in range(0, mmax + 1, 2)
        ) - 2 * sum(
            sum(cmn[m, n] * self.ave_cos(n + 1) for n in range(1, nmax + 1, 2))
            * self.In[m + 4]
            for m in range(1, mmax + 1, 2)
        )
        return norm

    def compute_cmn(self):
        """
        Compute the coefficients c_mn in P(r, theta), including normalization.
        """
        N = self.N - 3  # this is N-3
        if N == 0:  # this includes the normalization
            return sp.Matrix(1, 1, [12 / sp.pi])
        kmax = len(self.ck) - 1
        sumk = sum(
            self.ck[k] * sp.factorial(k + 4) * (N + 1 + r ** 2) ** (kmax - k)
            for k in range(kmax + 1)
        )
        sumk = sp.Poly(sumk.expand())
        mmax = sp.LM(sumk, r).args[1]
        assert mmax == N * (N + 5), "max power of r wrong"
        nmax = sp.LM(sumk, cost).args[1] if N > 1 else 1
        assert nmax == N, "max power of cos(theta) wrong"
        cmn = sp.Matrix(mmax + 1, nmax + 1, lambda m, n: sumk.nth(m, n)) / (N + 1) ** (
            mmax // 2 + 5
        )
        # compute the integral for normalization
        norm = self.normalization(cmn)
        return cmn / norm

    def compute_an(self):
        """
        Compute the coefficients a_n in P(r).
        """
        N = self.N  # this is the real N, not N-3
        nmax = N * (N - 1) // 2 - 2
        sn = [
            sum(self.cmn[2 * n, l] * self.ave_cos(l) for l in range(0, N - 2, 2))
            for n in range(nmax)
        ]
        tn = [
            sum(
                self.cmn[2 * n + 1, l] * self.ave_cos(l + 1) for l in range(1, N - 2, 2)
            )
            for n in range(nmax - 1)
        ]
        an = (
            [sn[0]]
            + [sn[n] + sn[n - 1] - 2 * tn[n - 1] for n in range(1, nmax)]
            + [sn[nmax - 1]]
        )
        return an

    def compute_bn(self):
        """
        Compute the coefficients b_n in P(theta).
        """
        N = self.N  # this is the real N, not N-3
        mmax = (N + 2) * (N - 3)
        rn = [None] * 6
        for l in range(3, 6):
            rn[l] = [
                sum(self.cmn[m, n] * self.In[m + l] for m in range(mmax + 1))
                for n in range(N - 2)
            ]
        bn = (
            [rn[3][0] + rn[5][0]]
            + [rn[3][n] + rn[5][n] - 2 * rn[4][n - 1] for n in range(1, N - 2)]
            + [-2 * rn[4][N - 3]]
        )
        bn = [x.simplify() for x in bn]
        return bn

    def compute_moments(self):
        """
        Compute the moments.
        """
        N = self.N  # this is the real N, not N-3
        nmax = N * (N - 1) // 2 - 2
        moments = {}
        moments["r"] = sum(self.an[n] * self.In[2 * n + 4] for n in range(nmax + 1))
        moments["r^2"] = sum(self.an[n] * self.In[2 * n + 5] for n in range(nmax + 1))
        moments["cos(theta)"] = sum(
            self.bn[n - 1] * self.ave_cos(n) for n in range(2, N, 2)
        )
        moments["cos^2(theta)"] = sum(
            self.bn[n - 2] * self.ave_cos(n) for n in range(2, N + 1, 2)
        )
        mmax = (N + 2) * (N - 3)
        nmax = N - 3
        moments["r*cos(theta)"] = sum(
            sum(self.cmn[m, n] * self.ave_cos(n + 1) for n in range(1, nmax + 1, 2))
            * (self.In[m + 6] + self.In[m + 4])
            for m in range(1, mmax + 1, 2)
        ) - 2 * sum(
            sum(self.cmn[m, n] * self.ave_cos(n + 2) for n in range(0, nmax + 1, 2))
            * self.In[m + 5]
            for m in range(0, mmax + 1, 2)
        )
        return moments

    def Pr(self, r):
        N = self.N
        nmax = len(self.an) - 1
        return (
            r ** 3
            / (1 + r ** 2 / (N - 2)) ** (nmax + 4)
            * sum(sp.N(self.an[n]) * r ** (2 * n) for n in range(nmax + 1))
        )

    def Ptheta(self, theta):
        N = self.N
        return sum(sp.N(self.bn[n]) * np.cos(theta) ** n for n in range(N - 1))

    def plot_Pr(self, nbin=50):
        r = np.linspace(0, 1, nbin)
        y = self.Pr(r)
        plt.plot(r, y)
        plt.show()

    def plot_Ptheta(self, nbin=50):
        theta = np.linspace(0, 2 * np.pi, nbin)
        y = self.Ptheta(theta)
        plt.plot(theta, y)
        plt.show()
