import numpy as np
from scipy.integrate import nquad
from scipy.special import gammaincc

# from csratio.timing.timing import timeit


class csratio_numerical:
    # This code could be optimized for performance, but it is already
    # fast enough for values of N that give results indistinguishable
    # from the N=\infty limit. So we don't bother.
    def __init__(self, N, moments=[], nquad_options={"epsabs": 1e-8, "epsrel": 1e-8}):
        # moments is list of desired moments or "all"
        assert N > 2, "N must be at least 3"
        self.N = N  # this is the real N, not N-3
        self.opts = nquad_options
        self.moments = self.compute_moments(moments)

    def compute_M(self, R, r, theta):
        """
        Compute the 3x3 matrix M.
        """
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        r2 = r * r
        sqrtR = np.sqrt(R)
        M = np.empty((3, 3), dtype=complex)
        M[0][0] = R * R * r2
        M[0][1] = -R * sqrtR * (r2 + x + 1j * y)
        M[0][2] = R * (x + 1j * y)
        M[1][0] = M[0][1].conjugate()
        M[1][1] = R * (r2 + 2 * x + 1)
        M[1][2] = -sqrtR * (1 + x + 1j * y)
        M[2][0] = M[0][2].conjugate()
        M[2][1] = M[1][2].conjugate()
        M[2][2] = 1.0
        return M

    def compute_A(self, R, r, theta):
        """
        Compute the matrix A for N>3.
        """
        N = self.N - 3  # this is N-3
        M = self.compute_M(R, r, theta)
        A = np.zeros((N, N), dtype=complex)
        Gamma = gammaincc(np.arange(2, N + 4), R)
        for i in range(N):
            # The rows grow factorially, therefore we pull out a factor
            # of (i+3)! from row i to avoid numerical instabilities.
            # (This only changes the determinant by an overall factor.)
            # The initegral is then normalized to 2*pi.
            factor1 = [1 / ((i + 2) * (i + 3)), 1 / (i + 3), 1]
            factor2 = [1, 1 / (i + 4), 1 / ((i + 4) * (i + 5))]
            for j in range(min(3, N, N - i)):  # min catches the corner cases
                A[i][i + j] = sum(
                    M[m][m - j] * Gamma[m + i] * factor1[m] for m in range(j, 3)
                )
                if j > 0:
                    A[i + j][i] = A[i][i + j].conjugate() * factor2[j]
        return A

    def integrand_K(self, R, r, theta):
        """
        Compute the integrand of K.
        """
        if self.N == 3:
            detA = 1
        else:
            A = self.compute_A(R, r, theta)
            detA = np.linalg.det(A).real  # convert to real (imaginary part is zero)
        return R ** 4 * np.exp(-R * (1 + r ** 2)) * detA

    def integrand_P(self, R, r, theta):
        """
        Compute P(z), which is part of the integrand.
        """
        return (
            r ** 3
            * (r ** 2 - 2 * r * np.cos(theta) + 1)
            * self.integrand_K(R, r, theta)
        )

    def compute_moments(self, which_moments):
        """
        Compute the moments by doing a 3d integral.
        """
        # The 2d density is normalized to 2*pi.
        # We optionally compute <1> to check the normalization.
        moments = {}
        functions = {
            "1": lambda R, r, theta: self.integrand_P(R, r, theta),
            "r": lambda R, r, theta: r * self.integrand_P(R, r, theta),
            "r^2": lambda R, r, theta: r ** 2 * self.integrand_P(R, r, theta),
            "cos(theta)": lambda R, r, theta: np.cos(theta)
            * self.integrand_P(R, r, theta),
            "cos^2(theta)": lambda R, r, theta: np.cos(theta) ** 2
            * self.integrand_P(R, r, theta),
            "r*cos(theta)": lambda R, r, theta: r
            * np.cos(theta)
            * self.integrand_P(R, r, theta),
        }
        if "1" not in which_moments:
            moments["1"] = 2 * np.pi
            functions.pop("1")
        if which_moments == "all":
            which_moments = functions.keys()
        for key in which_moments:
            # integrand is symmetric w.r.t. theta <-> 2*pi-theta
            moments[key], error = 2 * np.array(
                nquad(
                    functions[key], [[0, np.infty], [0, 1], [0, np.pi]], opts=self.opts
                )
            )
            if key == "1":
                print(key, moments[key], error / moments[key])
            else:
                # print(key, moments[key] / moments["1"], error / moments[key])
                moments[key] /= moments["1"]
        return moments

    def Pr(self, r):
        """
        Compute P(r) by doing a 2d integral.
        """
        return (
            2
            * nquad(
                lambda R, theta: self.integrand_P(R, r, theta),
                [[0, np.infty], [0, np.pi]],
                opts=self.opts,
            )[0]
            / self.moments["1"]
        )

    def Ptheta(self, theta):
        """
        Compute P(theta) by doing a 2d integral.
        """
        return (
            nquad(
                lambda R, r: self.integrand_P(R, r, theta),
                [[0, np.infty], [0, 1]],
                opts=self.opts,
            )[0]
            / self.moments["1"]
        )
