import numpy as np
from scipy import spatial, stats
from csratio.timing.timing import timeit


class csratio_rmt:
    def __init__(self, dim, stat, r_bins=50, theta_bins=50):
        assert dim > 2, "Matrix dimension must be at least 3."
        self.N = dim
        self.samples = stat // dim  # same number of zk values for all dims
        self.zk = timeit(self.compute_zk)
        self.moments = timeit(self.compute_moments)

    def eig_sample(self):
        N = self.N
        A = np.random.randn(N, N) + 1j * np.random.randn(N, N)
        ev = np.linalg.eigvals(A)
        return ev

    def zk_sample(self):
        ev = self.eig_sample()
        data = np.column_stack((ev.real, ev.imag))
        tree = spatial.KDTree(data)
        nn, idx = tree.query(data, k=3)
        zk = (ev[idx[:, 1]] - ev) / (ev[idx[:, 2]] - ev)
        return zk

    def compute_zk(self):
        zk = [self.zk_sample() for n in range(self.samples)]
        return np.concatenate(zk)

    def compute_moments(self):
        zk = self.zk
        moments = {
            "r": abs(zk),
            "r^2": (abs(zk)) ** 2,
            "cos(theta)": zk.real / abs(zk),
            "cos^2(theta)": (zk.real / abs(zk)) ** 2,
            "r*cos(theta)": zk.real,
        }
        for k, v in moments.items():
            moments[k] = (np.mean(v), stats.sem(v))
            print(k, moments[k])
        return moments
