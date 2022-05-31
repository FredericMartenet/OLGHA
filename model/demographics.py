"""
Main routines for demographic objects.
"""

import numpy as np


class Population:
    """
    Population class with parameters and population objects.

    Parameters
    ----------
    Tw : int
        Age at which economic life starts
    Tr : int
        Retirement age
    T : int
        Age of certain death
    n : float
        Population growth rate
    phi : int or np.array
        Survival probabilities by age

    """
    def __init__(self, Tw=20, Tr=65, T=100, n=0.02, phi=None):
        self.Tw = Tw
        self.Tr = Tr
        self.T = T
        self.n = n

        # age vector
        self.jvec = np.arange(0, T + 1)

        # boolean for retirement bya ge
        self.iret = 1 * (self.jvec >= self.Tr)

        # Survival probabilities rates: phi(j) is probability of surviving from j-1 to j
        if not phi:
            self.phi = np.array([
                1., 1., 1., 1., 0.99958024, 0.99984742, 0.9998256, 0.99985417, 0.99986156,
                0.99987262, 0.99988967, 0.99990342, 0.9999109, 0.9998889, 0.99983457, 0.99974538,
                0.99963829, 0.99953787, 0.99945125, 0.99938193, 0.99933839, 0.99930488, 0.99926748,
                0.99924242, 0.99924514, 0.99926275, 0.99929149, 0.99931654, 0.99933161, 0.99932689,
                0.9993071, 0.99927363, 0.99923712, 0.99919915, 0.99914045, 0.99907633, 0.99900216,
                0.99891874, 0.99882671, 0.99873562, 0.99863822, 0.99853303, 0.99841911, 0.99831145,
                0.99820416, 0.99809438, 0.99798009, 0.99785222, 0.99772708, 0.99761754, 0.99751595,
                0.99740357, 0.99726706, 0.99709373, 0.99687835, 0.99661299, 0.99630414, 0.99595392,
                0.9955756, 0.99518221, 0.99476333, 0.99431237, 0.99381049, 0.99320721, 0.99249086,
                0.99166128, 0.99073179, 0.98972359, 0.98867498, 0.98761743, 0.98654746, 0.98536994,
                0.98405218, 0.98266914, 0.98127292, 0.97985243, 0.97817818, 0.97613982, 0.97395495,
                0.97179702, 0.96951054, 0.96698265, 0.96399526, 0.95999319, 0.95463326, 0.94802055,
                0.94049785, 0.93241941, 0.92402846, 0.91541236, 0.90647692, 0.89714784, 0.88752895,
                0.87705071, 0.86563134, 0.85318403, 0.84041459, 0.8279652, 0.81597948, 0.80466526,
                0.79439827
            ])

        # Cumulative survival probabilities
        self.Phi = np.cumprod(np.append(1, self.phi[:-1]))

        # Stationary population distribution
        self.pi = self.pop_stationary(n)

        # Labor supply profile by age
        h = np.zeros((T + 1))
        h[Tw:Tr] = -3 + 50 * self.jvec[Tw:Tr] - 0.5 * self.jvec[Tw:Tr] ** 2
        self.h = h / np.sum(self.pi * h)

    def pop_stationary(self, n):
        """
        Computes the stationary population distribution for a given growth rate and survival probabilities.
        """
        n_cum = (1 + n) ** np.arange(0, self.T + 1)
        pi0 = 1 / np.sum(self.Phi / n_cum)
        pi = self.Phi / n_cum * pi0

        return pi

    def trans_pop(self, pi0, n, Ttrans=200):
        """
        Compute the population transition dynamics for fixed n and phi.
        """
        pitrans, ntrans = np.zeros((Ttrans, pi0.shape[0])), np.zeros((Ttrans, pi0.shape[0]))
        pitrans[0, :], ntrans[0, :] = pi0, pi0 / pi0[0]
        for t in np.arange(1, Ttrans):
            ntrans[t, 0] = (1 + n) ** t
            ntrans[t, 1:] = self.phi[1:] * ntrans[t - 1, :-1]
            pitrans[t, :] = ntrans[t, :] / np.sum(ntrans[t, :])

        return pitrans

    def mass_ret_work(self, pi):
        """
        Returns the mass of retirees and workers.
        """
        if pi.ndim == 1:
            retirees = np.sum(pi * self.iret)
            workers = np.sum(pi * self.h * (1 - self.iret))
        else:
            retirees = np.sum(pi * self.iret[np.newaxis, ], axis=1)
            workers = np.sum(pi * self.h[np.newaxis, ] * (1 - self.iret[np.newaxis, ]), axis=1)

        return retirees, workers
