import numpy as np


def pop_stationary(n, phi, T=100):
    """Computes the stationary population distribution for a given growth rate and vector of survival probabilities."""
    phi_lag = np.append(1, phi[:-1])
    Phi = np.cumprod(phi_lag)
    n_cum = (1 + n) ** np.arange(0, T + 1)
    pi0 = 1 / np.sum(Phi / n_cum)
    pi = Phi / n_cum * pi0

    return pi


def trans_pop(pi0, n, phi, Ttrans=200):
    """ Compute the population transition dynamics for fixed n and phi."""
    pitrans = np.zeros((Ttrans, pi0.shape[0]))
    pitrans[0,:] = pi0
    ntrans = np.zeros((Ttrans, pi0.shape[0]))
    ntrans[0,:] = pi0 / pi0[0]
    for t in np.arange(1, Ttrans):
        ntrans[t,0] = (1 + n) ** t
        ntrans[t,1:] = phi[1:] * ntrans[t-1,:-1]
        pitrans[t,:] = ntrans[t,:] / np.sum(ntrans[t,:])

    return pitrans


