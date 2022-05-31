import numpy as np
from numba import njit, guvectorize


def set_parameters(r=0.05, beta=0.94, sigma=2.0, sigma_eps=0.5, N_eps=7, N_a=200, amax=1000):
    """
    Defines the main parameters of the model.
    """
    params = dict()

    # INTEREST RATE
    params['r'] = r

    # PREFERENCES
    params['sigma'] = sigma  # Inverse elasticity of substitution
    params['beta'] = beta  # Subjective discount factor

    # IDIOSYNCRATIC PRODUCTIVITY
    params['rho_eps'] = 0.5
    params['sigma_eps'] = sigma_eps
    params['N_eps'] = N_eps  # Number of idiosyncatic states (epsilon)
    params['y_eps'], params['pi_eps'], params['Pi_eps'] = markov_incomes(
        rho=params['rho_eps'], sigma_y=params['sigma_eps'], N=params['N_eps'])

    # ASSET GRID
    params['N_a'] = N_a  # Number of grid points
    params['amax'] = amax  # Maximum value of the grid
    params['a'] = agrid(amin=0.0, amax=params['amax'], N=params['N_a'])  # Borrowing limit and asset grid

    return params


@guvectorize(['void(float64[:], float64[:], float64[:], float64[:])'], '(n),(nq),(n)->(nq)')
def interpolate_y(x, xq, y, yq):
    """
    Efficient linear interpolation exploiting monotonicity.

    Complexity O(n+nq), so most efficient when x and xq have comparable number of points.
    Extrapolates linearly when xq out of domain of x.

    Parameters
    ----------
    x: array
        ascending data points
    xq: array
        ascending query points
    y: array
        data points
    yq: array
        empty to be filled with interpolated points
    """
    nxq, nx = xq.shape[0], x.shape[0]

    xi = 0
    x_low = x[0]
    x_high = x[1]
    for xqi_cur in range(nxq):
        xq_cur = xq[xqi_cur]
        while xi < nx - 2:
            if x_high >= xq_cur:
                break
            xi += 1
            x_low = x_high
            x_high = x[xi + 1]

        xqpi_cur = (x_high - xq_cur) / (x_high - x_low)
        yq[xqi_cur] = xqpi_cur * y[xi] + (1 - xqpi_cur) * y[xi + 1]


@guvectorize(['void(float64[:], float64[:], uint32[:], float64[:])'], '(n),(nq)->(nq),(nq)')
def interpolate_coord(x, xq, xqi, xqpi):
    """
    Efficient linear interpolation exploiting monotonicity. xq = xqpi * x[xqi] + (1-xqpi) * x[xqi+1]

    Parameters
    ----------
    x: array
        ascending data points
    xq: array
        ascending query points
    xq: array
        empty to be filled with indices of lower bracketing gridpoints
    xqpi: array
        empty to be filled with weights on lower bracketing gridpoints

    """
    nxq, nx = xq.shape[0], x.shape[0]

    xi = 0
    x_low = x[0]
    x_high = x[1]
    for xqi_cur in range(nxq):
        xq_cur = xq[xqi_cur]
        while xi < nx - 2:
            if x_high >= xq_cur:
                break
            xi += 1
            x_low = x_high
            x_high = x[xi + 1]

        xqpi[xqi_cur] = (x_high - xq_cur) / (x_high - x_low)
        xqi[xqi_cur] = xi


@njit(fastmath=True)
def forward_step(D, Pi_T, a_pol_i, a_pol_pi):
    """
    Single forward step to update distribution using an arbitrary asset policy.

    Efficient implementation of D_t = Lam_{t-1}' @ D_{t-1} using sparsity of Lam_{t-1}.

    Parameters
    ----------
    D: np.ndarray
        Beginning-of-period distribution over s_t, a_(t-1)
    Pi_T: np.ndarray
        Transpose Markov matrix that maps s_t to s_(t+1)
    a_pol_i: np.ndarray
        Left gridpoint of asset policy
    a_pol_pi: np.ndarray
        Weight on left gridpoint of asset policy

    Returns
    ----------
    Dnew : np.ndarray
        Beginning-of-next-period dist s_(t+1), a_t

    """
    # first create Dnew from updating asset state
    Dnew = np.zeros((D.shape[0], D.shape[1]))
    for s in range(D.shape[0]):
        for i in range(D.shape[1]):
            apol = a_pol_i[s, i]
            api = a_pol_pi[s, i]
            d = D[s, i]
            Dnew[s, apol] += d * api
            Dnew[s, apol + 1] += d * (1 - api)

    # then use transpose Markov matrix to update income state
    Dnew = Pi_T @ Dnew

    return Dnew


def markov_rouwenhorst(rho, sigma, N=7):
    """
    Rouwenhorst method to discretize an AR(1) process
    """
    # parametrize Rouwenhorst for n=2
    p = (1 + rho) / 2
    Pi = np.array([[p, 1 - p], [1 - p, p]])

    # implement recursion to build from n=3 to n=N
    for n in range(3, N + 1):
        P1, P2, P3, P4 = (np.zeros((n, n)) for _ in range(4))
        P1[:-1, :-1] = p * Pi
        P2[:-1, 1:] = (1 - p) * Pi
        P3[1:, :-1] = (1 - p) * Pi
        P4[1:, 1:] = p * Pi
        Pi = P1 + P2 + P3 + P4
        Pi[1:-1] /= 2

    # invariant distribution and scaling
    pi = stationary(Pi)
    s = np.linspace(-1, 1, N)
    s *= (sigma / np.sqrt(var(s, pi)))

    return s, pi, Pi


def markov_incomes(rho, sigma_y, N=11):
    """
    Simple helper method that assumes AR(1) process in logs for incomes and scales aggregate income
    to 1, also that takes in sdy as the *cross-sectional* sd of log incomes
    """
    sigma = sigma_y * np.sqrt(1 - rho ** 2)
    s, pi, Pi = markov_rouwenhorst(rho, sigma, N)
    y = np.exp(s) / np.sum(pi * np.exp(s))
    return y, pi, Pi


def mean(x, pr):
    pr = pr / np.sum(pr)
    return np.sum(pr * x)


def cov(x, y, pr):
    pr = pr / np.sum(pr)
    return np.sum(pr * (x - mean(x, pr)) * (y - mean(y, pr)))


def var(x, pr):
    pr = pr / np.sum(pr)
    return cov(x, x, pr)


def ineq(ss, pop):
    """
    Inequality statistics.
    """
    T, Neps, Na = ss['a'].shape
    a_flat = ss['a'].reshape(T, 1, Neps * Na).squeeze()  # reshape multi-dimensional policies
    Dst_flat = ss['D'].reshape(T, 1, Neps * Na).squeeze()  # flatten out the joint distribution

    # Lorenz curves
    a = np.einsum('js,js->s', pop.pi[:, np.newaxis], a_flat)
    p = np.einsum('js,js->s', pop.pi[:, np.newaxis], Dst_flat)
    p = p / np.sum(p)  # Make sure sums to one
    a_sorted = np.sort(a)  # Sort vectors from lowest to highest
    a_sorted_i = np.argsort(a)
    p_a_sorted = p[a_sorted_i]  # Recover associated probabilities
    lorenz_a_pctl, lorenz_a = lorenz(a_sorted, p_a_sorted)  # Get Lorenz curves

    return lorenz_a_pctl, lorenz_a


def lorenz(x, pr):
    """
    Returns Lorenz curve.
    """
    # first do percentiles of the total population
    pctl = np.concatenate(([0], pr.cumsum() - pr / 2, [1]))
    # now do percentiles of total wealth (returns only zeros if sum(pr*x) = 0)
    wealthshare = (x * pr / np.sum(x * pr) if np.sum(x * pr) != 0 else np.zeros_like(x))
    wealthpctl = np.concatenate(([0], wealthshare.cumsum() - wealthshare / 2, [1]))
    return pctl, wealthpctl


def find_nearest(array, value):
    array = np.asarray(array)
    return (np.abs(array - value)).argmin()


def agrid(amax, N, amin=0):
    """
    Grid a+pivot evenly log-spaced between amin+pivot and amax+pivot
    """
    pivot = np.abs(amin) + 0.25
    a = np.geomspace(amin + pivot, amax + pivot, N) - pivot
    a[0] = amin  # make sure *exactly* equal to amin

    return a


@njit
def within_tolerance(x1, x2, tol):
    """
    Efficiently test max(abs(x1-x2)) <= tol for arrays of same dimensions x1, x2.
    """
    y1 = x1.ravel()
    y2 = x2.ravel()
    for i in range(y1.shape[0]):
        if np.abs(y1[i] - y2[i]) > tol:
            return False
    return True


def stationary(Pi, pi_seed=None, tol=1E-11, maxit=10_000):
    """
    Find invariant distribution of a Markov chain by iteration.
    """
    if pi_seed is None:
        pi = np.ones(Pi.shape[0]) / Pi.shape[0]
    else:
        pi = pi_seed

    for it in range(maxit):
        pi_new = pi @ Pi
        if within_tolerance(pi_new, pi, tol):
            break
        pi = pi_new
    else:
        raise ValueError(f'No convergence after {maxit} forward iterations!')
    pi = pi_new

    return pi


def make_path(x, T):
    """
    Takes in x as either a number, a vector or a matrix, turning it into a path.
    """
    x = np.asarray(x)
    if x.ndim <= 1:
        return np.tile(x, (T, 1))

    elif x.ndim == 2:
        return np.tile(x, (T, 1, 1))


def make_full_path(x, T):
    """
    Takes a path x (vector/matrix), and repeats the last line until x has T lines.
    """
    if x.ndim == 1:
        raise ValueError('x must be a column vector')

    if T < x.shape[0]:
        raise ValueError('T must be greater than the number of lines in x')

    return np.vstack((x, make_path(x[-1], T - x.shape[0])))


def pack_jacobians(jacdict, inputs, outputs, T):
    """
    If we have T*T jacobians from nI inputs to nO outputs in jacdict, combine into (nO*T)*(nI*T) jacobian matrix.
    """
    nI, nO = len(inputs), len(outputs)

    outjac = np.empty((nO * T, nI * T))
    for iO in range(nO):
        subdict = jacdict.get(outputs[iO], {})
        for iI in range(nI):
            outjac[(T * iO):(T * (iO + 1)), (T * iI):(T * (iI + 1))] = make_matrix(
                subdict.get(inputs[iI], np.zeros((T, T))), T)
    return outjac


def unpack_jacobians(bigjac, inputs, outputs, T):
    """
    If we have an (nO*T)*(nI*T) jacobian and provide names of nO outputs and nI inputs, output nested dictionary
    """
    nI, nO = len(inputs), len(outputs)

    jacdict = {}
    for iO in range(nO):
        jacdict[outputs[iO]] = {}
        for iI in range(nI):
            jacdict[outputs[iO]][inputs[iI]] = bigjac[(T * iO):(T * (iO + 1)), (T * iI):(T * (iI + 1))]
    return jacdict


def make_matrix(A, T):
    """
    If A is not an outright ndarray, e.g. it is SimpleSparse, call its .matrix(T) method
    to convert it to T*T array.
    """
    if not isinstance(A, np.ndarray):
        return A.matrix(T)
    else:
        return A


def pack_vectors(vs, names, T):
    v = np.zeros(len(names)*T)
    for i, name in enumerate(names):
        if name in vs:
            v[i*T:(i+1)*T] = vs[name]
    return v


def unpack_vectors(v, names, T):
    vs = {}
    for i, name in enumerate(names):
        vs[name] = v[i*T:(i+1)*T]
    return vs
