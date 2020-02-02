import numpy as np
from numba import njit, guvectorize
from demographics import pop_stationary

"""Model parameters"""

def set_parameters(r = 0.05, beta=0.94, sigma=2.0, sigma_eps=0.5, Tw=20, Tr=65, T=100,  N_eps=7, N_a=200, amax=1000, n=0.02):
    params = dict()

    # MODEL AGES
    params['Tw'] = Tw  # age at which start work
    params['Tr'] = Tr  # Retirement
    params['T'] = T  # last year of life for sure
    params['jvec'] = np.arange(0, params['T'] + 1)  # age vector

    # DEMOGRAPHICS
    params['phi'] = np.array(  # phi(j) is probability of surviving from j-1 to j
        [1., 1., 1., 1., 0.99958024, 0.99984742, 0.9998256, 0.99985417, 0.99986156, 0.99987262, 0.99988967, 0.99990342,
         0.9999109, 0.9998889, 0.99983457, 0.99974538, 0.99963829, 0.99953787, 0.99945125, 0.99938193, 0.99933839,
         0.99930488, 0.99926748, 0.99924242, 0.99924514, 0.99926275, 0.99929149, 0.99931654, 0.99933161, 0.99932689,
         0.9993071, 0.99927363, 0.99923712, 0.99919915, 0.99914045, 0.99907633, 0.99900216, 0.99891874, 0.99882671,
         0.99873562, 0.99863822, 0.99853303, 0.99841911, 0.99831145, 0.99820416, 0.99809438, 0.99798009, 0.99785222,
         0.99772708, 0.99761754, 0.99751595, 0.99740357, 0.99726706, 0.99709373, 0.99687835, 0.99661299, 0.99630414,
         0.99595392, 0.9955756, 0.99518221, 0.99476333, 0.99431237, 0.99381049, 0.99320721, 0.99249086, 0.99166128,
         0.99073179, 0.98972359, 0.98867498, 0.98761743, 0.98654746, 0.98536994, 0.98405218, 0.98266914, 0.98127292,
         0.97985243, 0.97817818, 0.97613982, 0.97395495, 0.97179702, 0.96951054, 0.96698265, 0.96399526, 0.95999319,
         0.95463326, 0.94802055, 0.94049785, 0.93241941, 0.92402846, 0.91541236, 0.90647692, 0.89714784, 0.88752895,
         0.87705071, 0.86563134, 0.85318403, 0.84041459, 0.8279652, 0.81597948, 0.80466526, 0.79439827])
    params['n'] = n
    params['pi'] = pop_stationary(n, params['phi'], T=params['T'])

    # INTEREST RATE
    params['r'] = r

    # TECHNOLOGY
    params['alpha'] = 0.33  # Capital share
    params['delta'] = 0.06  # Depreciation rate

    # PREFERENCES
    params['sigma'] = sigma  # *Inverse* elasticity of substitution
    params['beta'] = beta  # Subjective discount factor

    # INCOME PROCESS
    params['h'] = np.zeros((params['T']+1))
    params['h'][Tw:Tr] = -3 + 50 * params['jvec'][Tw:Tr] - 0.5 * params['jvec'][Tw:Tr] ** 2
    params['h'] = params['h'] / np.sum(params['pi'] * params['h'])

    # MASS OF RETIREES AND WORKES
    iret = 1 * (params['jvec'] >= params['Tr'])
    params['retirees'] = np.sum(params['pi'] * iret)
    params['workers'] = np.sum(params['pi'] * params['h'] * (1-iret))

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


"""
Fast linear interpolation for two ordered vectors of similar length; also iterates forward on 
distribution using linearized rule. uses decorator @jit from numba to speed things up.
"""

# Numba's guvectorize decorator compiles functions and allows them to be broadcast by NumPy when dimensions differ.
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


"""Various helper functions"""

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
    Efficiently test max(abs(x1-x2)) <= tol for arrays of same dimensions x1, x2
    """
    y1 = x1.ravel()
    y2 = x2.ravel()

    for i in range(y1.shape[0]):
        if np.abs(y1[i] - y2[i]) > tol:
            return False
    return True


def stationary(Pi, pi_seed=None, tol=1E-11, maxit=10_000):
    """
    Find invariant distribution of a Markov chain by iteration
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
    """Takes a path x (vector/matrix), and repeats the last line until x has T lines."""
    if x.ndim == 1:
        raise ValueError('x must be a column vector')

    if T < x.shape[0]:
        raise ValueError('T must be greater than the number of lines in x')

    return np.vstack((x, make_path(x[-1], T - x.shape[0])))


def pack_jacobians(jacdict, inputs, outputs, T):
    """If we have T*T jacobians from nI inputs to nO outputs in jacdict, combine into (nO*T)*(nI*T) jacobian matrix."""
    nI, nO = len(inputs), len(outputs)

    outjac = np.empty((nO * T, nI * T))
    for iO in range(nO):
        subdict = jacdict.get(outputs[iO], {})
        for iI in range(nI):
            outjac[(T * iO):(T * (iO + 1)), (T * iI):(T * (iI + 1))] = make_matrix(subdict.get(inputs[iI],
                                                                                               np.zeros((T, T))), T)
    return outjac


def unpack_jacobians(bigjac, inputs, outputs, T):
    """If we have an (nO*T)*(nI*T) jacobian and provide names of nO outputs and nI inputs, output nested dictionary"""
    nI, nO = len(inputs), len(outputs)

    jacdict = {}
    for iO in range(nO):
        jacdict[outputs[iO]] = {}
        for iI in range(nI):
            jacdict[outputs[iO]][inputs[iI]] = bigjac[(T * iO):(T * (iO + 1)), (T * iI):(T * (iI + 1))]
    return jacdict


def make_matrix(A, T):
    """If A is not an outright ndarray, e.g. it is SimpleSparse, call its .matrix(T) method
    to convert it to T*T array."""
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