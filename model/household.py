import numpy as np
from model.utils import find_nearest, interpolate_coord, forward_step, interpolate_y


def household_ss_olg(params, w, tau, d):
    """Solves the household's problem for a given real wage, interest rate, and social security policy."""
    # Construct labor income
    iret = 1 * (params['jvec'] >= params['Tr'])  # Retirement indicator
    y_js = (1 - tau) * w * params['y_eps'][np.newaxis, :] * params['h'][:, np.newaxis] + d * iret[:, np.newaxis]

    # allocate arrays to store results
    uc, c, a, D = (np.empty((params['T'] + 1, params['N_eps'], params['N_a'])) for _ in range(4))
    a[:params['Tw']], c[:params['Tw']], D[:params['Tw']] = 0.0, 0.0, 0.0  # Make sure = 0 before age Tw

    # Backward iteration to obtain policy function
    for j in reversed(range(params['Tw'], params['T']+1)):
        if j == params['T']:
            # Compute cash-on-hand and bequest factor
            coh_T = get_coh(y_js[j,], params['r'], params['phi'][j], params['a'])
            # Call backward iteration function
            a[j,], c[j,] = constrained(coh_T, params['a'])
            uc[j,] = c[j,] ** (-params['sigma'])

        # Compute cash-on-hand and bequest factor
        coh = get_coh(y_js[j-1,], params['r'], params['phi'][j-1], params['a'])
        # Call backward iteration function
        a[j-1,], c[j-1,], uc[j-1,] = \
            backward_iterate_olg(uc[j,], params['a'], params['Pi_eps'], coh, params['r'], params['beta'], params['sigma'])

    # initialize age-Tw distribution: point mass at 0
    Dst_start = np.zeros((params['N_eps'], params['N_a']))
    Dst_start[:, find_nearest(params['a'], 0.0)] = 1.0

    # to make matrix multiplication more efficient, make separate copy of Pi transpose
    Pi_T = params['Pi_eps'].T.copy()

    # forward iteration to obtain distributions at each future age
    for j in range(params['Tw'], params['T']):
        if j == params['Tw']:
            D_temp = Dst_start
            D[j,] = D_temp
        else:
            D_temp = D[j,]

        # get interpolated rule corresponding to a and iterate forward
        a_pol_i, a_pol_pi = interpolate_coord(params['a'], a[j,])
        D[j+1,] = forward_step(D_temp, Pi_T, a_pol_i, a_pol_pi)

    # Assets
    A_j = np.einsum('jsa,jsa->j', D, a)  # by age j
    A = np.einsum('j,j', params['pi'], A_j)  # Aggregate assets

    # Consumption
    C_j = np.einsum('jsa,jsa->j', D, c)  # by age j
    C = np.einsum('j,j', params['pi'], C_j)  # Aggregate consumption

    return {'D': D,  # Distribution of agents
            'a': a, 'A_j': A_j, 'A': A,  # Asset policy, by age, aggregate
            'c': c, 'C_j': C_j, 'C': C,  # Consumption policy, by age, aggregate
            }


def constrained(coh, a):
    """Calculate consumption for constrained agents (or in final period of life)"""
    c = coh - a[0]
    a = np.zeros_like(c)
    return a, c


def backward_iterate_olg(ucp, a, Pi, coh, r, beta, sigma):
    """One-step backward iteration function."""
    # Implied consumption today consistent with Euler equation (on tomorrow's grid for assets a')
    c_nextgrid = (beta * (1+r) * (Pi @ ucp)) ** (-1/sigma)

    # We have consumption today for each a' tomorrow (a mapping from total cash on hand today c''+a' to a' tomorrow)
    # Interpolate to get mapping of actual cash on hand in each state to assets tomorrow a'
    a_pol = interpolate_y(c_nextgrid + a, coh, a)

    # Derive the consumption policy function using the above mapping from cash-on-hand today to a' tomorrow and the
    # mapping from consumption today to a' (c_nextgrid)
    c_pol = interpolate_y(a, a_pol, c_nextgrid)

    # Replace constrained agents with minimum asset level (bottom of grid)
    cstr_i = a_pol < a[0]  # index of constrained bins
    if np.any(cstr_i):
        a_pol[cstr_i], c_pol[cstr_i] = constrained(coh[cstr_i], a)

    # Replace zeros by 1E-10 to compute marginal utility without dividing by zero
    c_pol[c_pol <= 0] = 1E-10

    # Calculate marginal utility
    uc = c_pol ** (-sigma)

    return a_pol, c_pol, uc


def get_coh(y_eps, r, phi, a):
    """Construct cash-on-hand."""
    return y_eps[:, np.newaxis] + (1 + r) / phi * a
