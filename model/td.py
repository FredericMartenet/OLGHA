"""
Main routines for transitional dynamics.
"""

import numpy as np
import time
from model.utils import find_nearest, interpolate_coord, forward_step, make_full_path, unpack_vectors, pack_vectors
from model.household import get_coh, constrained, backward_iterate_olg


def td_olg(
        paths_trans,
        params,
        pi_trans,
        dst0,
        gov_ss,
        prod,
        pop,
        disp=False,
):
    """
    Compute the transitional dynamics for a given path of the interest rate.

    Parameters
    ----------
    paths_trans : dict
        Dict containing the path for the interest rate 'r'
    params : np.array
        Dict containing model parameters
    pi_trans : np.array
        Array with population distributions along the transition
    dst0: np.array
        Initial distribution of agents by state
    gov_ss: class
        Government class
    prod: class
        Production class
    pop: class
        Population class
    disp: bool, optional
        Display errors or not
    """
    start = time.time()

    # Transition length
    Ttrans = len(paths_trans['r'])

    # Total transition length for a complete cross-section in the last year Ttrans
    Ttransfull = Ttrans + pop.T - pop.Tw

    # FULL PATHS OF RELEVANT OBJECTS
    # ------------------------------

    # Interest rate
    paths_trans['r_full'] = make_full_path(paths_trans['r'], Ttransfull)

    # Demographics
    retirees, workers = pop.mass_ret_work(pi_trans)

    # Production aggregates
    w, kl = prod.agg(paths_trans['r_full'])
    K = kl.squeeze() * workers

    # Government policy
    gov_td = gov_ss.adjust(w.squeeze(), retirees)

    # Household income
    y_tjs = (
        (1 - gov_td.tau) * w[:, :, np.newaxis] * params['y_eps'][np.newaxis, np.newaxis, :]
        * pop.h[np.newaxis, :, np.newaxis]
        + gov_td.d[:, np.newaxis, np.newaxis] * pop.iret[np.newaxis, :, np.newaxis]
    )

    # BACKWARD ITERATIONS TO GET POLICIES
    # -----------------------------------

    uc, c, a, D = (np.empty((Ttransfull, pop.T + 1, params['N_eps'], params['N_a'])) for _ in range(4))

    # Make sure = 0 before age Tw
    a[:, :pop.Tw, :, :], c[:, :pop.Tw, :, :], D[:, :pop.Tw, :, :] = 0.0, 0.0, 0.0

    for cohort in reversed(range(-pop.T + pop.Tw, Ttrans)):
        # backward iteration on age for that cohort, starting from oldest
        for j in reversed(range(pop.Tw, pop.T+1)):

            # considering household age j born in c, so lives at t
            t = cohort + j - pop.Tw

            # only consider what happens after date 0, not relevant for the of agents born before 0
            if t >= 0:
                if j == pop.T:
                    # Compute cash-on-hand
                    coh_T = get_coh(y_tjs[t, j, ], paths_trans['r_full'][t], pop.phi[j], params['a'])

                    # Compute consumption
                    a[t, j, ], c[t, j, ] = constrained(coh_T, params['a'])
                    uc[t, j, ] = c[t, j, ] ** (-params['sigma'])

                # Compute cash-on-hand and bequest factor
                coh = get_coh(y_tjs[t-1, j-1, ], paths_trans['r_full'][t-1], pop.phi[j-1], params['a'])
                # call backward iteration function: period before is t-1, agents aged j-1
                a[t-1, j-1, ], c[t-1, j-1, ], uc[t-1, j-1, ] = \
                    backward_iterate_olg(uc[t, j, ], params['a'], params['Pi_eps'], coh,
                                         paths_trans['r_full'][t-1], params['beta'], params['sigma'])

    # FORWARD ITERATIONS TO GET DISTRIBUTIONS
    # ---------------------------------------

    # Initial distribution (given)
    D[0, ] = dst0

    # go forward at each t
    dst_start = np.zeros((params['N_eps'], params['N_a']))
    dst_start[:, find_nearest(params['a'], 0.0)] = 1.0
    for t in range(Ttrans):
        for j in range(pop.Tw, pop.T):
            if j == pop.Tw:
                dst_temp = dst_start
                D[t, j, ] = dst_temp
            else:
                dst_temp = D[t, j, ]

            # get interpolated rule corresponding to a at t,j,h and iterate forward to next age
            a_pol_i, a_pol_pi = interpolate_coord(params['a'], a[t, j, ])
            D[t+1, j+1, ] = forward_step(dst_temp, params['Pi_eps'].T.copy(), a_pol_i, a_pol_pi)

    # AGGREGATION
    # -----------

    # Truncate at Ttrans
    D_trunc, a_trunc, c_trunc, pi_trunc = D[:Ttrans, ], a[:Ttrans, ], c[:Ttrans, ], pi_trans[:Ttrans, ]

    # Assets
    A_j = np.einsum('tjsa,tjsa->tj', a_trunc, D_trunc)  # Asset profile
    A = np.einsum('tj,tj->t', pi_trunc, A_j)  # Aggregate assets

    # Consumption
    C_j = np.einsum('tjsa,tjsa->tj', c_trunc, D_trunc)
    C = np.einsum('tj,tj->t', pi_trunc, C_j)

    # Market clearing error
    nad = A[:Ttrans, ] - K[:Ttrans, ]

    end = time.time()
    if disp:
        print(f'Total time: {end-start:.2f} sec')

    return {
        # Paths of inputs
        **paths_trans,
        # Policy functions and distributions
        'uc': uc, 'c_pol': c, 'a_pol': a, 'D': D,
        # Assets
        'A': A, 'A_j': A_j,
        # Consumption
        'C': C, 'C_j': C_j,
        # Capital
        'K': K[:Ttrans,], 'K_L': kl,
        # Errors in asset market clearing condition
        'nad': nad,
    }


def get_jacobian(
        paths_trans,
        params,
        pi_trans,
        dst0,
        gov_ss,
        prod,
        pop,
        inputs,
        outcomes,
        diff=1E-4,
):
    """
    Computes the Jacobian matrix using numerical differentiation.

    Parameters
    ----------
    paths_trans : dict
        Dict containing the path for the interest rate 'r'
    params : np.array
        Dict containing model parameters
    pi_trans : np.array
        Array with population distributions along the transition
    dst0: np.array
        Initial distribution of agents by state
    gov_ss: class
        Government class
    prod: class
        Production class
    pop: class
        Population class
    inputs: list
        List of inputs to compute Jacobian with respect to
    outcomes: list
        List of outcomes to compute Jacobian of
    diff: float, optional
        Size of the deviation for numerical derivative
    """

    # Length of transition
    Ttrans = len(paths_trans[inputs[0]])

    # Store Jacobian is a nested dict
    jac = {o: {i: np.zeros((Ttrans, Ttrans)) for i in inputs} for o in outcomes}

    # Run transition function at the given input paths to compute derivative around
    td = td_olg(paths_trans, params, pi_trans, dst0, gov_ss, prod, pop, disp=False)

    # For all inputs, simulate with respect to a shock at each date up to Ttrans only
    for i in inputs:
        for t in range(Ttrans):
            # Make a copy of the inputs
            paths_trans_diff = paths_trans.copy()

            # Add a shock of magnitude 'diff' at horizon t
            paths_trans_diff[i] = paths_trans[i] + diff * (np.arange(Ttrans)[:, np.newaxis] == t)

            # Compute transition dynamics
            td_out = td_olg(paths_trans_diff, params, pi_trans, dst0, gov_ss, prod, pop, disp=False)

            # store results as column t of J[o][i] for each outcome o
            for o in outcomes:
                jac[o][i][:, t] = (td_out[o] - td[o]) / diff

    return jac


def td_ge(
        jac_mat,
        paths_trans,
        params,
        pi_trans,
        dst0,
        gov_ss,
        prod,
        pop,
        outcomes,
        inputs,
        maxit=50,
        xtol=1E-6,
        disp=True,
):
    """Compute general equilibrium solution using the Jacobian with Newton's method.

    Parameters
    ----------
    jac_mat : np.array
        Jacobian matrix
    paths_trans: dict
        Dict containing paths of inputs along the transition
    params : np.array
        Dict containing model parameters
    pi_trans : np.array
        Array with population distributions along the transition
    dst0: np.array
        Initial distribution of agents by state
    gov_ss: class
        Government class
    prod: class
        Production class
    pop: class
        Population class
    inputs: list
        List of inputs to compute Jacobian with respect to
    outcomes: list
        List of outcomes to compute Jacobian of
    maxit: float, optional
        Maximum number of iterations
    xtol: float, optional
        Tolerance parameter for convergence
    disp: bool, optional
        Display errors or not
    """

    # Invese Jacobian
    jac_mat_inv = np.linalg.inv(jac_mat)

    # Iterate until convergence
    for it in range(maxit):
        # Run PE transition dynamics
        td = td_olg(paths_trans, params, pi_trans, dst0, gov_ss, prod, pop, disp=False)

        # Store length of transition and inputs
        T = len(td[inputs[0]])
        Us = {i: np.full(T, paths_trans[i].squeeze()) for i in inputs}

        # Stack inputs into one vector
        Uvec = pack_vectors(Us, inputs, T)

        # Store errors
        err = {o: np.max(np.abs(td[o])) for o in outcomes}

        # Display errors
        if disp:
            print(f'On iteration {it}')
            for k in err:
                print(f'   max error for {k} is {err[k]:.2E}')

        # Stop if converged
        if all(v < xtol for v in err.values()):
            break

        else:
            # Update using inverse Jacobian
            Hvec = pack_vectors(td, outcomes, T)
            Uvec -= jac_mat_inv @ Hvec
            Us = unpack_vectors(Uvec, inputs, T)
            for i in inputs:
                paths_trans[i] = Us[i][:, np.newaxis]

    return td
