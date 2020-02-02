import numpy as np
import time
from utils import find_nearest, interpolate_coord, forward_step, make_full_path, unpack_vectors, pack_vectors
from production import production_agg
from government import government
from household import get_coh, constrained, backward_iterate_olg


def td_olg(paths_trans, params, pi_trans, D0):
    """
    Compute the transitonal dynamics for a given path of the interest rate.

    Parameters
    ----------
    paths_trans :
        Dict containing the path for the interest rate 'r'
    params : np.ndarray
        Dict containing model parameters
    pi_trans : np.ndarray
        Array with population distributions along the transition
    D0: np.ndarray
        Initial distribution of agents

    """
    start = time.time()

    Ttrans = len(paths_trans['r'])  # Transition length
    Ttransfull = Ttrans + params['T'] - params['Tw']  # Total transition length for a complete cross-section in Ttrans

    # FULL PATHS OF RELEVANT OBJECTS
    # -------------------------

    # Interest rate
    paths_trans['r_full'] = make_full_path(paths_trans['r'], Ttransfull)

    # Demographics
    iret = 1 * (params['jvec'] >= params['Tr'])
    workers = np.sum(pi_trans * params['h'][np.newaxis,] * (1 - iret[np.newaxis,]), axis=1)
    retirees = np.sum(pi_trans * iret[np.newaxis,], axis=1)

    # Production aggregates
    w, K_L = production_agg(params['alpha'], paths_trans['r_full'], params['delta'])
    K = K_L.squeeze() * workers  # K/N ratio

    # Government policy
    tau, d = government(w.squeeze(), retirees)

    # Household income
    y_tjs = (1 - tau) * w[:, :, np.newaxis] * params['y_eps'][np.newaxis, np.newaxis, :] * \
            params['h'][np.newaxis, :, np.newaxis] + d[:, np.newaxis, np.newaxis] * iret[np.newaxis, :, np.newaxis]

    # BACKWARD ITERATIONS TO GET POLICIES
    # -----------------------------------
    uc, c, a, D = (np.empty((Ttransfull, params['T'] + 1, params['N_eps'], params['N_a'])) for _ in range(4))
    for cohort in reversed(range(-params['T'] + params['Tw'], Ttrans)):
        # backward iteration on age for that cohort, starting from oldest
        for j in reversed(range(params['Tw'], params['T']+1)):
            t = cohort + j - params['Tw']  # considering household age j born in c, so lives at t
            if t >= 0:  # only consider what happens after date 0, not relevant for the of agents born before 0
                if j == params['T']:
                    # Compute cash-on-hand
                    coh_T = get_coh(y_tjs[t,j,], paths_trans['r_full'][t], params['phi'][j], params['a'])
                    # Compute consumption
                    a[t,j,], c[t,j,] = constrained(coh_T, params['a'])
                    uc[t,j,] = c[t,j,] ** (-params['sigma'])

                # Compute cash-on-hand and bequest factor
                coh = get_coh(y_tjs[t-1,j-1,], paths_trans['r_full'][t-1], params['phi'][j-1], params['a'])
                # call backward iteration function: period before is t-1, agents aged j-1
                a[t-1,j-1,], c[t-1,j-1,], uc[t-1,j-1,] = \
                    backward_iterate_olg(uc[t,j,], params['a'], params['Pi_eps'], coh,
                                         paths_trans['r_full'][t-1], params['beta'], params['sigma'])

    # FORWARD ITERATIONS TO GET DISTRIBUTIONS
    # ---------------------------------------
    D[0,] = D0  # Initial distribution (given)
    Dst_start = np.zeros((params['N_eps'], params['N_a']))
    Dst_start[:, find_nearest(params['a'], 0.0)] = 1.0
    # go forward at each t
    for t in range(Ttrans):
        for j in range(params['Tw'], params['T']):
            if j == params['Tw']:
                D_temp = Dst_start
                D[t, j,] = D_temp
            else:
                D_temp = D[t, j,]

            # get interpolated rule corresponding to a at t,j,h and iterate forward to next age
            a_pol_i, a_pol_pi = interpolate_coord(params['a'], a[t,j,])
            D[t+1,j+1,] = forward_step(D_temp, params['Pi_eps'].T.copy(), a_pol_i, a_pol_pi)

    # AGGREGATION
    # -----------
    D_trunc, a_trunc, c_trunc, pi_trunc = D[:Ttrans,], a[:Ttrans,], c[:Ttrans,], pi_trans[:Ttrans,]  # Truncate

    # Assets
    A_j = np.einsum('tjsa,tjsa->tj', a_trunc, D_trunc)  # Asset profile
    A = np.einsum('tj,tj->t', pi_trunc, A_j)  # Aggregate assets

    # Consumption
    C_j = np.einsum('tjsa,tjsa->tj', c_trunc, D_trunc)
    C = np.einsum('tj,tj->t', pi_trunc, C_j)

    # Market clearing error
    nad = A[:Ttrans,] - K[:Ttrans,]

    end = time.time()
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
        'K': K, 'K_L': K_L,
        # Errors in asset market clearing condition
        'nad': nad,
    }


def get_Jacobian(paths_trans, params, pi_trans, D0, inputs, outcomes, diff=1E-4):
    """"Computes the Jacobian matrix using numerical differentiation."""
    Ttrans = len(paths_trans[inputs[0]])  # Length of transition
    J = {o: {i: np.zeros((Ttrans,Ttrans)) for i in inputs} for o in outcomes}  # Jacobian is a nested dict
    td = td_olg(paths_trans, params, pi_trans, D0)  # Run transition function at the given input paths
    # For all inputs, simulate with respect to a shock at each date up to Ttrans only
    for i in inputs:
        for t in range(Ttrans):
            paths_trans_diff = paths_trans.copy()  # Make a copy of the inputs
            # Add a shock of magnitude 'diff' at horizon t
            paths_trans_diff[i] = paths_trans[i] + diff * (np.arange(Ttrans)[:,np.newaxis] == t)
            # Compute transition dynamics
            td_out = td_olg(paths_trans_diff, params, pi_trans, D0)
            # store results as column t of J[o][i] for each outcome o
            for o in outcomes:
                J[o][i][:, t] = (td_out[o] - td[o]) / diff

    return J


def td_GE_olg(H_X, paths_trans, params, pi_trans, D0, outcomes, inputs, maxit=50, xtol=1E-8, disp=False):
    """Compute general equilibrium solution using the Jacobian with Newton's method."""
    H_X_inv = np.linalg.inv(H_X)  # Invese Jacobian
    for it in range(maxit):  # Iterate until convergence
        td_GE = td_olg(paths_trans, params, pi_trans, D0)  # Run PE transition dynamics
        T = len(td_GE[inputs[0]])  # Store length of transition
        Us = {i: np.full(T, paths_trans[i].squeeze()) for i in inputs}  # Store inputs
        Uvec = pack_vectors(Us, inputs, T)  # Stack inputs into one vector
        err = {o: np.max(np.abs(td_GE[o])) for o in outcomes}  # Store errors
        if disp:  # Display errors
            print(f'On iteration {it}')
            for k in err:
                print(f'   max error for {k} is {err[k]:.2E}')
        if all(v < xtol for v in err.values()):  # Stop if converged
            break
        else:  # Update using inverse Jacobian
            Hvec = pack_vectors(td_GE, outcomes, T)
            Uvec -= H_X_inv @ Hvec
            Us = unpack_vectors(Uvec, inputs, T)
            for i in inputs:
                paths_trans[i] = Us[i][:, np.newaxis]

    return td_GE