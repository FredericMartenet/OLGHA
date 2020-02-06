import scipy.optimize as opt
from utils import set_parameters
from production import ss_production
from government import government
from household import household_ss_olg


def ss_olg():
    # Set parameters of the model
    params = set_parameters()
    # Compute real wage and capital
    w, K_L = ss_production(params['r'], params['alpha'], params['delta'])
    K = K_L * params['workers']
    # Balance government budget
    tau, d = government(w, params['retirees'], tau=0.2)
    # Calibrate 'beta' such that A=K
    params['beta'] = opt.newton(error, x0=0.95, args=(K, params, r, w, tau, d))
    # Store steady-state objects at the calibrated 'beta'
    ss = household_ss_olg(params, params['r'], w, tau, d)

    return ss


def error(beta, K, params, w, tau, d):
    """Returns (A/N - K/N) at a certain 'beta'"""
    params['beta'] = beta
    return household_ss_olg(params, w, tau, d)['A'] - K
