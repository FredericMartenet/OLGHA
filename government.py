
def government(w, retirees, tau=0.2):
    """Computes the level of benefits consistent with budget balance for a given mass of retirees and real wage."""
    d = tau * w / retirees
    return tau, d
