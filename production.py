

def f(k, l, alpha):
    """Production function (eta=1 for Cobb Douglas otherwise CES). Zstar must imply Y=1 in steady-state."""
    return k ** alpha * l ** (1 - alpha)


def fk(k, alpha, l = 1.0):
    """Returns marginal product of capital."""
    return alpha * f(k, l, alpha) / k


def fk_inv(fk, alpha):
    """Inverse marginal product: returns capital-labor ratio compatible with given fk."""
    return (alpha / fk) ** (1 / (1 - alpha))


def fl(koverl, alpha):
    """Return marginal product of labor at given K/L ratio."""
    return (1 - alpha) * koverl ** alpha


def ss_production(r, alpha, delta):
    K_L = (alpha / (r + delta)) ** (1 / (1 - alpha))
    w = fl(K_L, alpha)
    return w, K_L



def production_agg(alpha, r, delta):
    """Calculates various aggregates from a given interest rate and technological parameters."""
    K_L = fk_inv(r + delta, alpha)  # Calculate capital-labor ratio at this r
    w = fl(K_L, alpha)  # Calculate real wage at this capital-labor ratio
    return w, K_L