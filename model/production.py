"""
Main routine for production side of the economy.
"""


class Production:
    """
    Production parameters and functions.

    Parameters
    ----------
    alpha: float
        Capital share
    delta: float
        Depreciation rate
    """

    def __init__(self, alpha=.33, delta=.06):
        self.alpha = alpha
        self.delta = delta

    def f(self, k, l=1.0):
        """
        Production function.
        """
        return k ** self.alpha * l ** (1 - self.alpha)

    def fk(self, k, l=1.0):
        """
        Marginal product of capital.
        """
        return self.alpha * self.f(k, l) / k

    def fk_inv(self, fk):
        """
        Inverse marginal product: returns capital-labor ratio compatible with given fk.
        """
        return (self.alpha / fk) ** (1 / (1 - self.alpha))

    def fl(self, koverl):
        """
        Return marginal product of labor at given K/L ratio.
        """
        return (1 - self.alpha) * koverl ** self.alpha

    def ss(self, r):
        """
        Returns steady-state capital-labor ratio (K/l) and wage wate (w).
        """
        kl = (self.alpha / (r + self.delta)) ** (1 / (1 - self.alpha))
        w = self.fl(kl)
        return w, kl

    def agg(self, r):
        """
        Calculates K/L ratio and wage rate from a given interest rate and technological parameters.
        """
        kl = self.fk_inv(r + self.delta)
        w = self.fl(kl)
        return w, kl
