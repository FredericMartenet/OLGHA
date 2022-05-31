"""
Main routines for government objects.
"""


class Government:
    """
    Government class that stored government parameters and methods to clear
    the government's budget constraint.

    Parameters
    ----------
    tau: float
        Social security tax rate
    d: float
        social security benefits
    adjust_rule: str
        adjust the government budget
    """
    def __init__(self, tau=0.2, d=2., adjust_rule='d'):
        self.tau = tau  # social security tax
        self.d = d  # social security benefits
        self.adjust_rule = adjust_rule  # adjustment rule chosen
        self.adjust_dispatch = {'d': self.adjust_d, 'tau': self.adjust_tau}

    def adjust(self, w, retirees):
        """
        Adjust the government's budget constraint.
        """
        return self.adjust_dispatch[self.adjust_rule](w, retirees)

    def adjust_tau(self, w, retirees):
        """
        Budget adustment using taxes.
        """
        tau = self.d / w * retirees
        return Government(tau, self.d, self.adjust_rule)

    def adjust_d(self, w, retirees):
        """
        Budget adustment using benefits.
        """
        d = self.tau * w / retirees
        return Government(self.tau, d, self.adjust_rule)
