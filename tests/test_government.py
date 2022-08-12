import numpy as np
from model.government import Government


def test_adjust():
    tau = Government(adjust_rule='tau').adjust(.5,.5).tau
    d = Government(adjust_rule='d').adjust(.5,.5).d
    assert np.allclose([tau, d], [2.,.2])
