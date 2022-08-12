from model.household import household_ss_olg
import numpy as np
from model.demographics import Population
from model.government import Government


def test_household_ss_olg():
    params = {
        "y_eps": np.array([1,1]),
        "N_eps": 2,
        "N_a": 10,
        "a": np.arange(0,10,1),
        "r": .01,
        "sigma": 1.,
        "Pi_eps": np.array([[0.9, .1], [.1, .9]]),
        "beta": .99,
    }
    ss = household_ss_olg(
        params,
        .5,
        Government(),
        Population()
    )
    assert np.allclose(
        [ss['A'], ss['C'], ss['C_j'][25]],
        [0.03339992780863382, 1.2400562852451857, 1.4087153783748283]
    )
