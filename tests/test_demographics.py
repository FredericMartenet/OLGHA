from model.demographics import Population
import numpy as np


def test_pop_stationary():
    assert np.isclose(Population(Tr=22, T=24).pop_stationary(0.01)[3:5].sum(), .08699898205023288)


def test_trans_pop():
    pop = Population(Tr=22, T=24)
    assert np.isclose(pop.trans_pop(pop.pi, pop.n, Ttrans=3)[2][3:5].sum(), .09385972496793107)


def test_mass_ret_work():
    pop = Population(Tr=22, T=24)
    assert np.isclose(pop.mass_ret_work(pop.pi)[0], .09513369055719238)
