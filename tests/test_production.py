from model.production import Production
import numpy as np


def test_f():
    assert np.isclose(Production().f(.5), .7955364837549187)


def test_fk():
    assert np.isclose(Production().fk(.5), .5250540792782463)


def test_fk_inv():
    assert np.isclose(Production().fk_inv(.5250540792782463), .5)


def test_fl():
    assert np.isclose(Production().fl(.5), .5330094441157954)


def test_ss():
    assert np.allclose(Production().ss(.01),  (1.43799461876806, 10.118085803698506))


def test_agg():
    assert np.allclose(Production().agg(.01),  (1.43799461876806, 10.118085803698506))
