import numpy as np


def normpdf_custom(x, mean, sd):
    assert_normal(sd)
    var = sd ** 2
    denom = (2 * np.pi * var) ** .5
    diff = x - mean
    num = np.exp(-diff ** 2 / (2 * var))
    probs = num / denom
    return probs


def get_pdf_from_dist(y, type, params):
    if type[0] == 0:
        return normpdf_custom(y, params[:,0], params[:,1])
    return probs


def assert_normal(sd):
    assert np.all(sd>0), "Make sure all sigmas are positive (second parameter)"
