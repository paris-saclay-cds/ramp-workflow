import numpy as np


def norm_pdf(x, mean, sd):
    assert_normal(sd)
    var = sd ** 2
    denom = (2 * np.pi * var) ** .5
    diff = x - mean
    num = np.exp(-diff ** 2 / (2 * var))
    probs = num / denom
    return probs


def assert_normal(sd):
    assert np.all(sd > 0), "Make sure all sigmas are positive " \
                           "(second parameter)"


def uniform_pdf(x, a, b):
    assert_uniform(a, b)
    probs = np.zeros(x.shape)
    probs= np.where(np.logical_and(x >= a, x <= b), 1. / (b - a) , probs)
    return probs


def assert_uniform(a, b):
    assert np.all(a < b), "Make sure all \"a\" (first parameter)" \
                          " are bigger than \"b\" (second parameter)"


def get_pdf_from_dist(y, type, params):
    # All the distributions for a given dimension and a given y are the same.
    # The value need to be repeated to still be present when doing CV
    type_cure_dists = type[0]

    if type_cure_dists == 0:
        return norm_pdf(y, params[:, 0], params[:, 1])
    if type_cure_dists == 1:
        return uniform_pdf(y, params[:, 0], params[:, 1])
    return probs
