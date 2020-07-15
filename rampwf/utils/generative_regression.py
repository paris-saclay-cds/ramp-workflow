import numpy as np
import scipy.stats

# The maximum numbers of parameters a distribution would need
# Only matters for bagging mixture models in
# prediction_types.generative_regression
MAX_MDN_PARAMS = 6
EMPTY_DIST = -1

distributions_dict = {
    "norm"     : 0,
    "uniform"  : 1,
    "beta"     : 2,
    "truncnorm": 3,
    "foldnorm" : 4,
    "vonmises" : 5
}

_inverted_scipy_dist_dict = dict(map(reversed, distributions_dict.items()))


# Bevington page 84.
# http://hosting.astro.cornell.edu/
# academics/courses/astro3310/Books/Bevington_opt.pdf
def rejection_sampling(
        pdf, params, xmin=0, xmax=1, discrete=False, nb_attemps=1000):
    if discrete:
        x = np.arange(xmin, xmax)
        x_sampler = np.random.randint
    else:
        x = np.linspace(xmin, xmax, nb_attemps)
        x_sampler = np.random.uniform
    y = pdf(x, params)
    pmin = 0.
    pmax = np.max(y)
    i = 0
    while i < nb_attemps:
        i += 1
        x = x_sampler(xmin, xmax)
        y = np.random.uniform(pmin, pmax)

        if y < pdf(x, params):
            return x
    return np.nan


def distributions_dispatcher(d_type=-1):
    try:
        name = _inverted_scipy_dist_dict[d_type]
    except KeyError:
        raise KeyError("%s not a valid distribution type." % d_type)
    return getattr(scipy.stats, name)


def sample_from_dist(d_type, params):
    dist = distributions_dispatcher(d_type)
    return dist.rvs(params)


def get_n_params(dist):
    shapes = dist.shapes
    n_params = 2
    if shapes is not None:
        n_params += len(dist.shapes)
    return n_params


def get_components(curr_idx, y_pred):
    n_dists = int(y_pred[0, curr_idx])
    curr_idx += 1
    id_params_start = curr_idx + n_dists * 2
    weights = y_pred[:, curr_idx:curr_idx + n_dists]
    assert (weights >= 0).all(), "Weights should all be positive."
    weights /= weights.sum(axis=1)[:, np.newaxis]
    types = y_pred[:, curr_idx + n_dists:id_params_start]
    curr_idx = id_params_start
    dists = []
    paramss = []
    for i in range(n_dists):
        non_empty_mask = ~np.array(types[:, i] == EMPTY_DIST)
        currtype = int(types[:, i][non_empty_mask][0])
        # TODO: raise exception if type is not consistent
        dists.append(distributions_dispatcher(currtype))
        end_params = curr_idx + get_n_params(dists[i])
        paramss.append(y_pred[:, curr_idx:end_params])
        curr_idx = end_params

    return curr_idx, n_dists, weights, types, dists, paramss
