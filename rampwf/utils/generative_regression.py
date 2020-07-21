import numpy as np
import scipy.stats

# The maximum numbers of parameters a distribution would need
# Only matters for bagging mixture models in
# prediction_types.generative_regression
MAX_MDN_PARAMS = 6
EMPTY_DIST = -1

distributions_dict = {
 'norm': 0,
 'uniform': 1,
 'beta': 2,
 'truncnorm': 3,
 'foldnorm': 4,
 'vonmises': 5,
 'entropy': 6,
 'rv_discrete': 7,
 'rv_continuous': 8,
 'rv_histogram': 9,
 'ksone': 10,
 'kstwo': 11,
 'kstwobign': 12,
 'alpha': 13,
 'anglit': 14,
 'arcsine': 15,
 'betaprime': 16,
 'bradford': 17,
 'burr': 18,
 'burr12': 19,
 'fisk': 20,
 'cauchy': 21,
 'chi': 22,
 'chi2': 23,
 'cosine': 24,
 'dgamma': 25,
 'dweibull': 26,
 'expon': 27,
 'exponnorm': 28,
 'exponweib': 29,
 'exponpow': 30,
 'fatiguelife': 31,
 'foldcauchy': 32,
 'f': 33,
 'weibull_min': 34,
 'weibull_max': 35,
 'frechet_r': 36,
 'frechet_l': 37,
 'genlogistic': 38,
 'genpareto': 39,
 'genexpon': 40,
 'genextreme': 41,
 'gamma': 42,
 'erlang': 43,
 'gengamma': 44,
 'genhalflogistic': 45,
 'gompertz': 46,
 'gumbel_r': 47,
 'gumbel_l': 48,
 'halfcauchy': 49,
 'halflogistic': 50,
 'halfnorm': 51,
 'hypsecant': 52,
 'gausshyper': 53,
 'invgamma': 54,
 'invgauss': 55,
 'geninvgauss': 56,
 'norminvgauss': 57,
 'invweibull': 58,
 'johnsonsb': 59,
 'johnsonsu': 60,
 'laplace': 61,
 'levy': 62,
 'levy_l': 63,
 'levy_stable': 64,
 'logistic': 65,
 'loggamma': 66,
 'loglaplace': 67,
 'lognorm': 68,
 'gilbrat': 69,
 'maxwell': 70,
 'mielke': 71,
 'kappa4': 72,
 'kappa3': 73,
 'moyal': 74,
 'nakagami': 75,
 'ncx2': 76,
 'ncf': 77,
 't': 78,
 'nct': 79,
 'pareto': 80,
 'lomax': 81,
 'pearson3': 82,
 'powerlaw': 83,
 'powerlognorm': 84,
 'powernorm': 85,
 'rdist': 86,
 'rayleigh': 87,
 'loguniform': 88,
 'reciprocal': 89,
 'rice': 90,
 'recipinvgauss': 91,
 'semicircular': 92,
 'skewnorm': 93,
 'trapz': 94,
 'triang': 95,
 'truncexpon': 96,
 'tukeylambda': 97,
 'vonmises_line': 98,
 'wald': 99,
 'wrapcauchy': 100,
 'gennorm': 101,
 'halfgennorm': 102,
 'crystalball': 103,
 'argus': 104,
 'binom': 105,
 'bernoulli': 106,
 'betabinom': 107,
 'nbinom': 108,
 'geom': 109,
 'hypergeom': 110,
 'logser': 111,
 'poisson': 112,
 'planck': 113,
 'boltzmann': 114,
 'randint': 115,
 'zipf': 116,
 'dlaplace': 117,
 'skellam': 118,
 'yulesimon': 119
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
