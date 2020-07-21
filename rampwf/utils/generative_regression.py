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
 'vonmisesentropy': 5,
 'rv_discrete': 6,
 'rv_continuous': 7,
 'rv_histogram': 8,
 'ksone': 9,
 'kstwo': 10,
 'kstwobign': 11,
 'alpha': 12,
 'anglit': 13,
 'arcsine': 14,
 'betaprime': 15,
 'bradford': 16,
 'burr': 17,
 'burr12': 18,
 'fisk': 19,
 'cauchy': 20,
 'chi': 21,
 'chi2': 22,
 'cosine': 23,
 'dgamma': 24,
 'dweibull': 25,
 'expon': 26,
 'exponnorm': 27,
 'exponweib': 28,
 'exponpow': 29,
 'fatiguelife': 30,
 'foldcauchy': 31,
 'f': 32,
 'weibull_min': 33,
 'weibull_max': 34,
 'frechet_r': 35,
 'frechet_l': 36,
 'genlogistic': 37,
 'genpareto': 38,
 'genexpon': 39,
 'genextreme': 40,
 'gamma': 41,
 'erlang': 42,
 'gengamma': 43,
 'genhalflogistic': 44,
 'gompertz': 45,
 'gumbel_r': 46,
 'gumbel_l': 47,
 'halfcauchy': 48,
 'halflogistic': 49,
 'halfnorm': 50,
 'hypsecant': 51,
 'gausshyper': 52,
 'invgamma': 53,
 'invgauss': 54,
 'geninvgauss': 55,
 'norminvgauss': 56,
 'invweibull': 57,
 'johnsonsb': 58,
 'johnsonsu': 59,
 'laplace': 60,
 'levy': 61,
 'levy_l': 62,
 'levy_stable': 63,
 'logistic': 64,
 'loggamma': 65,
 'loglaplace': 66,
 'lognorm': 67,
 'gilbrat': 68,
 'maxwell': 69,
 'mielke': 70,
 'kappa4': 71,
 'kappa3': 72,
 'moyal': 73,
 'nakagami': 74,
 'ncx2': 75,
 'ncf': 76,
 't': 77,
 'nct': 78,
 'pareto': 79,
 'lomax': 80,
 'pearson3': 81,
 'powerlaw': 82,
 'powerlognorm': 83,
 'powernorm': 84,
 'rdist': 85,
 'rayleigh': 86,
 'loguniform': 87,
 'reciprocal': 88,
 'rice': 89,
 'recipinvgauss': 90,
 'semicircular': 91,
 'skewnorm': 92,
 'trapz': 93,
 'triang': 94,
 'truncexpon': 95,
 'tukeylambda': 96,
 'vonmises_line': 97,
 'wald': 98,
 'wrapcauchy': 99,
 'gennorm': 100,
 'halfgennorm': 101,
 'crystalball': 102,
 'argus': 103,
 'binom': 104,
 'bernoulli': 105,
 'betabinom': 106,
 'nbinom': 107,
 'geom': 108,
 'hypergeom': 109,
 'logser': 110,
 'poisson': 111,
 'planck': 112,
 'boltzmann': 113,
 'randint': 114,
 'zipf': 115,
 'dlaplace': 116,
 'skellam': 117,
 'yulesimon': 118
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
