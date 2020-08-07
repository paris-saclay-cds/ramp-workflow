import numpy as np
import scipy.stats
import inspect

# The maximum numbers of parameters a distribution would need
# Only matters for bagging mixture models in
# prediction_types.generative_regression
MAX_MDN_PARAMS = 6
EMPTY_DIST = -1

# We explcitly enumerate all scipy distributions here so their codes
# do not change even if scipy adds new distributions. We considered but
# discarded using a hash code.
distributions_dict = {
 'norm': 0,
 'uniform': 1,
 'beta': 2,
 'truncnorm': 3,
 'foldnorm': 4,
 'vonmises': 5,
 'ksone': 6,
 'kstwo': 7,
 'kstwobign': 8,
 'alpha': 9,
 'anglit': 10,
 'arcsine': 11,
 'betaprime': 12,
 'bradford': 13,
 'burr': 14,
 'burr12': 15,
 'fisk': 16,
 'cauchy': 17,
 'chi': 18,
 'chi2': 19,
 'cosine': 20,
 'dgamma': 21,
 'dweibull': 22,
 'expon': 23,
 'exponnorm': 24,
 'exponweib': 25,
 'exponpow': 26,
 'fatiguelife': 27,
 'foldcauchy': 28,
 'f': 29,
 'weibull_min': 30,
 'weibull_max': 31,
 'frechet_r': 32,
 'frechet_l': 33,
 'genlogistic': 34,
 'genpareto': 35,
 'genexpon': 36,
 'genextreme': 37,
 'gamma': 38,
 'erlang': 39,
 'gengamma': 40,
 'genhalflogistic': 41,
 'gompertz': 42,
 'gumbel_r': 43,
 'gumbel_l': 44,
 'halfcauchy': 45,
 'halflogistic': 46,
 'halfnorm': 47,
 'hypsecant': 48,
 'gausshyper': 49,
 'invgamma': 50,
 'invgauss': 51,
 'geninvgauss': 52,
 'norminvgauss': 53,
 'invweibull': 54,
 'johnsonsb': 55,
 'johnsonsu': 56,
 'laplace': 57,
 'levy': 58,
 'levy_l': 59,
 'levy_stable': 60,
 'logistic': 61,
 'loggamma': 62,
 'loglaplace': 63,
 'lognorm': 64,
 'gilbrat': 65,
 'maxwell': 66,
 'mielke': 67,
 'kappa4': 68,
 'kappa3': 69,
 'moyal': 70,
 'nakagami': 71,
 'ncx2': 72,
 'ncf': 73,
 't': 74,
 'nct': 75,
 'pareto': 76,
 'lomax': 77,
 'pearson3': 78,
 'powerlaw': 79,
 'powerlognorm': 80,
 'powernorm': 81,
 'rdist': 82,
 'rayleigh': 83,
 'loguniform': 84,
 'reciprocal': 85,
 'rice': 86,
 'recipinvgauss': 87,
 'semicircular': 88,
 'skewnorm': 89,
 'trapz': 90,
 'triang': 91,
 'truncexpon': 92,
 'tukeylambda': 93,
 'vonmises_line': 94,
 'wald': 95,
 'wrapcauchy': 96,
 'gennorm': 97,
 'halfgennorm': 98,
 'crystalball': 99,
 'argus': 100,
 'binom': 101,
 'bernoulli': 102,
 'betabinom': 103,
 'nbinom': 104,
 'geom': 105,
 'hypergeom': 106,
 'logser': 107,
 'poisson': 108,
 'planck': 109,
 'boltzmann': 110,
 'randint': 111,
 'zipf': 112,
 'dlaplace': 113,
 'skellam': 114,
 'yulesimon': 115
}

_inverted_scipy_dist_dict = dict(map(reversed, distributions_dict.items()))


def distributions_dispatcher(d_type=-1):
    try:
        name = _inverted_scipy_dist_dict[d_type]
    except KeyError:
        raise KeyError("%s not a valid distribution type." % d_type)
    return getattr(scipy.stats, name)


def get_n_params(dist):
    return len(inspect.signature(dist._parse_args).parameters)


class MixtureYPred:
    def __init__(self):
        """
        Object made to convert outputs of generative regressors to a numpy array
        representation (y_pred) used in RAMP.
        Works for autoregressive and independent, not the full case.
        """
        self.dims = []

    def add(self, weights, types, params):
        """
        Must be called every time we get a prediction, creates the
        distribution list

        Parameters
        ----------
        weights : numpy array (n_timesteps, n_dist_per_dim)
            the weights of the mixture for current dim

        types : numpy array (n_timesteps, n_dist_per_dim)
            the types of the mixture for current dim

        params : numpy array (n_timesteps, n_dist_per_dim*n_param_per_dist)
            the params of the mixture for current dim, the order must
            correspond to the one of types
        """
        n_dists_curr = types.shape[1]
        sizes = np.full((len(types), 1), n_dists_curr)
        result = np.concatenate(
            (sizes, weights, types, params), axis=1)
        self.dims.append(result)
        return self

    def finalize(self, order=None):
        """
        Must called be once all the dims were added

        Parameters
        ----------
        order : list
            The order in which the dims should be sorted
        """
        dims_original_order = np.array(self.dims)
        if order is not None:
            dims_original_order = dims_original_order[np.argsort(order)]
        return np.concatenate(dims_original_order, axis=1)


def get_components(curr_idx, y_pred):
    """
    Extracts dimentions from the whole y_pred array to use them elsewhere
    (e.g. to compute the pdf).
    It is meant to be called like so:

    curr_idx=0
    for dim in dims:
        curr_idx, ... = currget_components(curr_idx, y_pred)

    Parameters
    ----------
    curr_idx : int
        The current index in the whole y_pred.
    y_pred : numpy array
        Should be built using MixtureYPred "add" and "finalize".

    Return
    ------
    curr_idx : int
        The current index in the whole y_pred after recovering the current
        dimension
    n_dists : int
        The number of components in the mixture for the current dim
    weights : numpy array (n_timesteps, n_dist_per_dim)
        The weights of the mixture for current dim
    types : numpy array (n_timesteps, n_dist_per_dim)
        The types of the mixture for current dim
    dists : list of objects extending AbstractDists
        A list of distributions to be used for current dim
    paramss : numpy array (n_timesteps, n_dist_per_dim*n_param_per_dist)
        The params of the mixture for current dim, that allign with the
        other returned values
    """
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
