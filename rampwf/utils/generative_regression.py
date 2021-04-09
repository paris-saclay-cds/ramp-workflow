import inspect

import numpy as np
from scipy import stats
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_random_state

# The maximum numbers of parameters a distribution would need
# Only matters for bagging mixture models in
# prediction_types.generative_regression
MAX_MIXTURE_PARAMS = 6
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
    return getattr(stats, name)


def get_n_params(dist):
    return len(inspect.signature(dist._parse_args).parameters)


class MixtureYPred:
    """
    Object made to convert outputs of generative regressors to a numpy array
    representation (y_pred) used in RAMP.
    Works for autoregressive and independent, not the full case.
    """

    def __init__(self):
        self.dims = []

    def add(self, weights, types, params):
        """
        Must be called every time we get a prediction, creates the
        distribution list

        Parameters
        ----------
        weights : numpy array (n_timesteps, n_component_per_dim)
            the weights of the mixture for current dim

        types : numpy array (n_timesteps, n_component_per_dim)
            the types of the mixture for current dim

        params : numpy array (n_timesteps,
                              n_component_per_dim*n_param_per_dist)
            the params of the mixture for current dim, the order must
            correspond to the one of types
        """
        n_components_curr = types.shape[1]
        sizes = np.full((len(types), 1), n_components_curr)
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
    """Extracts dimensions from the whole y_pred array.

    These dimensions can then be used elsewhere (e.g. to compute the pdf).
    It is meant to be called like so:

    curr_idx=0
    for dim in dims:
        curr_idx, ... = get_components(curr_idx, y_pred)

    Parameters
    ----------
    curr_idx : int
        The current index in the whole y_pred.
    y_pred : numpy array
        An array built using MixtureYPred "add" and "finalize".

    Return
    ------
    curr_idx : int
        The current index in the whole y_pred after recovering the current
        dimension
    n_components : int
        The number of components in the mixture for the current dim
    weights : numpy array (n_timesteps, n_component_per_dim)
        The weights of the mixture for current dim
    types : numpy array (n_timesteps, n_component_per_dim)
        The types of the mixture for current dim
    dists : list of objects extending AbstractDists
        A list of distributions to be used for current dim
    paramss : numpy array (n_timesteps, n_component_per_dim*n_param_per_dist)
        The params of the mixture for current dim, that align with the
        other returned values
    """
    n_components = int(y_pred[0, curr_idx])
    curr_idx += 1
    id_params_start = curr_idx + n_components * 2
    weights = y_pred[:, curr_idx:curr_idx + n_components]
    assert (weights >= 0).all(), "Weights should all be positive."
    weights /= weights.sum(axis=1)[:, np.newaxis]
    types = y_pred[:, curr_idx + n_components:id_params_start]
    curr_idx = id_params_start
    dists = []
    paramss = []
    for i in range(n_components):
        non_empty_mask = ~np.array(types[:, i] == EMPTY_DIST)
        curr_types = types[:, i][non_empty_mask]
        curr_type = curr_types[0]
        assert np.all(curr_type == curr_types)  # component types must be fixed
        dists.append(distributions_dispatcher(curr_type))
        end_params = curr_idx + get_n_params(dists[i])
        paramss.append(y_pred[:, curr_idx:end_params])
        curr_idx = end_params
    return curr_idx, n_components, weights, types, dists, paramss


class BaseGenerativeRegressor(BaseEstimator):
    """Base class for generative regressors.

    Provides a sample method for generative regressors which return an explicit
    density (they have a predict method).
    Provides a predict method for generative regressors which do not have an
    explicity density but can be sampled from easily (they have a sample
    method).

    Parameters
    ----------
    decomposition : None or string
        Decomposition of the joint distribution for multivariate outputs.
    """
    # number of samples used to estimate the conditional distribution of the
    # output given the input. we use a class attribute to not have to
    # to call super().__init__() in the submissions.
    n_samples = 30

    def __init__(self):
        # this method is here to be able to instantiate the class for testing
        # purpose
        self.decomposition = None

    def samples_to_distributions(self, samples):
        """Estimate output conditional distributions.

        For each timestep, estimate the conditional output distribution from
        the draws contained in the samples array. The distribution is
        estimated with a kernel density estimator thus returning a mixture.

        This method is useful for generative regressors that can only be
        sampled from but do not provide an explicit likelihood. Examples
        of such generative regressors are Variational Auto Encoders and flow
        based methods.

        Parameters
        ----------
        samples : numpy array of shape [n_timesteps, n_targets, n_samples]
            For each timestep, an array of samples sampled from a generative
            regressor.

        Return
        ------
        weights : numpy array of float
            discrete probabilities of each component of the mixture
        types : list of strings
            scipy names referring to component of the mixture types.
            see https://docs.scipy.org/doc/scipy/reference/stats.html.
            In this case, they are normal
        params : numpy array
            Parameters for each component in the mixture. mus are the given
            samples, sigmas are estimated using silverman method.
        """
        n_timesteps, n_targets, n_samples = samples.shape
        mus = samples
        weights = np.full((n_timesteps, n_targets * n_samples), 1 / n_samples)
        sigmas = np.empty((n_timesteps, n_targets, n_samples))
        for i in range(n_timesteps):
            kde = stats.gaussian_kde(samples[i, ...], bw_method='silverman')
            bandwidths = np.sqrt(np.diag(kde.covariance)).reshape(-1, 1)
            sigmas[i, ...] = np.repeat(bandwidths, n_samples, axis=1)

        params = np.empty((n_timesteps, mus.shape[1] * mus.shape[2] * 2))
        params[:, 0::2] = mus.reshape(n_timesteps, -1)
        params[:, 1::2] = sigmas.reshape(n_timesteps, -1)
        types = ['norm'] * n_samples * mus.shape[1]

        return weights, types, params

    def _sample(self, distribution, rng):
        """Draw one sample from the input distribution.

        The distribution is assumed to be a mixture (a gaussian mixture if
        decomposition is set to None).

        Parameters
        ----------
        distribution : tuple of numpy arrays (weights, types, parameters)
            A mixture distribution characterized by weights, distribution types
            and associated distribution parameters.

        rng : Random state object
            The RNG or the state of the RNG to be used when sampling.

        Returns
        -------
        y_sampled : numpy array, shape (1, n_targets) if decomposition is not
        None, shape (n_samples, n_targets) if decomposition is None
            The sampled targets. n_targets is equal to 1 if decomposition is
            not None.
        """
        if self.decomposition is None:
            weights, types, params = distribution
            # the weights are all the same for each dimension: we keep only
            # the ones of the first dimension, the final shape is
            # (n_samples, n_components)
            n_samples = weights.shape[0]
            weights = weights.reshape(self._n_targets, n_samples, -1)[0]

            # we convert the params mus and sigmas back to their shape
            # (n_samples, n_targets, n_components) as it is then easier to
            # retrieve the ones that we need.
            all_mus = params[:, 0::2].reshape(n_samples, self._n_targets, -1)
            all_sigmas = params[:, 1::2].reshape(
                n_samples, self._n_targets, -1)

            # sample from the gaussian mixture
            weights /= np.sum(weights, axis=1)[:, np.newaxis]
            # vectorize sampling of one component for each sample
            cum_weights = weights.cumsum(axis=1)
            sampled_components = (
                (cum_weights > rng.rand(n_samples)[:, np.newaxis])
                .argmax(axis=1))
            # get associated means and sigmas
            all_ind = np.arange(n_samples)
            sampled_means = all_mus[all_ind, :, sampled_components]
            sampled_sigmas = all_sigmas[all_ind, :, sampled_components]

            y_sampled = rng.randn(n_samples, self._n_targets) * sampled_sigmas
            y_sampled += sampled_means
        else:  # autoregressive or independent decomposition.
            weights, types, params = distribution

            n_dists = len(types)
            try:
                types = [distributions_dict[type_name] for type_name in types]
            except KeyError:
                message = ('One of the type names is not a valid Scipy '
                           'distribution')
                raise AssertionError(message)
            types = np.array([types, ] * len(weights))

            w = weights[0].ravel()
            w = w / sum(w)
            selected = rng.choice(n_dists, p=w)
            dist = distributions_dispatcher(int(types[0, selected]))

            # find which params to take: this is needed if we have a
            # mixture of different distributions with different number of
            # parameters
            sel_id = 0
            for k in range(selected):
                curr_type = distributions_dispatcher(int(types[0, k]))
                sel_id += get_n_params(curr_type)
            y_sampled = dist.rvs(
                *params[0, sel_id:sel_id + get_n_params(dist)])
            y_sampled = np.array(y_sampled)
        return y_sampled

    def sample(self, X, rng=None, restart=None):
        """Draw a sample from the conditional output distribution given X.

        X is assumed to contain only one timestep. The conditional output
        distribution given X is predicted and a sample is drawn from this
        distribution.

        This method must be overriden for generative regressors that can be
        naturally sampled from such as Variational Auto Encoders.

        Parameters
        ----------
        X : numpy array, shape (1, n_features)
            Input timestep for which we want to sample the output from the
            conditional predicted distribution.
        rng : Random state object
            The RNG or the state of the RNG to be used when sampling.
        restart : string
            Name of the restart column. None is no restart.
        """
        rng = check_random_state(rng)

        if restart is not None:
            distribution = self.predict(X, restart)
        else:
            distribution = self.predict(X)

        return self._sample(distribution, rng)

    def predict(self, X, restart=None):
        """Predict conditional output distributions for each timestep in X.

        This method is to be used for generative regressors only providing a
        sample method. Samples are drawn with the sample method and
        distributions are estimated from these samples.

        This method should be overriden for generative regressors providing
        an explicity density.

        Parameters
        ----------
        X : numpy array, shape (n_timesteps, n_features)
            Input timesteps for which we want an estimated output distribution.
        restart : string
            Name of the restart column. None is no restart.
        """
        samples = []
        for _ in range(self.n_samples):
            if restart is not None:
                sampled = self.sample(X, restart)
            else:
                sampled = self.sample(X)
            samples.append(sampled)
        samples = np.stack(samples, axis=2)

        return self.samples_to_distributions(samples)
