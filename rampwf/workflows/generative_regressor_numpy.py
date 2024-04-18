import os
import json

import numpy as np
from scipy.stats import norm
from sklearn.utils.validation import check_random_state

from ..utils.importing import import_module_from_source
from ..utils import MixtureYPred, distributions_dict
from .base_workflow import BaseWorkflow


class GenerativeRegressorNumpy(BaseWorkflow):
    """Generative regressor workflow with numpy inputs and outputs.

    Relying only on numpy and not on pandas allows to gain on speed when
    calling the regressor multiple times sequentially (e.g. in MPC).

    The generative regressor submission expected by this workflow can specify a
    `decomposition` attribute indicating how the conditional joint distribution
    of the targets is decomposed. This attribute can either be set to None,
    'autoregressive' or 'independent'. If no attribute is specified the
    submission is assumed to be an autoregressive one (i.e., similar to
    decomposition='autoregressive').

        - If None, the submission is expected to return the parameters of a
        multivariate Gaussian mixture where each Gaussian component has a
        diagonal covariance matrix.

        - If 'autoregressive', the conditional joint distribution of the
        targets is decomposed using the chain rule
        p(y_1, y_2, ...| x) = p(y_1 | x) * p(y_2 | y_1, x) * ... The submission
        is expected to return the parameters of a 1d mixture, one 1d mixture
        being learnt for each component of the chain rule decomposition.
        The parameters of the 1d mixture for the target dimension j are learnt
        using the values of the previous target dimensions (<j) as additional
        inputs. The order of the decomposition can be specified in an
        `order.json` file located in the submission folder. If there is no such
        file, the default order is used.

        - If 'independent', the conditional joint distribution of the
        targets is decomposed as if the targets were independent
        p(y_1, y_2, ...| x) = p(y_1 | x) * p(y_2 | x) * ... The submission
        is expected to return the parameters of a 1d mixture, one 1d mixture
        being learnt for each target dimension. Compared to 'autoregressive'
        each 1d mixture is learnt for y_j without the knowledge of the other
        target dimensions y_1, ..., y_{j-1}.

        The multivariate Gaussian mixture returned by a multivariate generative
        regressor submission is assumed to be such that each Gaussian component
        has a diagonal covariance matrix. The regressor is parametrized as
        follows:
            weights : numpy array (n_timesteps, n_components_per_dim*n_dims)
                The weights of the mixtures.
                They should sum up to one for each instance.
            types : numpy array (n_timesteps, n_components_per_dim*n_dims)
                The types of the mixtures.
                Only gaussian is supported in this case.
                For more info look at rampwf.utils.distributions_dict.
            params : numpy array (n_timesteps,
                                  n_components_per_dim*n_param_per_dist*n_dims)
                The params of the mixture for current dim, the order must
                correspond to the one of types

        The mixture returned by a 1d generative regressor (decomposition set to
        'autoregressive' or 'independent') is parameterized as follows:
            weights : numpy array (n_timesteps, n_components_per_dim)
                The weights of the mixture for current dim.
                They should sum up to one for each instance.
            types : numpy array (n_timesteps, n_components_per_dim)
                The types of the mixture for current dim
                For more info look at rampwf.utils.distributions_dict
            params : numpy array (n_timesteps,
                                  n_components_per_dim*n_param_per_dist)
                the params of the mixture for current dim, the order must
                correspond to the one of types

    Parameters
    ----------
    max_n_components : int
        The maximum number of components a generative regressor can output for
        its returned mixture.

    target_column_names : list of strings
        Names of the target columns.

    restart_name : string
        Name of the restart column.
    """

    def __init__(self, target_column_names, max_n_components,
                 check_sizes=None, check_indexs=None,
                 workflow_element_names=None,
                 restart_name=None,
                 **kwargs):
        if workflow_element_names is None:
            workflow_element_names = ['generative_regressor']
        self.check_indexs = check_indexs
        self.check_sizes = check_sizes
        self.element_names = workflow_element_names
        self.target_column_names = target_column_names
        self.max_n_components = max_n_components
        self.restart_name = restart_name
        self.kwargs = kwargs

    def _reorder_targets(self, module_path, y_array):
        """Find submitted order and reorder the targets."""
        order_path = os.path.join(module_path, 'order.json')
        try:
            with open(order_path, "r") as json_file:
                order = json.load(json_file)
                # Check if the names in the order and observables are all here
                if set(order.keys()) == set(self.target_column_names):
                    # We sort the variable names by user-defined order
                    order = [k for k, _ in sorted(
                        order.items(), key=lambda item: item[1])]
                    # Map it to original order
                    order = [self.target_column_names.index(i) for i in order]
                    print(order)
                    y_array = y_array[:, order]
                else:
                    raise RuntimeError("Order variables are not correct")
        except FileNotFoundError:
            print("Using default order")
            order = range(len(self.target_column_names))

        return y_array, order

    def train_submission(self, module_path, X_array, y_array, train_is=None):
        """Train submission.

        Parameters
        ----------
        module_path : string
            Path of the submission.

        X_array : numpy array, shape (n_samples, n_features)
            Data that will be used as inputs.

        y_array : numpy array, shape (n_samples, n_targets)
            Targets. Should be a 2D array even if there is only one target
            dimension.

        train_is : numpy array, shape (n_training_points)
            The training indices.

        Returns
        -------
        regressors : list of regressor
            Fitted generative regressor. If decomposition is None, the list
            contains only one element. Otherwise, the list contains the
            regressors for all the target dimensions.

        order : list
            The order of the target dimensions. Only matters for
            autoregressive=True.
        """
        if y_array.ndim != 2:
            raise ValueError('y_array should be a 2D array')
        if train_is is None:
            train_is = slice(None, None, None)

        generative_regressor = import_module_from_source(
            os.path.join(module_path, self.element_names[0] + '.py'),
            self.element_names[0],
            sanitize=False
        )

        X_array = X_array[train_is]
        y_array = y_array[train_is]

        reg = generative_regressor.GenerativeRegressor(
            self.max_n_components, 0, **self.kwargs)

        decomposition = getattr(reg, 'decomposition', 'autoregressive')
        if decomposition not in [None, 'autoregressive', 'independent']:
            raise ValueError(
                'decomposition attribute should be None, autoregressive '
                'or independent. It is {}'.format(decomposition))

        if decomposition is None:
            # set the _n_targets attribute needed in the _sample method of
            # BaseGenerativeRegressor. we use a property to prevent the
            # attribute from being changed later and raise an informative error
            if hasattr(reg, '_n_targets'):
                raise AttributeError(
                    'The _n_targets attribute is used internally by '
                    'ramp-workflow and cannot be used. Please use another '
                    'name for your attribute.')
            n_targets = len(self.target_column_names)

            def getter(self):
                return n_targets

            def setter(self, value):
                raise AttributeError(
                    'Cannot set attribute _n_targets as it is used internally '
                    'by ramp-workflow. Please use another name for your '
                    'attribute.')

            generative_regressor.GenerativeRegressor._n_targets = property(
                getter, setter)

            # return order for compatibility with autoregressive
            order = range(len(self.target_column_names))

            reg.fit(X_array, y_array)

            # use a list for compatibility with other decomposition values
            regressors = [reg]
        else:
            # autoregressive or independent decomposition
            # fit one regressor for each target dimension
            # reorder targets if order is given in submission
            y_array, order = self._reorder_targets(module_path, y_array)

            regressors = []
            for j in range(len(self.target_column_names)):
                if j != 0:
                    reg = generative_regressor.GenerativeRegressor(
                        self.max_n_components, j, **self.kwargs)

                y = y_array[:, j].reshape(-1, 1)
                reg.fit(X_array, y)

                if decomposition == 'autoregressive':
                    # add the current target dimension to the inputs used to
                    # train the next target dimension regressor
                    X_array = np.hstack([X_array, y])
                regressors.append(reg)

        return regressors, order

    def test_submission(self, trained_model, X_array):
        """Test submission.

        Note that if decomposition is None, the model is evaluated using the
        autoregressive decomposition of a multi-d Gaussian mixture.

        Parameters
        ----------
        trained_model : pair of lists
            The trained models and the order returned by train_submission.

        X_array : numpy array, shape (n_samples, n_features)
            Data that will be used as inputs. The data also contains the
            target dimensions in the last columns as they are used for the
            autoregressive predictions.

        Return
        ------
        mixture_y_pred : numpy array
            Predicted mixtures for the passed inputs.
        """
        mixture_y_pred = self._predict_submission(trained_model, X_array)
        return mixture_y_pred

    def _predict_submission(self, trained_model, X_array):
        """Predict submission.

        Returns the predicted mixtures for passed inputs. See the docstring of
        test_submission for more details.
        """
        regressors, order = trained_model
        n_columns = X_array.shape[1]
        X_array = X_array.copy()
        n_regressors = len(regressors)
        decomposition = getattr(
            regressors[0], 'decomposition', 'autoregressive')
        if decomposition not in [None, 'autoregressive', 'independent']:
            raise ValueError('decomposition attribute is not valid')

        # separate the target from the real inputs. the targets
        # are needed for the autoregressive predictions. note that if
        # decomposition is None, the model is evaluated using an
        # autoregressive decomposition.
        y = X_array[:, -len(self.target_column_names):]
        X_array = X_array[:, :-len(self.target_column_names)]

        if decomposition is None:
            # single multi-d Gaussian mixture
            regressor = regressors[0]
            n_targets = len(self.target_column_names)
            X = X_array
            dists = regressor.predict(X)

            weights, types, params = dists

            n_components_curr = len(types)

            try:
                types = [distributions_dict[type_name] for type_name in types]
            except KeyError:
                message = ('One of the type names is not a valid Scipy '
                           'distribution')
                raise AssertionError(message)
            types = np.array([types, ] * len(weights))

            assert n_components_curr <= self.max_n_components * y.shape[1]
            n_components_per_dim = n_components_curr // n_targets

            # We convert the multi-d Gaussian mixture into its chain rule
            # decomposition so that we can use the same evaluation
            # implementation as for the autoregressive mode. This is done
            # thanks to known formula for Gaussian mixture, see
            # https://stats.stackexchange.com/q/348941. As only diagonal
            # covariance matrices are considered for the components of the
            # multi-d Gaussian mixture, most of the work consists in computing
            # the weights of each 1d conditional Gaussian mixture,
            if not types.any() and not np.all(weights == 1):
                mus = params[:, 0::2]
                sigmas = params[:, 1::2]
                l_probs = np.empty(
                    (len(types), n_components_per_dim, n_targets))
                weights_s = weights[:, :n_components_per_dim]

                # We get the logpdf of every component for every point
                for i in range(n_targets):
                    for j in range(n_components_per_dim):
                        l_probs[:, j, i] = norm.logpdf(
                            y[:, i], mus[:, i], sigmas[:, i])

                # We use a mask to do autoregression
                p_excluded = np.empty_like(l_probs)
                for i in range(n_targets):
                    mask = np.ones(n_targets, dtype=bool)
                    mask[i:] = 0
                    p_excluded[:, :, i] = l_probs[:, :, mask].sum(axis=2)

                # We finish computing the different parts of the formula,
                # and convert from log space
                diffs = np.empty_like(p_excluded)
                for i in range(n_components_per_dim):
                    inner_diff = []
                    for j in range(n_components_per_dim):
                        inner_diff.append(
                            weights_s[:, j:j + 1, np.newaxis] *
                            np.exp(p_excluded[:, i:i + 1, :] -
                                   p_excluded[:, j:j + 1, :]))
                    diffs[:, i:i + 1, :] = np.array(inner_diff).sum(axis=0)

                final_w = weights_s[:, :, np.newaxis] / diffs

                # Finally, we put the weights in place
                weights = final_w.reshape(len(types), -1, order='F')

            # We assume that every dimension is predicted with the same dists
            sizes = np.full((len(types), n_targets), n_components_per_dim)
            size_concatenated = (
                weights.shape[1] + n_components_curr + params.shape[1])
            step = (size_concatenated + n_targets) // n_targets
            mixture_y_pred = np.empty(
                (len(types), n_targets + size_concatenated))

            mixture_y_pred[:, 0::step] = sizes

            offset = 1
            for i in range(offset, n_components_per_dim + offset):
                mixture_y_pred[:, i::step] = \
                    weights[:, i - offset::n_components_per_dim]

            offset += n_components_per_dim
            for i in range(offset, n_components_per_dim + offset):
                mixture_y_pred[:, i::step] = \
                    types[:, i - offset::n_components_per_dim]

            offset += n_components_per_dim
            for i in range(offset, params.shape[1] // n_targets + offset):
                mixture_y_pred[:, i::step] = \
                    params[:, i - offset::params.shape[1] // n_targets]

        else:  # autoregressive or independent decomposition.
            if decomposition == 'autoregressive':
                X_array = np.hstack([X_array, y[:, order]])
            else:
                X = X_array

            mixture = MixtureYPred()
            for i, reg in enumerate(regressors):

                if decomposition == 'autoregressive':
                    X = X_array[:, :n_columns - n_regressors + i]

                dists = reg.predict(X)

                weights, types, params = dists
                n_components_curr = len(types)
                try:
                    types = [
                        distributions_dict[type_name] for type_name in types]
                except KeyError:
                    message = ('One of the type names is not a valid Scipy '
                               'distribution')
                    raise AssertionError(message)
                types = np.array([types, ] * len(weights))
                assert n_components_curr <= self.max_n_components
                mixture.add(weights, types, params)

            mixture_y_pred = mixture.finalize(order)

        return mixture_y_pred

    def get_dist(self, trained_model):
        """Return the distributions predicted for the last passed inputs.

        Parameters
        ----------
        trained_model : tuple
            The model returned by train_submission.

        Returns
        -------
        y_pred_mixture : numpy array
            Predicted distributions of the targets.
        """
        regressors, order = trained_model
        decomposition = getattr(
            regressors[0], 'decomposition', 'autoregressive')
        if decomposition not in [None, 'autoregressive', 'independent']:
            raise ValueError('decomposition attribute is not valid')

        if not hasattr(regressors[0], 'distributions'):
            raise AttributeError(
                'To access uncertainty of the model you need to implement'
                'distributions attribute in your generative_regressor class')

        if decomposition is None:
            dists = regressors[0].distributions
            n_components_curr = len(dists[1])

            weights, types, params = self._format_distribution_types(dists)

            sizes = np.full((len(types), 1), n_components_curr)
            result = np.concatenate(
                (sizes, weights, types, params), axis=1)
            mixture_y_pred = result
        else:
            mixture = MixtureYPred()
            for _, reg in enumerate(regressors):
                dists = reg.distributions

                weights, types, params = self._format_distribution_types(
                    dists)
                mixture.add(weights, types, params)
            mixture_y_pred = mixture.finalize(order)
        return mixture_y_pred

    def step(self, trained_model, X_array, random_state=None):
        """Sample the targets for the samples in X_array.

        Parameters
        ----------
        trained_model : list
            The list of trained models returned by train_submission.

        X_array : numpy array, shape (n_samples, n_features)
            Inputs. Note that compared to test_submission
            the targets are not in this array as this is what we want to
            sample.

        random_state : Random state object
            The RNG or the state of the RNG to be used when sampling.

        Returns
        -------
        y_sampled : numpy array, shape (n_samples, n_targets)
            The sampled targets.
        """

        rng = check_random_state(random_state)
        regressors, order = trained_model
        decomposition = getattr(
            regressors[0], 'decomposition', 'autoregressive')

        n_samples = X_array.shape[0]

        if decomposition is None:
            reg = regressors[0]

            y_sampled = reg.sample(X_array, rng=rng)
        else:  # autoregressive or independent decomposition.
            n_features_init = X_array.shape[1]

            if decomposition == 'autoregressive':
                # preallocate array by concatenating with unknown predicted
                # array
                predicted_array = np.full(
                    (n_samples, len(regressors)), fill_value=np.nan)
                X = np.concatenate(
                    [X_array, predicted_array], axis=1)
                X_used = X[:, :n_features_init]
            else:
                X_used = X_array

            y_sampled = np.zeros((n_samples, len(regressors)))
            for j, reg in enumerate(regressors):
                if j >= 1 and decomposition == 'autoregressive':
                    X[:, n_features_init + (j - 1)] = y_sampled[:, j - 1]
                    X_used = X[:, :n_features_init + j]

                samples = reg.sample(X_used, rng=rng)
                y_sampled[:, j] = samples

            y_sampled = y_sampled[:, np.argsort(order)]
        return y_sampled

    def _format_distribution_types(self, distributions):
        """
        Convert types to an array of types.

        Parameters
        ----------
        distributions:
            Tuples of weights, types, params

        Returns
        -------
        weights, types, params:
            The mixture distribution parameters.
        """
        weights, types, params = distributions

        n_components_curr = len(types)
        try:
            types = [
                distributions_dict[type_name] for type_name in types]
        except KeyError:
            message = ('One of the type names is not a valid Scipy '
                       'distribution')
            raise AssertionError(message)
        types = np.array([types, ] * len(weights))
        assert n_components_curr <= self.max_n_components

        return weights, types, params
