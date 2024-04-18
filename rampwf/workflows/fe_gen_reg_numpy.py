# Author: Jonas Gonzalez <johnasgonzalez@gmail.com>
# License: BSD 3 clause
import os

import numpy as np

from .feature_extractor_numpy import FeatureExtractorNumpy
from .generative_regressor_numpy import GenerativeRegressorNumpy
from .base_workflow import BaseWorkflow


class FEGenRegNumpy(BaseWorkflow):
    """
    Feature extractor + Generative regressor workflow with numpy backend.

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

    def __init__(self, check_sizes, check_indexs,
                 target_column_observation_names,
                 target_column_action_names,
                 max_n_components,
                 restart_name,
                 workflow_element_names=None,
                 ):

        self.max_n_components = max_n_components

        self.target_column_observation_names = target_column_observation_names
        self.target_column_action_names = target_column_action_names
        self.restart_name = restart_name

        if workflow_element_names is None:
            workflow_element_names = ['feature_extractor',
                                      'generative_regressor']
        self.element_names = workflow_element_names

        self.feature_extractor_workflow = FeatureExtractorNumpy()

        self.regressor_workflow = GenerativeRegressorNumpy(
            target_column_observation_names, self.max_n_components,
            workflow_element_names=[self.element_names[1]],
            restart_name=restart_name,
            check_sizes=check_sizes, check_indexs=check_indexs)

    def train_submission(self, module_path, X_array, y_array, train_is=None):

        if train_is is None:
            train_is = slice(None, None, None)

        # pass only the inputs without the targets to the feature extractor
        X_array_used = X_array[:, :-len(self.target_column_observation_names)]
        fe = self.feature_extractor_workflow.train_submission(
            module_path, X_array_used, y_array, train_is)
        X_tf_used_array = self.feature_extractor_workflow.test_submission(
            fe, X_array_used[train_is])
        reg = self.regressor_workflow.train_submission(
            module_path, X_tf_used_array, y_array[train_is])

        return fe, reg

    def test_submission(self, trained_model, X_array):
        fe, reg = trained_model

        # pass only the inputs without the targets to the feature extractor
        X_array_used = X_array[:, :-len(self.target_column_observation_names)]
        X_tf_used_array = fe.transform(X_array_used)

        # append the targets as they are needed for the generative regressor
        y = X_array[:, -len(self.target_column_observation_names):]
        X_tf_array = np.concatenate([X_tf_used_array, y], axis=1)
        y_pred_mixture = self.regressor_workflow.test_submission(reg, X_tf_array)
        return y_pred_mixture

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
        fe, reg = trained_model

        y_pred_mixture = self.regressor_workflow.get_dist(reg)

        n_dists = y_pred_mixture[0, 0]
        if n_dists > self.max_n_components:
            raise ValueError(
                'The maximum number of distributions allowed is '
                f'{self.max_n_components} but you use {n_dists}.')

        return y_pred_mixture

    def step(self, trained_model, X_array, random_state=None):
        """Sample observations.

        Sample from the predicted distribution obtained for X_array.

        Parameters
        ----------
        trained_model : tuple
            Trained model returned by the train_submission method.

        X_array : numpy array
            Inputs. Note that compared to test_submission
            the targets are not in this array as this is what we want to
            sample.

            Each sample of X_df is assumed to contain one observation and one
            action, the action being the one selected after the observation.

        random_state : int, RandomState instance or None, default=None
            If int, random_state is the seed used by the random number
            generator; If RandomState instance, random_state is the random
            number generator; If None, the random number generator is the
            RandomState instance used by np.random.

        Return
        ------
        sampled : numpy array, shape (n_samples, n_targets)
            The next observations.
        """
        fe, reg = trained_model

        X_array = self.feature_extractor_workflow.test_submission(
            fe, X_array)

        sampled = self.regressor_workflow.step(reg, X_array, random_state)

        return sampled
