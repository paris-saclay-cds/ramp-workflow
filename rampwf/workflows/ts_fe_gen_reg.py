"""A time series feature extractor followed by a generative regressor.

Train and test a time series feature extractor followed by a generative
regressor.

The input object is an `xarray` `Dataset`, containing possibly several
`DataArrays` corresponding to the input sequence. It contains a special burn
in period in the beginning (carried by X_ds.n_burn_in) for which we do not
give ground truth and we do not require the user to provide predictions.
The ground truth sequence `y_array` in train and the output of the user
submission `ts_fe.transform` are thus `n_burn_in` shorter than the input
sequence `X_ds`, making the training and testing slightly complicated.
"""

# Author: Gabriel Hurtado <gabriel.j.hurtado@gmail.com>
# License: BSD 3 clause
import pandas as pd
from .ts_feature_extractor import TimeSeriesFeatureExtractor
from .generative_regressor import GenerativeRegressor


class TSFEGenReg:
    def __init__(self,
                 check_sizes, check_indexs, max_dists,
                 target_column_observation_names,
                 target_column_action_names,
                 restart_name=None,
                 timestamp_name='time',
                 workflow_element_names=None):

        self.max_dists = max_dists
        self.target_column_observation_names = target_column_observation_names
        self.target_column_action_names = target_column_action_names
        self.restart_name = restart_name
        self.timestamp_name = timestamp_name

        # columns used for the feature extractor
        self.cols_for_extractor = (
            self.target_column_observation_names +
            self.target_column_action_names)
        if self.restart_name is not None:
            self.cols_for_extractor += [self.restart_name]

        if workflow_element_names is None:
            workflow_element_names = ['ts_feature_extractor',
                                      'generative_regressor']
        self.element_names = workflow_element_names

        self.feature_extractor_workflow = TimeSeriesFeatureExtractor(
            check_sizes=check_sizes, check_indexs=check_indexs,
            workflow_element_names=[self.element_names[0]],
            restart_name=self.restart_name)

        self.regressor_workflow = GenerativeRegressor(
            target_column_observation_names, self.max_dists,
            workflow_element_names=[self.element_names[1]],
            restart_name=restart_name,
            check_sizes=check_sizes, check_indexs=check_indexs)

    def train_submission(self, module_path, X_df, y_array, train_is=None):
        """Train model.

        The state feature of time t returned by the feature extractor can be
        built using any data from times < t.

        Parameters
        ----------
        module_path : string,
            Path of the model to train.

        X_df : pandas dataframe
            Training data. Each sample contains the data of a given timestep.
            Only the columns target_column_observation_names,
            target_column_action_names and restart_name are used. X_df
            must contain these columns.

        y_array : numpy array, shape (n_samples, n_targets)
            Training targets. Must be a 2D array even if there is only one
            target.

        Returns
        -------
        fe, reg : tuple
            Trained feature extractor and generative regressor.
        """
        X_df_used = X_df[self.cols_for_extractor]
        fe = self.feature_extractor_workflow.train_submission(
            module_path, X_df_used, y_array, train_is)
        if train_is is None:
            train_is = slice(None, None, None)
        X_df_tf = self.feature_extractor_workflow.test_submission(
            fe, X_df_used[{self.timestamp_name: train_is}])

        reg = self.regressor_workflow.train_submission(
            module_path, X_df_tf, y_array[train_is])

        return fe, reg

    def test_submission(self, trained_model, X_df):
        """Predict outputs of X_df with a trained model.

        Parameters
        ----------
        trained_model : tuple
            The model returned by train_submission.

        X_df : pandas dataframe
            Inputs. Each sample contains the data of a given timestep. Note
            that the targets have to be included in this dataframe. This allows
            the prediction of an autoregressive submission which is based on
            the chaining rule: targets <j are needed to predict target j.

        Returns
        -------
        y_pred_mixture : numpy array
            Predicted distributions of the targets.
        """

        fe, reg = trained_model

        # pass only the inputs without the targets to the feature extractor
        X_df_tf = self.feature_extractor_workflow.test_submission(
            fe, X_df[self.cols_for_extractor])

        # append the targets to X_df_tf as they are needed for the generative
        # regressor predictions
        truth_names = ['y_' + obs for obs in
                       self.target_column_observation_names]
        X_test_df = pd.concat(
            [X_df_tf, X_df[truth_names].to_dataframe()], axis=1)
        y_pred_mixture = self.regressor_workflow.test_submission(reg, X_test_df)

        nb_dists = y_pred_mixture[0, 0]
        if nb_dists > self.max_dists:
            raise ValueError(
                'The maximum number of distributions allowed is '
                f'{self.max_dists} but you use {nb_dists}.')

        return y_pred_mixture

    def step(self, trained_model, X_df, random_state=None):
        """Sample next observation.

        Sample one target observation from the predicted distribution obtained
        with the history X_df. Note that this only samples one target
        observation and not one target observation for each timestep of X_df.

        Parameters
        ----------
        trained_model : tuple
            Trained model returned by the train_submission method.

        X_df : pandas dataframe
            History used to sample the target observation.

            Each sample of X_df is assumed to contain one observation and one
            action, the action being the one selected after the observation.
            The action of the last row is the one for which we want to sample
            the next observation.

        random_state : int, RandomState instance or None, default=None
            If int, random_state is the seed used by the random number
            generator; If RandomState instance, random_state is the random
            number generator; If None, the random number generator is the
            RandomState instance used by np.random.

        Return
        ------
        sample_df : pandas dataframe, shape (1, n_targets)
            The next observation.
        """

        fe, reg = trained_model

        X_df_tf = self.feature_extractor_workflow.test_submission(
            fe, X_df[self.cols_for_extractor])

        # depending on the feature extractor, more than one extracted state can
        # be returned in X_df_tf. here we only sample for the next
        # timestep, i.e. from the last state built with the feature extractor.
        # we use [-1] so that a pandas DataFrame is returned and not a Series
        X_df_tf = X_df_tf.iloc[[-1]]

        sampled = self.regressor_workflow.step(reg, X_df_tf, random_state)
        sampled_df = pd.DataFrame(sampled)

        new_names = self.target_column_observation_names
        sampled_df.set_axis(new_names, axis=1, inplace=True)

        return sampled_df
