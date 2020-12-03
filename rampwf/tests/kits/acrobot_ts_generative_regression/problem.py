import os
import copy

import pandas as pd

import rampwf as rw

problem_title = 'Acrobot system identification'
_max_components = 100  # max number of mixture components in generative regressors

_target_column_observation_names = [
    'theta_1', 'theta_2', 'theta_dot_1', 'theta_dot_2']
_target_column_action_names = ['torque']
_restart_name = 'is_done'
# number of guaranteed steps in time series history
_n_burn_in = 0
timestamp_name = 'fake_ts'

Predictions = rw.prediction_types.make_generative_regression(
    _max_components, label_names=_target_column_observation_names)

score_types = [
    rw.score_types.MDLikelihoodRatio('lr', precision=2),
    rw.score_types.MDOutlierRate('or', precision=4),
    rw.score_types.MDR2('r2', precision=6),
    rw.score_types.MDKSCalibration('ks', precision=4),
]
# generate scores for each output dimension
_score_types = copy.deepcopy(score_types)
for o_i, o in enumerate(_target_column_observation_names):
    dim_score_types = copy.deepcopy(_score_types)
    for score_type in dim_score_types:
        score_type.name = '{}_{}'.format(o, score_type.name)
        score_type.output_dim = o_i
    score_types += dim_score_types

cv = rw.cvs.PerRestart(restart_name=_restart_name)
get_cv = cv.get_cv

workflow = rw.workflows.TSFEGenReg(
    check_sizes=[137], check_indexs=[13], max_n_components=_max_components,
    target_column_observation_names=_target_column_observation_names,
    target_column_action_names=_target_column_action_names,
    restart_name=_restart_name,
    timestamp_name='time',
    n_burn_in=_n_burn_in,
)


def get_train_data(path='.', data_label=''):
    return _read_data(path, 'X_train.csv', data_label)


def get_test_data(path='.', data_label=''):
    return _read_data(path, 'X_test.csv', data_label)


def _read_data(path, X_name, data_label=''):
    """Reading and preprocessing data.

    Parameters
    ----------
    path : string
        Data directory.

    X_name : string
        Name of the csv data file. This data file contains a sequence of
        observations and action. Each sample/row is assumed to contain one
        observation and one action, the action being the one selected after
        the observation. Each row also contains a flag (the restart column)
        equal to 1 if the sequence has been reset with a new random
        observation, 0 otherwise. Finally each row preceding a restart is
        assumed to contain the last observation of the sequence, the one
        obtained just before the sequence was reset, and a NaN value for
        the associated action.

    data_label : string
        Subfolder in /data where X_name is. Also used for creating
        subdirectories in /submissions/<submission>/training_output
        if --save-output is specified.

    Return
    ------
    X_df : pandas DataFrame
        Preprocessed data. Same format as the original data file but with
        targets appended. Each row thus contains a transition
        (past observation, action, new observation). Indeed, as the chaining
        rule is used, when training/testing the model the feature p - 1 of the
        target is needed to predict feature p of the target.

    y_array : numpy array, shape (n_samples, n_targets)
        Targets. The targets are the observations following the action
        contained in each row of the input data file.
    """

    X_df = pd.read_csv(os.path.join(path, 'data', X_name))
    # rename timestamp_name to time for compatibility with
    # TimeSeriesFeatureExtractor
    X_df = X_df.rename(columns={timestamp_name: 'time'})
    X_df = X_df.set_index('time', drop=True)
    # make sure that we have float for action because of check_ds...
    X_df = X_df.astype({_target_column_action_names[0]: 'float'})
    X_df = X_df.astype({_restart_name: 'int64'})

    # reorder columns according to _target_column_observation_names
    X_df = X_df.reindex(
        columns=_target_column_observation_names +
        _target_column_action_names + [_restart_name])
    # Target for observation
    y_df = X_df[_target_column_observation_names][1:]
    y_df.reset_index(drop=True, inplace=True)

    # We drop the last step of X since we do not have data
    # for a(t) at last timestep
    X_df = X_df.iloc[:-1]
    date = X_df.index.copy()

    # Since in validation we will need to gradually give y to the
    # conditional regressor, we now have to add y in X.
    # The names are made to avoid duplicated names between inputs and targets.
    extra_truth = ['y_' + obs for obs in _target_column_observation_names]
    columns_X = list(X_df.columns)

    y_df_no_name = pd.DataFrame(y_df.values)
    X_df.reset_index(drop=True, inplace=True)
    X_df = pd.concat([X_df, y_df_no_name], axis=1)

    new_names = columns_X + extra_truth
    X_df.set_axis(new_names, axis=1, inplace=True)

    X_df.set_index(date, inplace=True)
    X_df.dropna(how='any', inplace=True)

    return X_df, X_df[extra_truth].values
