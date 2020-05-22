# -*- coding: utf-8 -*-
"""The :mod:`rampwf.utils.command_line` submodule provide command line
utils.
"""
from __future__ import print_function
from __future__ import unicode_literals
import re
import sys
import os
from os import listdir
from os.path import join, isdir
from collections import defaultdict
import numpy as np
import pandas as pd

from .testing import (
    assert_submission, assert_notebook, convert_notebook, blend_submissions)


def create_ramp_test_submission_parser():
    import argparse
    parser = argparse.ArgumentParser(
        prog='ramp_test_submission',
        description='Test your ramp-kit before attempting a submission.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--ramp_kit_dir',
                        default='.',
                        type=str,
                        help='Root directory of the ramp-kit to test.')
    parser.add_argument('--ramp_data_dir',
                        default='.',
                        type=str,
                        help='Directory containing the data. This directory'
                        ' should contain a "data" folder.')
    parser.add_argument('--ramp_submission_dir',
                        default='submissions',
                        type=str,
                        help='Directory where the submissions are stored. It '
                        'should contain a "submissions" directory.')
    parser.add_argument('--submission',
                        default='starting_kit',
                        type=str,
                        help='The kit to test. It should be located in the'
                        ' "submissions" folder of the starting kit. If "ALL",'
                        ' all submissions in the directory will be tested.')
    parser.add_argument('--quick-test', dest='quick_test', action='store_true',
                        help='Specify this flag to test the submission on a '
                             'small subset of the data.'
                        )
    parser.add_argument('--pickle', dest='pickle', action='store_true',
                        help='Specify this flag to pickle the submission '
                             'after training.')
    parser.add_argument('--save-output', dest='save_output',
                        action='store_true',
                        help='Specify this flag to save predictions, scores, '
                             'eventual error trace, and state after training.')
    parser.add_argument('--save-y-preds', dest='save_output',
                        action='store_true',
                        help='Specify this flag to save predictions, scores, '
                             'eventual error trace, and state after training. '
                             'Deprecated.')
    parser.add_argument('--retrain', dest='retrain',
                        action='store_true',
                        help='Specify this flag to retrain the submission '
                             'on the full training set after the CV loop.')
    return parser


def ramp_test_submission():
    parser = create_ramp_test_submission_parser()
    args = parser.parse_args()

    if args.quick_test:
        import os
        os.environ['RAMP_TEST_MODE'] = '1'

    is_pickle = False
    if args.pickle:
        is_pickle = True

    save_output = False
    if args.save_output:
        save_output = True

    retrain = False
    if args.retrain:
        retrain = True

    if args.submission == "ALL":
        ramp_submission_dir = join(args.ramp_kit_dir, 'submissions')
        submission = [directory
                      for directory in listdir(ramp_submission_dir)
                      if isdir(join(ramp_submission_dir, directory))]
    else:
        submission = [args.submission]

    for sub in submission:
        assert_submission(ramp_kit_dir=args.ramp_kit_dir,
                          ramp_data_dir=args.ramp_data_dir,
                          ramp_submission_dir=args.ramp_submission_dir,
                          submission=sub,
                          is_pickle=is_pickle,
                          save_output=save_output,
                          retrain=retrain)


def create_ramp_test_notebook_parser():
    import argparse
    parser = argparse.ArgumentParser(
        prog='ramp_test_notebook',
        description='Test your notebook before submitting a ramp.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--ramp_kit_dir',
                        default='.',
                        type=str,
                        help='Root directory of the ramp-kit to test.')
    return parser


def ramp_test_notebook():
    parser = create_ramp_test_notebook_parser()
    args = parser.parse_args()
    assert_notebook(ramp_kit_dir=args.ramp_kit_dir)


def ramp_convert_notebook():
    parser = create_ramp_test_notebook_parser()
    args = parser.parse_args()
    convert_notebook(ramp_kit_dir=args.ramp_kit_dir)


def create_ramp_blend_submissions_parser():
    import argparse
    parser = argparse.ArgumentParser(
        prog='ramp_blend_submissions',
        description='Blend several submissions.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--ramp_kit_dir',
                        default='.',
                        type=str,
                        help='Root directory of the ramp-kit to test.')
    parser.add_argument('--ramp_data_dir',
                        default='.',
                        type=str,
                        help='Directory containing the data. This directory'
                        ' should contain a "data" folder.')
    parser.add_argument('--ramp_submission_dir',
                        default='submissions',
                        type=str,
                        help='Directory where the submissions are stored. It '
                        'should contain a "submissions" directory.')
    parser.add_argument('--submissions',
                        default='ALL',
                        type=str,
                        help='The submissions to blend. They should be located'
                        ' in the "submissions" folder of the starting kit.'
                        ' Specify submissions separated by a comma without'
                        ' spaces. If "ALL", all submissions in the directory'
                        ' will be blended.')
    parser.add_argument('--save-output', dest='save_output',
                        action='store_true',
                        help='Specify this flag to save predictions '
                             'after blending.')
    parser.add_argument('--min-improvement', dest='min_improvement',
                        default='0.0',
                        help='The minimum score improvement when adding.'
                        ' submissions to the ensemble.')
    return parser


def ramp_blend_submissions():
    parser = create_ramp_blend_submissions_parser()
    args = parser.parse_args()

    save_output = False
    if args.save_output:
        save_output = True

    if args.submissions == "ALL":
        ramp_submission_dir = join(args.ramp_kit_dir, 'submissions')
        submissions = [directory
                       for directory in listdir(ramp_submission_dir)
                       if isdir(join(ramp_submission_dir, directory))]
    else:
        submissions = args.submissions.split(',')

    blend_submissions(ramp_kit_dir=args.ramp_kit_dir,
                      ramp_data_dir=args.ramp_data_dir,
                      ramp_submission_dir=args.ramp_submission_dir,
                      submissions=submissions,
                      save_output=save_output,
                      min_improvement=float(args.min_improvement))


def create_ramp_leaderboard_parser():
    import argparse
    parser = argparse.ArgumentParser(
        prog='ramp_leaderboard',
        description='RAMP leaderboard, a simple command line to display'
                    'the leaderboard.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--ramp_kit_dir',
        default='.',
        type=str,
        help='Root directory of the ramp-kit to test.'
    )
    parser.add_argument(
        '--cols',
        default=None,
        type=str,
        help='List of columns (separated by ",") to display. By default'
             'it is "train_metric,valid_metric,test_metric" where metric is'
             ' the first metric according to alphabetical order. '
             'Use --help-cols to know what are the column names. Column names '
             'are of the form "metric_step" where metric could be e.g., "nll",'
             ' or "acc", and step could be "train", "valid", or "test".'
    )
    parser.add_argument(
        '--sort_by',
        default=None,
        type=str,
        help='List of columns (separated by ",") to sort the leaderboard.'
             'By default it is "test_metric,valid_metric,train_metric"'
             'where metric is the first metric according to alphabetical '
             'order if --metric is not provided otherwise the metric '
             'provided by --metric.'
    )
    parser.add_argument(
        '--metric',
        default=None,
        type=str,
        help='Metric to display. Instead of specifying --cols, we can'
             'specify the metric to display. For example, --metric=acc is'
             'equivalent to --cols=train_acc,valid_acc,test_acc.'
    )
    parser.add_argument(
        '--precision',
        default=2,
        type=int,
        help='Precision for rounding'
    )
    parser.add_argument(
        '--asc',
        dest='asc',
        action='store_true',
        help='Sort ascending'
    )
    parser.add_argument(
        '--desc',
        dest='asc',
        action='store_false',
        help='Sort descending'
    )
    parser.add_argument(
        '--help-cols',
        dest='help_cols',
        action='store_true',
        default=False,
        help='get the list of columns'
    )
    parser.add_argument(
        '--help-metrics',
        dest='help_metrics',
        action='store_true',
        default=False,
        help='get the list of metrics'
    )
    return parser


def ramp_leaderboard():
    """
    RAMP leaderboard, a simple command line to display the leaderboard.

    IMPORTANT: order to display correctly the leaderboard
    you need to save your predictions, e.g.,
    using `ramp_test_submission --submission <name> --save-y-preds`

    :param ramp_kit_dir: folder of ramp-kit to use
    :param cols: list of columns (separated by ",") to display. By default
        it is "train_metric,valid_metric,test_metric" where metric is
        the first metric according to alphabetical order. Use --help-cols to
        know what are the column names. Column names are of the form
        "metric_step" where metric could be e.g., "nll", or "acc", and step
        could be "train", "valid", or "test".
    :param sort_by: list of columns (separated by ",") to sort the leaderboard.
        By default it is "test_metric,valid_metric,train_metric"
        where metric is the first metric according to alphabetical order if
        --metric is not provided otherwise the metric provided by --metric.
    :param asc: sort ascending if True, otherwise descending
    :param metric: metric to display. Instead of specifying --cols, we can
        specify the metric to display. For example, --metric=acc is
        equivalent to --cols=train_acc,valid_acc,test_acc.
    :param precision: precision for rounding
    :param help-cols: get the list of columns
    :param help-metrics: get the list of metrics

    Examples:

    ramp_leaderboard --metric=acc

    ramp_leaderboard --cols=train_acc,valid_acc,test_acc

    ramp_leaderboard --cols=train_nll --sort-by=train_nll,train_acc --asc
    """
    parser = create_ramp_leaderboard_parser()
    args = parser.parse_args()

    try:
        scores = _build_scores_dict(args.ramp_kit_dir)
    except (IOError, OSError) as ex:
        print(ex)
        sys.exit(1)

    if len(scores) == 0:
        print('No submissions are available.')
        print('(Please make sure that you train '
              'your submissions using `ramp_test_submission --submission '
              '<name> --save-y-preds` in order to save the predictions)')
        sys.exit(0)

    df = _build_leaderboard_df(scores, precision=args.precision)
    if args.help_cols:
        for col in df.columns:
            print(col)
        sys.exit(0)

    if args.help_metrics:
        metrics = _get_metrics(df)
        for metric in metrics:
            print(metric)
        sys.exit(0)

    if args.cols and args.metric:
        print('--cols and --metric cannot both be provided')
        sys.exit(1)

    cols = args.cols.split(',') if args.cols else None
    sort_by = args.sort_by.split(',') if args.sort_by else None

    try:
        df = _filter_and_sort_leaderboard_df(
            df,
            cols=cols,
            metric=args.metric,
            sort_by=sort_by,
            asc=args.asc
        )
    except ValueError as ex:
        print(ex)
        sys.exit(1)
    try:
        from ..externals.tabulate import tabulate
        print(tabulate(df, headers='keys', tablefmt='grid'))
    except ImportError:
        print(df)


def _filter_and_sort_leaderboard_df(
        df, cols=None, metric=None,
        sort_by=None, asc=False):
    """
    Filter and sorts rows of df.

    df is a DataFrame obtained using _build_leaderboard_df.

    Parameters
    ----------

    cols : None or list of str
        columns to take from df.
        if a column does not exist in df, the function
        will raise a ValueError.

    metric : None or str
        metric to take from df. Equivalent to
        specifying cols to ['train_metric', 'valid_metric', 'test_metric'].
        If the specified metric does not exist, the function will raise a
        ValueError.
        Note that we cannot both provide cols and metric, only one of them has
        to be provided otherwise the functions returns None.

    sort_by : None or list of str
        columns to sort by. That is, sort by the col of the first
        element sort_by, then the second element, etc.
        If one of the columns do not exist, the function
        will raise a ValuError

    asc : bool
        True if sorting ascending otherwise descending

    Returns
    -------

    pd.DataFrame : df with filtering and sorting

    Raises
    ------

    ValueError if one of the `cols` do not exist in `df`, or
    `metric` do not exist in `df`, or one of the columns in
    `sort_by` do not exist in `df`.
    """
    if cols and metric:
        return
    elif cols is not None:
        valid_cols = set(df.columns.tolist())
        for col in cols:
            if col not in valid_cols:
                cols_s = '\n'.join(list(valid_cols))
                raise ValueError(
                    'Column "{}" does not exist.'
                    ' Available columns are : \n{}'.format(col, cols_s))
        show_cols = cols
    elif metric is not None:
        if 'train_' + metric not in df.columns:
            metrics = '\n'.join(_get_metrics(df))
            raise ValueError(
                'Metric "{}" does not exist.'
                ' Available metrics are : \n{}'.format(metric, metrics))
        show_cols = [
            '{}_{}'.format(step, metric)
            for step in ('train', 'valid', 'test')]
    else:
        metrics = _get_metrics(df)
        metrics = sorted(metrics)
        metric = metrics[0]
        show_cols = [
            '{}_{}'.format(step, metric)
            for step in ('train', 'valid', 'test')]
    if sort_by is not None:
        sort_cols = sort_by
        valid_cols = set(df.columns)
        for col in sort_cols:
            if col not in valid_cols:
                cols_s = '\n'.join(list(valid_cols))
                raise ValueError(
                    'Column "{}" does not exist.'
                    ' Available columns are :\n{}'.format(col, cols_s))
    elif metric:
        sort_cols = ['{}_{}_mean'.format(step, metric)
                     for step in ('test', 'valid', 'train')]
    else:
        metrics = _get_metrics(df)
        metrics = sorted(metrics)
        metric = metrics[0]
        sort_cols = ['{}_{}'.format(step, metric)
                     for step in ('test', 'valid', 'train')]
    df = df.sort_values(by=sort_cols, ascending=asc)
    df = df[['submission'] + show_cols]
    return df


def _build_leaderboard_df(scores_dict, precision=2):
    """
    Get a pd.DataFrame representing the leaderboard.

    Each row is a submission and each column
    is the mean or std of score on train or valid or test data for a metric.
    There is a column name 'submission' for the submission names.
    The rest of column names are prefixed by the 'train' or 'valid' or 'test'
    and followed by the metric and optionally followed by '_mean' or '_std'.
    Example of column names are: 'train_acc', 'valid_acc', 'train_acc_mean',
    'train_acc_std'. 'train_acc' will be a string of the form 'mean ± std'
    whereas 'train_acc_mean' and 'train_acc_std' will be floats.

    Parameters
    ----------

    scores_dict : dict
        scores dictionary returned by _build_scores_dict

    Returns
    -------

    pd.DataFrame


    """
    rows = []
    if len(scores_dict) == 0:
        return pd.DataFrame()

    for submission_name, scores_folds in scores_dict.items():
        scs = [pd.DataFrame(s) for s in scores_folds.values()]
        # compute mean and std over folds for current submission
        mean_scores = sum([s for s in scs]) / len(scs)
        metrics = mean_scores.columns
        std_scores = np.sqrt(
            sum([s**2 for s in scs]) / len(scs) -
            mean_scores ** 2
        )
        # create row for the current submission
        row = {}
        row['submission'] = submission_name
        for step in mean_scores.index.values:
            for metric in mean_scores.columns:
                colname = '{}_{}'.format(step, metric)

                mu = mean_scores.loc[step, metric]
                fmt = '{:.' + str(precision) + 'f}'
                mu = fmt.format(mu)

                fmt = '{:.' + str(precision + 1) + 'f}'
                std = std_scores.loc[step, metric]
                std = fmt.format(std)
                row[colname] = '{} ± {}'.format(mu, std)
                row[colname + '_mean'] = mu
                row[colname + '_std'] = std
        rows.append(row)
    columns_order = [
        '{}_{}'.format(step, metric)
        for metric in metrics
        for step in ('train', 'valid', 'test')
    ]
    columns_order += [
        '{}_{}_{}'.format(step, metric, part)
        for metric in metrics
        for step in ('train', 'valid', 'test')
        for part in ('mean', 'std')
    ]
    return pd.DataFrame(rows, columns=['submission'] + columns_order)


def _get_metrics(leaderboard_df):
    """
    Get the list of metrics used in `leaderboard_df`.

    Parameters
    ----------

    leaderboard_df : pd.DataFrame
        data frame returned by _build_leaderboard_df

    Returns
    -------

    list of str

    Example of return value : ['acc', 'nll']
    """
    metrics = [
        re.search('train_(.+)_mean', col).group(1)
        for col in leaderboard_df.columns
        if col.startswith('train_') and col.endswith('_mean')
    ]
    metrics = list(set(metrics))
    return metrics


def _build_scores_dict(ramp_kit_dir='.'):
    """
    Build a nested dictionary of scores.

    Using the training_output/ folder
    of each submission.

    The structure of the folder `ramp_kit_dir` should be
    something like this:

    submissions/submission1/training_output/fold_0/scores.csv
    submissions/submission1/training_output/fold_1/scores.csv
    submissions/submission2/training_output/fold_0/scores.csv
    submissions/submission2/training_output/fold_1/scores.csv

    The scores.csv files should be like the following.
    There is one required column, "step" which specifies the step,
    it can be either 'train' or 'valid' or 'test'.
    The other columns are the metrics and are problem dependent.
    Example of a scores.csv file:

    step,acc,error,f1_70,nll
    train,0.57,0.42,0.33,1.17
    test,0.70,0.29,0.66,0.80
    valid,0.65,0.35,0.33,0.52

    The structure of the returned dict corresponding to the above
    folder structure will be like this:

    {
        'submission1': {
            0: {
                'acc': {'train': ..., 'valid': ..., 'test': ...},
                'nll': {'train': ..., 'valid': ..., 'test': ...},
            },
            1: {
                'acc': {'train': ..., 'valid': ..., 'test': ...},
                'nll': {'train': ..., 'valid': ..., 'test': ...}
            },
        },
        'submission2': {
            0: {
                'acc': {'train': ..., 'valid': ..., 'test': ...},
                'nll': {'train': ..., 'valid': ..., 'test': ...},
            },
            1: {
                'acc': {'train': ..., 'valid': ..., 'test': ...},
                'nll': {'train': ..., 'valid': ..., 'test': ...}
            },
        }
    }
    Here 0 and 1 are fold numbers.
    'acc' and 'nll' are metrics.
    """
    submissions_folder = os.path.join(ramp_kit_dir, 'submissions')
    scores = defaultdict(dict)
    for submission_name in os.listdir(submissions_folder):
        submission_folder = os.path.join(submissions_folder, submission_name)
        training_output = os.path.join(submission_folder, 'training_output')
        if not os.path.exists(training_output):
            continue
        for fold_name in os.listdir(training_output):
            if not fold_name.startswith('fold_'):
                continue
            _, fold_number = fold_name.split('_')
            fold_number = int(fold_number)
            scores_file = os.path.join(
                training_output, fold_name, 'scores.csv')
            if not os.path.exists(scores_file):
                continue
            fold_scores = pd.read_csv(scores_file)
            fold_scores = fold_scores.set_index('step').to_dict()
            scores[submission_name][fold_number] = fold_scores

    if len(scores) == 0:
        return scores
    # check consistency (all the submissions must have the same structure)
    # -- consistency of nb of folds
    nb_folds_per_submission = [len(folds) for folds in scores.values()]
    assert len(set(nb_folds_per_submission)) == 1
    # -- consistency of metric names
    metrics_per_submission = [
        frozenset(metrics.keys())
        for folds in scores.values()
        for metrics in folds.values()
    ]
    assert len(set(metrics_per_submission)) == 1

    # check correctness of scores dict of each fold of each sumbission
    # -- we should have 3 steps per fold, 'train', 'valid', 'test'
    assert all(
        set(metric.keys()) == set(('train', 'valid', 'test'))
        for folds in scores.values()
        for metrics in folds.values()
        for metric in metrics.values()
    )
    return scores
