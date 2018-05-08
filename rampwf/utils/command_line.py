# -*- coding: utf-8 -*-
"""The :mod:`rampwf.utils.command_line` submodule provide command line
utils.
"""
from __future__ import print_function
from __future__ import unicode_literals
import sys
from os import listdir
from os.path import join, isdir
from collections import defaultdict
import os
import numpy as np
import pandas as pd
from tabulate import tabulate

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
    parser.add_argument('--save-y-preds', dest='save_y_preds',
                        action='store_true',
                        help='Specify this flag to save predictions '
                             'after training.')
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

    save_y_preds = False
    if args.save_y_preds:
        save_y_preds = True

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
                          submission=sub,
                          is_pickle=is_pickle,
                          save_y_preds=save_y_preds,
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
    parser.add_argument('--submissions',
                        default='ALL',
                        type=str,
                        help='The submissions to blend. They should be located'
                        ' in the "submissions" folder of the starting kit.'
                        ' Specify submissions separated by a comma without'
                        ' spaces. If "ALL", all submissions in the directory'
                        ' will be blended.')
    parser.add_argument('--save-y-preds', dest='save_y_preds',
                        action='store_true',
                        help='Specify this flag to save predictions '
                             'after training.')
    parser.add_argument('--min-improvement', dest='min_improvement',
                        default='0.0',
                        help='The minimum score improvement when adding.'
                        ' submissions to the ensemble.')
    return parser


def ramp_blend_submissions():
    parser = create_ramp_blend_submissions_parser()
    args = parser.parse_args()

    save_y_preds = False
    if args.save_y_preds:
        save_y_preds = True

    if args.submissions == "ALL":
        ramp_submission_dir = join(args.ramp_kit_dir, 'submissions')
        submissions = [directory
                       for directory in listdir(ramp_submission_dir)
                       if isdir(join(ramp_submission_dir, directory))]
    else:
        submissions = args.submissions.split(',')

    blend_submissions(ramp_kit_dir=args.ramp_kit_dir,
                      ramp_data_dir=args.ramp_data_dir,
                      submissions=submissions,
                      save_y_preds=save_y_preds,
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
    RAMP leaderboard, a simple command line to display
    the leaderboard.
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
        for metric in _get_metrics(df):
            print(metric)
        sys.exit(0)

    if args.cols and args.metric:
        print('--cols and --metric cannot both be provided')
        sys.exit(1)

    if args.cols:
        accepted_cols = set(args.cols.split(','))
        for col in accepted_cols:
            if col not in df.columns:
                print('Column "{}" does not exist.'
                      ' Available columns are : '.format(col))
                for c in df.columns:
                    print(c)
                sys.exit(1)
        cols = df.columns
        show_cols = [c for c in cols if c in accepted_cols]
    elif args.metric:
        if 'train_' + args.metric not in df.columns:
            print('Metric "{}" does not exist.'
                  ' Available metrics are : '.format(args.metric))
            for metric in _get_metrics(df):
                print(metric)
            sys.exit(1)
        show_cols = [
            '{}_{}'.format(step, args.metric)
            for step in ('train', 'valid', 'test')]
    else:
        metrics = _get_metrics(df)
        metrics = sorted(metrics)
        metric = metrics[0]
        show_cols = [
            '{}_{}'.format(step, metric)
            for step in ('train', 'valid', 'test')]
    if args.sort_by:
        sort_cols = args.sort_by.split(',')
        for col in sort_cols:
            if col not in df.columns:
                print('Column "{}" does not exist.'
                      ' Available columns are : '.format(col))
                for c in df.columns:
                    print(c)
                sys.exit(1)
    elif args.metric:
        sort_cols = ['{}_{}'.format(step, args.metric)
                     for step in ('test', 'valid', 'train')]
    else:
        metrics = _get_metrics(df)
        metrics = sorted(metrics)
        metric = metrics[0]
        sort_cols = ['{}_{}'.format(step, metric)
                     for step in ('test', 'valid', 'train')]
    df = df.sort_values(by=sort_cols, ascending=args.asc)
    df = df[['submission'] + show_cols]
    print(tabulate(df, headers='keys', tablefmt='grid'))


def _get_metrics(df):
    """
    Get the list of metrics used in df

    df : pd.DataFrame
        data frame with the scores of each submission
        obtained with _build_leaderboard_df

    Returns
    -------

    list of str

    Example of return value : ['acc', 'nll']
    """
    metrics = [
        c.split('_')[1]
        for c in df.columns if c.startswith('train')
    ]
    metrics = set(metrics)
    metrics = list(metrics)
    return metrics


def _build_leaderboard_df(scores_dict, precision=2):
    """
    Get a pd.DataFrame where each row is a submission and each column
    is the mean or std of score on train or valid or test data for a metric.
    Column names are prefixed by the 'train' or 'valid' or 'test' and
    followed by the metric and optionally followed by '_mean' or '_std'.
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
    for submission_name, scores_folds in scores_dict.items():
        scs = scores_folds.values()
        mean_scores = sum([s for s in scs]) / len(scs)
        std_scores = np.sqrt(
            sum([s**2 for s in scs]) / len(scs) -
            mean_scores ** 2
        )
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
        for metric in mean_scores.columns
        for step in ('train', 'valid', 'test')
    ]
    columns_order += [
        '{}_{}_{}'.format(step, metric, part)
        for metric in mean_scores.columns
        for step in ('train', 'valid', 'test')
        for part in ('mean', 'std')
    ]
    return pd.DataFrame(rows, columns=['submission'] + columns_order)


def _build_scores_dict(ramp_kit_dir='.'):
    """
    Build a nested dictionary of scores using the training_output/ folder
    of each submission.
    The structure of the returned dict can be like this:
    {
        'submission1': {
            0: {'acc': ..., 'nll': ...},
            1: {'acc': ..., 'nll': ...},
        },
        'submission2': {
            0: {'acc': ..., 'nll': ...},
            1: {'acc': ..., 'nll': ...},
        }
    }
    0 and 1 are fold numbers.
    acc and nll are scores.
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
            fold_scores = fold_scores.set_index('step')
            scores[submission_name][fold_number] = fold_scores
    return scores
