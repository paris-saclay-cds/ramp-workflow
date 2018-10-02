# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import os
from tempfile import mkdtemp
from shutil import rmtree
import pytest

import pandas as pd

from rampwf.utils.command_line import create_ramp_test_submission_parser
from rampwf.utils.command_line import create_ramp_test_notebook_parser
from rampwf.utils.command_line import _get_metrics
from rampwf.utils.command_line import _build_leaderboard_df
from rampwf.utils.command_line import _filter_and_sort_leaderboard_df
from rampwf.utils.command_line import _build_scores_dict


def test_cmd_ramp_test_submission_parser():

    # defaults
    parser = create_ramp_test_submission_parser()
    args = parser.parse_args([])
    assert args.ramp_kit_dir == '.'
    assert args.ramp_data_dir == '.'
    assert args.submission == 'starting_kit'

    # specifying keyword args
    parser = create_ramp_test_submission_parser()
    args = parser.parse_args([
        '--ramp_kit_dir', './titanic/', '--ramp_data_dir', './titanic/',
        '--submission', 'other'])
    assert args.ramp_kit_dir == './titanic/'
    assert args.ramp_data_dir == './titanic/'
    assert args.submission == 'other'


def test_cmd_ramp_test_notebook_parser():

    # defaults
    parser = create_ramp_test_notebook_parser()
    args = parser.parse_args([])
    assert args.ramp_kit_dir == '.'

    # specifying keyword args
    parser = create_ramp_test_notebook_parser()
    args = parser.parse_args(['--ramp_kit_dir', './titanic/'])
    assert args.ramp_kit_dir == './titanic/'


def test_cmd_ramp_leaderboard_build_scores_dict():
    with TemporaryDirectory() as tmpdirname:
        # case where 'submissions' folder does not exit
        with pytest.raises(OSError):
            _build_scores_dict(tmpdirname)

        # case where 'submissions' folder exists but no scores
        # are available

        os.mkdir(os.path.join(tmpdirname, 'submissions'))
        scores = _build_scores_dict(tmpdirname)
        assert len(scores) == 0
        training_output = os.path.join(
            tmpdirname, 'submissions', '{sub}', 'training_output')
        os.makedirs(training_output.format(sub='s1'))
        os.makedirs(training_output.format(sub='s2'))
        scores = _build_scores_dict(tmpdirname)
        assert len(scores) == 0

        os.makedirs(os.path.join(training_output, 'fold_0').format(sub='s1'))
        os.makedirs(os.path.join(training_output, 'fold_0').format(sub='s2'))

        scores = _build_scores_dict(tmpdirname)
        assert len(scores) == 0

        scores_content = [
            "step,acc,nll",
            "train,0.57,1.17",
            "valid,0.65,0.52",
            "test,0.70,0.80"
        ]
        f = os.path.join(
            training_output,
            'fold_0', 'scores.csv').format(sub='s1')
        # case where scores.csv file is empty
        with open(f, 'w') as fd:
            pass
        with pytest.raises(pd.errors.EmptyDataError):
            _build_scores_dict(tmpdirname)

        with open(f, 'w') as fd:
            fd.write(scores_content[0] + '\n')

        # case where all steps are missing in the scores.csv file
        with pytest.raises(AssertionError):
            _build_scores_dict(tmpdirname)

        # case where some step is missing in the scores.csv file
        with open(f, 'w') as fd:
            fd.write(scores_content[0] + '\n')
            fd.write(scores_content[1] + '\n')
        with pytest.raises(AssertionError):
            _build_scores_dict(tmpdirname)
        # case where everything is fine in scores.csv file
        with open(f, 'w') as fd:
            fd.write('\n'.join(scores_content))
        scores = _build_scores_dict(tmpdirname)
        assert len(scores) == 1
        assert list(scores.keys()) == ['s1']
        assert scores['s1'][0]['acc'] == {
            'train': 0.57, 'valid': 0.65, 'test': 0.7}
        assert scores['s1'][0]['nll'] == {
            'train': 1.17, 'valid': 0.52, 'test': 0.80}
        # case of two submissions that are ok
        f = os.path.join(
            training_output,
            'fold_0', 'scores.csv').format(sub='s2')
        with open(f, 'w') as fd:
            fd.write('\n'.join(scores_content))
        scores = _build_scores_dict(tmpdirname)
        assert len(scores) == 2
        assert set(scores.keys()) == set(['s1', 's2'])
        assert scores['s1'][0]['acc'] == {
            'train': 0.57, 'valid': 0.65, 'test': 0.7}
        assert scores['s1'][0]['nll'] == {
            'train': 1.17, 'valid': 0.52, 'test': 0.80}
        assert scores['s2'][0]['acc'] == {
            'train': 0.57, 'valid': 0.65, 'test': 0.7}
        assert scores['s2'][0]['nll'] == {
            'train': 1.17, 'valid': 0.52, 'test': 0.80}
        # case of two submissions with inconsistent metrics
        scores1_content = [
            "step,acc,nll",
            "train,0.57,1.17",
            "valid,0.65,0.52",
            "test,0.70,0.80"
        ]
        scores2_content = [
            "step,acc",
            "train,0.57",
            "valid,0.65",
            "test,0.70"
        ]
        f = os.path.join(
            training_output,
            'fold_0', 'scores.csv').format(sub='s1')
        with open(f, 'w') as fd:
            fd.write('\n'.join(scores1_content))
        f = os.path.join(
            training_output,
            'fold_0', 'scores.csv').format(sub='s2')
        with open(f, 'w') as fd:
            fd.write('\n'.join(scores2_content))
        with pytest.raises(AssertionError):
            _build_scores_dict(tmpdirname)
        # case of two submissions with inconsistent nb of folds
        os.makedirs(os.path.join(training_output, 'fold_1').format(sub='s1'))
        f = os.path.join(
            training_output,
            'fold_1', 'scores.csv').format(sub='s1')
        with open(f, 'w') as fd:
            fd.write('\n'.join(scores1_content))

        f = os.path.join(
            training_output,
            'fold_0', 'scores.csv').format(sub='s2')
        with open(f, 'w') as fd:
            fd.write('\n'.join(scores1_content))
        with pytest.raises(AssertionError):
            _build_scores_dict(tmpdirname)
        # case of two submissions, with two folds that are fine
        os.makedirs(os.path.join(training_output, 'fold_1').format(sub='s2'))

        s1f0 = [
            "step,acc,nll",
            "train,0.1,1",
            "valid,0.2,2",
            "test,0.3,3"
        ]
        s1f1 = [
            "step,acc,nll",
            "train,0.4,4",
            "valid,0.5,5",
            "test,0.6,6"
        ]
        s2f0 = [
            "step,acc,nll",
            "train,0.7,7",
            "valid,0.8,8",
            "test,0.9,9"
        ]
        s2f1 = [
            "step,acc,nll",
            "train,0.1,1",
            "valid,0.2,2",
            "test,0.3,3"
        ]
        scs = {'s1': {'fold_0': s1f0, 'fold_1': s1f1},
               's2': {'fold_0': s2f0, 'fold_1': s2f1}}
        for sub in ('s1', 's2'):
            for fold in ('fold_0', 'fold_1'):
                f = os.path.join(training_output, fold, 'scores.csv').format(
                    sub=sub)
                with open(f, 'w') as fd:
                    fd.write('\n'.join(scs[sub][fold]))
        scores = _build_scores_dict(tmpdirname)
        expected_scores = {
            's1': {
                0: {
                    'acc': {'train': 0.1, 'valid': 0.2, 'test': 0.3},
                    'nll': {'train': 1, 'valid': 2, 'test': 3},
                },
                1: {
                    'acc': {'train': 0.4, 'valid': 0.5, 'test': 0.6},
                    'nll': {'train': 4, 'valid': 5, 'test': 6},
                }
            },
            's2': {
                0: {
                    'acc': {'train': 0.7, 'valid': 0.8, 'test': 0.9},
                    'nll': {'train': 7, 'valid': 8, 'test': 9},
                },
                1: {
                    'acc': {'train': 0.1, 'valid': 0.2, 'test': 0.3},
                    'nll': {'train': 1, 'valid': 2, 'test': 3},
                },
            }
        }
        assert scores == expected_scores


class TemporaryDirectory(object):

    def __enter__(self):
        self.tmpdirname = mkdtemp()
        return self.tmpdirname

    def __exit__(self, *exc):
        rmtree(self.tmpdirname)


def test_cmd_ramp_leaderboard_build_leaderboard_df():
    scores_dict = {
        'submission1': {
            0: {
                'acc': {'train': 0.7, 'valid': 0.3, 'test': 0.1},
                'nll': {'train': 1.3, 'valid': 1.4, 'test': 1.5},
            },
            1: {
                'acc': {'train': 0.8, 'valid': 0.5, 'test': 0.4},
                'nll': {'train': 1.2, 'valid': 1.6, 'test': 1.6},
            }
        },
        'submission2': {
            0: {
                'acc': {'train': 0.4, 'valid': 0.7, 'test': 0.2},
                'nll': {'train': 1.1, 'valid': 1.9, 'test': 1.2},
            },
            1: {
                'acc': {'train': 0.3, 'valid': 0.3, 'test': 0.4},
                'nll': {'train': 1.2, 'valid': 1.0, 'test': 1.1},
            },
        }
    }

    leaderboard_df = _build_leaderboard_df(scores_dict, precision=2)
    assert set(leaderboard_df.columns) == set([
        'submission',
        'train_acc', 'valid_acc', 'test_acc',
        'train_acc_mean', 'valid_acc_mean', 'test_acc_mean',
        'train_acc_std', 'valid_acc_std', 'test_acc_std',

        'train_nll', 'valid_nll', 'test_nll',
        'train_nll_mean', 'valid_nll_mean', 'test_nll_mean',
        'train_nll_std', 'valid_nll_std', 'test_nll_std',
    ])
    assert (set(leaderboard_df['submission']) ==
            set(['submission1', 'submission2']))

    d = leaderboard_df.set_index('submission').to_dict(orient='index')
    assert d['submission1']['valid_acc'] == '0.40 ± 0.100'
    assert d['submission1']['test_acc'] == '0.25 ± 0.150'
    assert d['submission1']['train_nll'] == '1.25 ± 0.050'
    assert d['submission1']['valid_nll'] == '1.50 ± 0.100'
    assert d['submission1']['test_nll'] == '1.55 ± 0.050'
    assert d['submission1']['train_acc_mean'] == '0.75'
    assert d['submission1']['train_acc_std'] == '0.050'
    assert d['submission1']['valid_acc_mean'] == '0.40'
    assert d['submission1']['valid_acc_std'] == '0.100'
    assert d['submission1']['test_acc_mean'] == '0.25'
    assert d['submission1']['test_acc_std'] == '0.150'
    assert d['submission1']['train_nll_mean'] == '1.25'
    assert d['submission1']['train_nll_std'] == '0.050'
    assert d['submission1']['valid_nll_mean'] == '1.50'
    assert d['submission1']['valid_nll_std'] == '0.100'
    assert d['submission1']['test_nll_mean'] == '1.55'
    assert d['submission1']['test_nll_std'] == '0.050'
    assert d['submission2']['train_acc'] == '0.35 ± 0.050'
    assert d['submission2']['valid_acc'] == '0.50 ± 0.200'
    assert d['submission2']['test_acc'] == '0.30 ± 0.100'
    assert d['submission2']['train_nll'] == '1.15 ± 0.050'
    assert d['submission2']['valid_nll'] == '1.45 ± 0.450'
    assert d['submission2']['test_nll'] == '1.15 ± 0.050'
    assert d['submission2']['train_acc_mean'] == '0.35'
    assert d['submission2']['train_acc_std'] == '0.050'
    assert d['submission2']['valid_acc_mean'] == '0.50'
    assert d['submission2']['valid_acc_std'] == '0.200'
    assert d['submission2']['test_acc_mean'] == '0.30'
    assert d['submission2']['test_acc_std'] == '0.100'
    assert d['submission2']['train_nll_mean'] == '1.15'
    assert d['submission2']['train_nll_std'] == '0.050'
    assert d['submission2']['valid_nll_mean'] == '1.45'
    assert d['submission2']['valid_nll_std'] == '0.450'
    assert d['submission2']['test_nll_mean'] == '1.15'
    assert d['submission2']['test_nll_std'] == '0.050'


def test_cmd_ramp_leaderboard_filter_sort_leaderboard_df():
    scores_dict = {
        'submission1': {
            0: {
                'acc': {'train': 0.7, 'valid': 0.3, 'test': 0.1},
                'nll': {'train': 1.3, 'valid': 1.4, 'test': 1.5},
            },
            1: {
                'acc': {'train': 0.8, 'valid': 0.5, 'test': 0.4},
                'nll': {'train': 1.2, 'valid': 1.6, 'test': 1.6},
            }
        },
        'submission2': {
            0: {
                'acc': {'train': 0.4, 'valid': 0.7, 'test': 0.2},
                'nll': {'train': 1.1, 'valid': 1.9, 'test': 1.2},
            },
            1: {
                'acc': {'train': 0.3, 'valid': 0.3, 'test': 0.4},
                'nll': {'train': 1.2, 'valid': 1.0, 'test': 1.1},
            },
        }
    }
    df = _build_leaderboard_df(scores_dict)
    # by default use train_metric/valid_metric/test_metric
    # of the metric first in alphabetical order ('acc' here)
    df_ = _filter_and_sort_leaderboard_df(
        df,
        cols=None,
        metric=None,
        sort_by=None)
    assert df_.columns.tolist() == [
        'submission', 'train_acc', 'valid_acc', 'test_acc']
    # non existent column
    with pytest.raises(ValueError):
        df_ = _filter_and_sort_leaderboard_df(
            df,
            cols=['abc'],
            metric=None,
            sort_by=None)
    # no columns
    df_ = _filter_and_sort_leaderboard_df(
        df,
        cols=[],
        metric=None,
        sort_by=None)
    assert df_.columns.tolist() == ['submission']

    # some cols
    df_ = _filter_and_sort_leaderboard_df(
        df,
        cols=['train_nll', 'valid_acc'],
        metric=None,
        sort_by=None)
    assert df_.columns.tolist() == ['submission', 'train_nll', 'valid_acc']
    df_ = _filter_and_sort_leaderboard_df(
        df,
        cols=['valid_acc', 'train_nll'],
        metric=None,
        sort_by=None)
    assert df_.columns.tolist() == ['submission', 'valid_acc', 'train_nll']
    # giving both a metric and cols
    df_ = _filter_and_sort_leaderboard_df(
        df,
        cols=['valid_acc', 'train_nll'],
        metric='acc',
        sort_by=None)
    assert df_ is None

    # non existent metric
    with pytest.raises(ValueError):
        df_ = _filter_and_sort_leaderboard_df(
            df,
            cols=None,
            metric='accc',
            sort_by=None)
    # giving metric
    df_ = _filter_and_sort_leaderboard_df(
        df,
        cols=None,
        metric='nll',
        sort_by=None)
    assert df_.columns.tolist() == [
        'submission', 'train_nll', 'valid_nll', 'test_nll']
    # sorting by non existent col
    with pytest.raises(ValueError):
        df_ = _filter_and_sort_leaderboard_df(
            df,
            cols=None,
            metric=None,
            sort_by=['abc'])
    # sorting by col
    df_ = _filter_and_sort_leaderboard_df(
        df,
        cols=None,
        metric=None,
        sort_by=['test_nll_mean'],
        asc=True)
    assert df_['submission'].tolist() == ['submission2', 'submission1']
    df_ = _filter_and_sort_leaderboard_df(
        df,
        cols=None,
        metric=None,
        sort_by=['test_nll_mean'],
        asc=False)
    assert df_['submission'].tolist() == ['submission1', 'submission2']

    df_ = _filter_and_sort_leaderboard_df(
        df,
        cols=None,
        metric=None,
        sort_by=['train_acc_mean'],
        asc=False)
    assert df_['submission'].tolist() == ['submission1', 'submission2']


def test_cmd_ramp_leaderboard_get_metrics():
    assert _get_metrics(_build_leaderboard_df({})) == []
    scores_dict = {
        'submission1': {
            0: {
                'acc': {'train': 0.7, 'valid': 0.3, 'test': 0.1},
                'nll': {'train': 1.3, 'valid': 1.4, 'test': 1.5},
            },
            1: {
                'acc': {'train': 0.8, 'valid': 0.5, 'test': 0.4},
                'nll': {'train': 1.2, 'valid': 1.6, 'test': 1.6},
            }
        },
        'submission2': {
            0: {
                'acc': {'train': 0.4, 'valid': 0.7, 'test': 0.2},
                'nll': {'train': 1.1, 'valid': 1.9, 'test': 1.2},
            },
            1: {
                'acc': {'train': 0.3, 'valid': 0.3, 'test': 0.4},
                'nll': {'train': 1.2, 'valid': 1.0, 'test': 1.1},
            },
        }
    }
    df = _build_leaderboard_df(scores_dict)
    assert set(_get_metrics(df)) == set(['acc', 'nll'])
