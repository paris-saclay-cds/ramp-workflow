# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np

from rampwf.utils.cli.testing import get_submissions
from rampwf.utils.cli.show import _bagged_table_and_headers
from rampwf.utils.cli.show import _mean_table_and_headers
from rampwf.utils.cli.show import _load_score_submission
PATH = os.path.dirname(__file__)


def test_get_submissions(monkeypatch):
    iris_kit_path = os.path.join(
        PATH, '..', '..', '..', 'tests', 'kits', 'iris')
    monkeypatch.chdir(iris_kit_path)

    assert ['starting_kit'] == get_submissions(None, None, 'star')
    submissions = get_submissions(None, None, '')
    assert ['starting_kit', 'random_forest_10_10'] == submissions


def test_bagged_table_and_headers():
    path_submissions = os.path.join(
        PATH, '..', '..', '..', 'tests', 'kits', 'iris',
        'submissions')
    all_submissions = {
        sub: os.path.join(path_submissions, sub)
        for sub in os.listdir(path_submissions)
        if os.path.isdir(os.path.join(path_submissions, sub))
    }
    df1, headers1 = _bagged_table_and_headers(all_submissions)

    subs = []
    valid_scores = []
    test_scores = []
    for sub, path in all_submissions.items():
        bagged_scores_path = os.path.join(
            path, 'training_output', 'bagged_scores.csv')
        if not os.path.isfile(bagged_scores_path):
            continue
        bagged_scores_df = pd.read_csv(bagged_scores_path)
        n_folds = len(bagged_scores_df) // 2
        subs.append(sub)
        valid_scores.append(bagged_scores_df.iloc[n_folds - 1, 2])
        test_scores.append(bagged_scores_df.iloc[2 * n_folds - 1, 2])
        metric = bagged_scores_df.columns[2]
    df = pd.DataFrame()
    df['submission'] = subs
    df['valid {}'.format(metric)] = valid_scores
    df['test {}'.format(metric)] = test_scores
    headers = df.columns.to_numpy()

    pd.testing.assert_frame_equal(df1, df)
    np.testing.assert_array_equal(headers1, headers)


def test_mean_table_and_headers():
    path_submissions = os.path.join(
        PATH, '..', '..', '..', 'tests', 'kits', 'iris',
        'submissions')
    all_submissions = {
        sub: os.path.join(path_submissions, sub)
        for sub in os.listdir(path_submissions)
        if os.path.isdir(os.path.join(path_submissions, sub))
    }
    metric = []
    step = []
    df1, headers1 = _mean_table_and_headers(
        all_submissions, metric, step)

    data = {}
    for sub_name, sub_path in all_submissions.items():
        scores = _load_score_submission(sub_path, metric, step)
        if scores is None:
            continue
        data[sub_name] = scores
    df = pd.concat(data, names=['submission'])
    df = df.unstack(level=['step'])
    df = pd.concat([df.groupby('submission').mean(),
                    df.groupby('submission').std()],
                   keys=['mean', 'std'], axis=1, names=['stat'])
    df = df.reorder_levels([1, 2, 0], axis=1)
    step = ['train', 'valid', 'test'] if not step else step
    df = (df.sort_index(axis=1, level=0)
            .reindex(labels=step, level='step', axis=1))
    headers = (["\n".join(df.columns.names)] +
               ["\n".join(col_names)
                for col_names in df.columns.to_numpy()])

    pd.testing.assert_frame_equal(df1, df)
    np.testing.assert_array_equal(headers1, headers)
