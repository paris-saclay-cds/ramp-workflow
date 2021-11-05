import numpy as np
from numpy.testing import assert_allclose
import pandas as pd

from rampwf.cvs.time_series import KFoldPerEpisode
from rampwf.cvs.time_series import RollingPerEpisode
from rampwf.cvs.time_series import ShufflePerEpisode


def test_get_episode_starts():
    restart_name = 'restart'
    X_data = np.random.randn(10, 2)
    data = np.concatenate(
        (X_data,
         np.array([[1], [0], [0], [0], [1], [0], [0], [1], [0], [0]])), axis=1)
    X_df = pd.DataFrame(
        columns=['X_1', 'X_2', 'restart'], data=data)

    cv = KFoldPerEpisode(restart_name, 0)
    episode_starts = cv._get_episode_starts(X_df)
    assert_allclose(episode_starts, np.array([0, 4, 7]))

    cv = KFoldPerEpisode(restart_name, 2)
    episode_starts = cv._get_episode_starts(X_df)
    assert_allclose(episode_starts, np.array([0, 2, 3]))


def test_per_restart():
    restart_name = 'restart'
    X_data = np.random.randn(10, 2)
    y = np.array([[1], [0], [0], [0], [1], [0], [0], [1], [0], [0]])
    data = np.concatenate(
        (X_data,
         y), axis=1)
    X_df = pd.DataFrame(
        columns=['X_1', 'X_2', 'restart'], data=data)

    cv = KFoldPerEpisode(restart_name, 0)
    # adding the virtual next episode start index
    gen = cv.get_cv(X_df, y)
    train_is, test_is = next(gen)
    assert train_is == list(range(4, 10))
    assert test_is == list(range(4))

    cv = KFoldPerEpisode(restart_name, 2)
    gen = cv.get_cv(X_df, np.arange(3))
    train_is, test_is = next(gen)
    assert train_is == [2]
    assert test_is == list(range(2))

    cv = ShufflePerEpisode(restart_name, random_state=2)
    gen = cv.get_cv(X_df, y)
    train_is, test_is = next(gen)
    assert train_is == [0, 1, 2, 3, 4, 5, 6]
    assert test_is == [7, 8, 9]
    train_is, test_is = next(gen)
    train_is, test_is = next(gen)
    assert train_is == [4, 5, 6, 7, 8, 9]
    assert test_is == [0, 1, 2, 3]

    cv = RollingPerEpisode(restart_name)
    gen = cv.get_cv(X_df, y)
    train_is, test_is = next(gen)
    assert train_is == [0, 1, 2, 3]
    assert test_is == [4, 5, 6]
    train_is, test_is = next(gen)
    assert train_is == [0, 1, 2, 3, 4, 5, 6]
    assert test_is == [7, 8, 9]
