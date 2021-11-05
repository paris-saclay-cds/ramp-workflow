# Author: Balazs Kegl <balazs.kegl@gmail.com>
# License: BSD 3 clause
import numpy as np
from sklearn.model_selection import KFold, ShuffleSplit
from abc import ABCMeta


class TimeSeries(object):
    def __init__(self, n_cv=8, cv_block_size=0.5, period=12, unit='',
                 unit_2=None):
        """A time series cross validation class.

        It implements a block cross validation. We can't simply shuffle the
        observations z_t since we would lose both causality and the correlation
        structure that follows natural order. To formalize the issue, let us
        first define formally the predictor that we will produce in the RAMP.
        Let the time series be z_1, ..., z_T and the let target to predict at
        time t be y_t. The target is usually a function of the future
        z_{t+1}, ..., but it can be anything else. We want to learn a function
        that predicts y from the past, that is
        y_hat_t = f(z_1, ..., z_t) = f(Z_t), where Z_t = (z_1, ..., z_t) is the
        past. Now, the sample (Z_t, y_t) is a regular (although none iid)
        sample from the point of view of shuffling, so we can train on
        {Z_t, y_t}_{t in I_train} and test on (Z_t, y_t)_{t in I_test}, where
        I_train and I_test are arbitrary but disjunct train and test index
        sets, respectively (typically produced by sklearn's `ShuffleSplit`).
        Using shuffling would nevertheless allow a second order leakage from
        training points to test points that preceed them, by, e.g., aggregating
        the training set and adding the aggregate back as a feature. To avoid
        this, we use block-CV: on each fold, all t in I_test are larger than
        all t in I_train. We also make sure that all training and test
        sets contain consecutive observations, so recurrent nets and similar
        predictors, which rely on this, may be trained.

        The block cv can be parameterized by `cv_block_size`, `n_cv`, and
        `period`. `cv_block_size` is the relative size of the validation block.
        If it is, e.g., 0.3, it means that all folds have a common block which
        is (approximately) 0.7 times the length of the sequence. `n_cv` is the
        number of the folds. `period` can be used when we want that the length
        of each training block is a multiple of an integer (e.g., the number of
        months in a year), assuring that each block starts at the same phase
        (e.g., the beginning of the year).

        A classical simple validation (a single training block followed by a
        test block) can be done by setting `n_cv` to 1.
        """

        self.n_cv = n_cv
        self.cv_block_size = cv_block_size
        self.period = period
        self.unit = unit
        self.unit_2 = unit_2

    def get_cv(self, X, y):
        n = len(y)
        block_size = int(
            n * self.cv_block_size / self.n_cv / self.period) * self.period
        n_common_block = n - block_size * self.n_cv
        n_validation = n - n_common_block
        if self.unit_2 is None:
            print('length of common block: {} {}s'.format(
                n_common_block, self.unit))
            print('length of validation block: {} {}s'.format(
                n_validation, self.unit))
            print('length of each cv block: {} {}s'.format(
                block_size, self.unit))
        else:
            print('length of common block: {} {}s = {} {}s'.format(
                n_common_block, self.unit, n_common_block / self.period,
                self.unit_2))
            print('length of validation block: {} {}s = {} {}s'.format(
                n_validation, self.unit, n_validation / self.period,
                self.unit_2))
            print('length of each cv block: {} {}s = {} {}s'.format(
                block_size, self.unit, block_size / self.period, self.unit_2))
        for i in range(self.n_cv):
            train_is = np.arange(0, n_common_block + i * block_size)
            test_is = np.arange(n_common_block + i * block_size, n)
            yield (train_is, test_is)


def fold_to_str(idxs):
    """Printing cross validation folds."""
    idx_array = np.array(idxs)
    boundaries = np.array((idx_array[1:] - 1 != idx_array[:-1]).nonzero())
    boundaries = np.insert(boundaries, 0, -1)
    boundaries = np.append(boundaries, len(idxs) - 1)
    s = ''
    for lb, ub in zip(boundaries[:-1], boundaries[1:]):
        s += '{}..{}, '.format(idx_array[lb + 1], idx_array[ub])
    return s


class InsideEpisode(object):
    """CV inside each of the episodes.

    An episode in a time series is defined by a sequence of consecutive times.
    They are identified by a restart column whose value is equal to 1 at the
    start of each new episode. The term episode comes from the episode of
    a reinforcement learning task.

    A split into a training and test set is done inside each episode and all
    the training sets (respectively the test sets) are concatenated into one
    big training set (respectively test set).

    Using this CV with burn-in is not supported.
    """

    def __init__(self, cv_method=None, restart_name='restart'):
        """cv_method should typically be rw.cvs.TimeSeries().get_cv"""
        self.cv_method = cv_method
        self.restart_name = restart_name
        if self.cv_method is None:
            self.cv_method = KFold(
                n_splits=3, random_state=None, shuffle=False).split

    def get_cv(self, X_df, y):
        episode_starts = list(np.where(X_df[self.restart_name])[0])
        if len(episode_starts) == 0 or episode_starts[0] != 0:
            episode_starts.insert(0, 0)
        print('episode starts: {}'.format(episode_starts))
        n_episodes = len(episode_starts)
        # we add the start index of the virtual next episode to ease
        # the computation of the folds
        episode_starts.append(len(y))
        episode_list = []
        ranges = []
        for episode_id in range(n_episodes + 1):
            episode_range = list(range(
                episode_starts[episode_id], episode_starts[episode_id + 1]))
            ranges.append(np.array(episode_range))
            episode_list.append(self.cv_method(episode_range, episode_range))
        n_cv = len(list(self.cv_method(range(len(X_df)), range(len(X_df)))))

        train_idx = None
        test_idx = None
        for fold_i in range(n_cv):
            train_is = []
            test_is = []
            if fold_i > 0:
                X_df[self.restart_name][train_idx[0]] = 0
                X_df[self.restart_name][test_idx[0]] = 0
            for episode, curr_range in zip(episode_list, ranges):
                train_idx, test_idx = next(episode)
                train_idx = curr_range[train_idx]
                test_idx = curr_range[test_idx]
                train_is += list(train_idx)
                test_is += list(test_idx)
                X_df[self.restart_name][train_idx[0]] = 1
                X_df[self.restart_name][test_idx[0]] = 1
            print('CV fold {}: train {} valid {}'.format(
                fold_i, fold_to_str(train_is), fold_to_str(test_is)))
            yield (train_is, test_is)


class PerEpisode(metaclass=ABCMeta):
    """Abstract CV where folds are defined with the episodes.

    An episode in a time series is defined by a sequence of consecutive times.
    They are identified by a restart column whose value is equal to 1 at the
    start of each new episode. The term episode comes from the episode of
    a reinforcement learning task.

    For each split, some episodes are in the training and some others in the
    test. Children should implement a get_splits(n_episodes) function and
    initialize restart_name and n_burn_in.
    """

    def _get_episode_starts(self, X_df):
        """Episode start indices without burn-in samples.

        List of episode start indices if burn-in samples were first removed.
        This is used to slice the ground truth target array y.

        Parameters
        ----------
        X_df : pandas dataframe
            Contains a restart_name column with values equal to 1 for the
            start of an episode, 0 otherwise.

        Return
        ------
        episode_starts : numpy array
            Episode bound indices
        """
        episode_starts = np.where(X_df[self.restart_name])[0]
        episode_starts = list(episode_starts)
        if len(episode_starts) == 0 or episode_starts[0] != 0:
            episode_starts.insert(0, 0)

        # align starts to take burn in into account
        if self.n_burn_in > 0:
            align = np.arange(0, len(episode_starts)) * self.n_burn_in
            episode_starts[1:] = episode_starts[1:] - align[1:]

        print('episode starts: {}'.format(episode_starts))
        return episode_starts

    def get_cv(self, X_df, y):
        episode_starts = self._get_episode_starts(X_df)
        n_episodes = len(episode_starts)
        splits = self.get_splits(n_episodes)
        # we add the start index of the virtual next episode to ease
        # the computation of the folds
        episode_starts.append(len(y))
        for fold_i, (train_idx, test_idx) in enumerate(splits):
            # we sort so that the X samples extended with the burn in are
            # aligned with the y samples.
            # XXX it would be better to fix this when building
            # extended_train_is in the workflows
            train_idx.sort()
            test_idx.sort()
            train_is = []
            test_is = []
            for i in train_idx:
                train_is += (
                    list(range(episode_starts[i], episode_starts[i + 1])))
            for i in test_idx:
                test_is += (
                    list(range(episode_starts[i], episode_starts[i + 1])))
            print('CV fold {}: train {} valid {}'.format(
                fold_i, fold_to_str(train_is), fold_to_str(test_is)))
            yield (train_is, test_is)


class KFoldPerEpisode(PerEpisode):
    """K-fold CV where folds are defined with the episodes.

    An episode in a time series is defined by a sequence of consecutive times.
    They are identified by a restart column whose value is equal to 1 at the
    start of each new episode. The term episode comes from the episode of
    a reinforcement learning task.

    For each split, one of the episodes is the test and the rest forms the
    training set.

    Parameters
    ----------
    restart_name : string
        Name of the restart column.
    n_burn_in : int
        Number of steps used as burn in.
    """

    def __init__(self, restart_name='restart', n_burn_in=0):
        self.restart_name = restart_name
        self.n_burn_in = n_burn_in

    def get_splits(self, n_episodes):
        k_fold = KFold(n_splits=n_episodes, shuffle=False)
        return k_fold.split(np.arange(n_episodes))


class ShufflePerEpisode(PerEpisode):
    """Shuffle split on the episodes.

    An episode in a time series is defined by a sequence of consecutive times.
    They are identified by a restart column whose value is equal to 1 at the
    start of each new episode. The term episode comes from the episode of
    a reinforcement learning task.

    For each split, ``n_episodes_in_test`` episodes are selected at random to
    form a test set. The rest of the episodes forms the training set.

    Parameters
    ----------
    restart_name : string
        Name of the restart column.
    n_burn_in : int
        Number of steps used as burn in.
    n_splits : int
        Number of splits
    n_episodes_in_test : int
        Number of episodes in test set.
    random_state : object
        Random state used for the shuffle splits.
    """

    def __init__(self, restart_name='restart', n_burn_in=0,
                 n_splits=10, n_episodes_in_test=1,
                 random_state=None):
        self.restart_name = restart_name
        self.n_burn_in = n_burn_in
        self.n_splits = n_splits
        self.n_episodes_in_test = n_episodes_in_test
        self.random_state = random_state

    def get_splits(self, n_episodes):
        shuffle_cv = ShuffleSplit(
            n_splits=self.n_splits, test_size=self.n_episodes_in_test,
            random_state=self.random_state)
        return shuffle_cv.split(np.arange(n_episodes))


class RollingPerEpisode(PerEpisode):
    """Rolling split on the episodes.

    An episode in a time series is defined by a sequence of consecutive times.
    They are identified by a restart column whose value is equal to 1 at the
    start of each new episode. The term episode comes from the episode of
    a reinforcement learning task.

    For split j, the training episodes are 0..j, and the test episode is j+1.

    Parameters
    ----------
    restart_name : string
        Name of the restart column.
    n_burn_in : int
        Number of steps used as burn in.
    """

    def __init__(self, restart_name='restart', n_burn_in=0):
        self.restart_name = restart_name
        self.n_burn_in = n_burn_in

    def get_splits(self, n_episodes):
        return [(np.arange(j), np.array([j])) for j in range(1, n_episodes)]
