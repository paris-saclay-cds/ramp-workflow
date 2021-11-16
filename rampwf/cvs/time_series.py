"""A time series cross validation class.

It implements a block cross validation. We can't simply shuffle the
observations z_t since we would lose both causality and the correlation
structure that follows natural order. To formalize the issue, let us first
define formally the predictor that we will produce in the RAMP. Let the time
series be z_1, ..., z_T and the let target to predict at time t be
y_t. The target is usually a function of the future
z_{t+1}, ..., but it can be anything else. We want to learn a function
that predicts y from the past, that is y_hat_t = f(z_1, ..., z_t) = f(Z_t),
where Z_t = (z_1, ..., z_t) is the past. Now, the sample (Z_t, y_t) is a
regular (although none iid) sample from the point of view of shuffling, so we
can train on {Z_t, y_t}_{t in I_train} and test on
(Z_t, y_t)_{t in I_test}, where I_train
and I_test are arbitrary but disjunct train and test index
sets, respectively (typically produced by sklearn's `ShuffleSplit`). Using
shuffling would nevertheless allow a second order leakage from training
points to test points that precede them, by, e.g., aggregating the training
set and adding the aggregate back as a feature. To avoid this, we use
block-CV: on each fold, all t in I_test are larger than all
t in I_train. We also make sure that all training and test
sets contain consecutive observations, so recurrent nets and similar
predictors, which rely on this, may be trained.

The block cv can be parameterized by `cv_block_size`, `n_cv`, and `period`.
`cv_block_size` is the relative size of the validation block. If it is, e.g.,
0.3, it means that all folds have a common block which is (approximately) 0.7
times the length of the sequence. `n_cv` is the number of the folds. `period`
can be used when we want that the length of each training block is a multiple
of an integer (e.g., the number of months in a year), assuring that each block
starts at the same phase (e.g., the beginning of the year).

A classical simple validation (a single training block followed by a test
block) can be done by setting `n_cv` to 1.
"""

# Author: Balazs Kegl <balazs.kegl@gmail.com>
# License: BSD 3 clause
import numpy as np


class TimeSeries(object):
    def __init__(self, n_cv=8, cv_block_size=0.5, period=12, unit='',
                 unit_2=None):
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
                self. unit_2))
            print('length of each cv block: {} {}s = {} {}s'.format(
                block_size, self.unit, block_size / self.period, self.unit_2))
        for i in range(self.n_cv):
            train_is = np.arange(0, n_common_block + i * block_size)
            test_is = np.arange(n_common_block + i * block_size, n)
            yield (train_is, test_is)
