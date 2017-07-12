import numpy as np


class TimeSeriesCV:
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
