"""A growing fold cross validation class.

Concatenation of sets of folds with various train sizes, the same
number of folds each. Together with using fold_idx's, it is a flexible 
way to have a pool of folds and selecting the ones dynamically when the
models are trained and hyperopted. 

It needs to create all folds between the starting and stopping indices,
so it can be inefficient if the limits are set apart and the training
data is big.
"""

# Author: Balazs Kegl <balazs.kegl@gmail.com>
# License: BSD 3 clause
from sklearn.model_selection import ShuffleSplit


class GrowingFolds(object):
    def __init__(
        self,
        cv_method=ShuffleSplit,
        train_sizes=[0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 0.6, 0.7, 0.8, 0.9], 
        n_splits_per_size=100,
        random_state=57,
    ):
        self.cv_method = cv_method
        self.n_splits_per_size = n_splits_per_size
        self.train_sizes = train_sizes
        self.random_state = random_state

    def get_cv(self, X, y, fold_idxs=None):
        if fold_idxs is None:
            fold_idxs = range(self.n_splits_per_size * len(self.train_sizes))
        train_sizes_start = min(fold_idxs) // self.n_splits_per_size
        train_sizes_stop = max(fold_idxs) // self.n_splits_per_size + 1
        cvs = []
        for train_size in self.train_sizes[train_sizes_start:train_sizes_stop]:
            cv = self.cv_method(
                n_splits=self.n_splits_per_size, train_size=train_size,
                random_state=self.random_state)
            cvs = cvs + list(cv.split(X, y))
        splits = [cv for cv_i, cv in enumerate(cvs)
                  if cv_i + (train_sizes_start * self.n_splits_per_size)
                  in fold_idxs]
        return splits
