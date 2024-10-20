"""R times k fold cross validation class.
"""

# Author: Balazs Kegl <balazs.kegl@gmail.com>
# License: BSD 3 clause
import numpy as np
from sklearn.model_selection import KFold


class RTimesK(object):
    def __init__(
        self,
        k=10,
        r=3,
    ):
        self.k = k
        self.r = r
 
    def get_cv(self, X, y, fold_idxs=None):
        splits = []
        for i in range(self.r):
            cv = KFold(n_splits=self.k, shuffle=True, random_state=57 + i)
            splits.append(list(cv.split(X, y)))
        # reorder column first and flatten
        splits = [folds[i] for i in range(self.k) for folds in splits]
        if fold_idxs is not None:
            splits = splits[fold_idxs]
        return splits
