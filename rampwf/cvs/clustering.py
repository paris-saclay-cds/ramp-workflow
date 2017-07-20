"""A clustering (transfer learning) cross validation class.

Training and test data consist of a set of events ad each event has an
unknown number of clusters. At training time we know the classes (clusters) of
each event. At test time we only know which instances belong to the same event.
Events share statistical properties which can be exploited.

From another point of view, the setup is transfer learning: each event is
a new task, and classed in the task share statistical properties.

We use sklearn `ShuffleSplit` over the _events_, so train/test cut does not
cut into events (each event or task is either completely in the training or
completely in the test fold).
"""

# Author: Balazs Kegl <balazs.kegl@gmail.com>
# License: BSD 3 clause
import numpy as np
from sklearn.model_selection import ShuffleSplit


class Clustering(object):
    def __init__(self, n_cv=1, cv_test_size=0.5, random_state=57):
        self.n_cv = n_cv
        self.cv_test_size = cv_test_size
        self.random_state = random_state

    def get_cv(self, X, y):
        unique_event_ids = np.unique(y[:, 0])
        event_cv = ShuffleSplit(
            n_splits=self.n_cv, test_size=self.cv_test_size,
            random_state=self.random_state)
        for train_event_is, test_event_is in event_cv.split(unique_event_ids):
            train_is = np.where(
                np.in1d(y[:, 0], unique_event_ids[train_event_is]))[0]
            test_is = np.where(
                np.in1d(y[:, 0], unique_event_ids[test_event_is]))[0]
            yield train_is, test_is
