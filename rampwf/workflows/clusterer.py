"""A supervised clustering or transfer learning workflow.

Training and test data consist of a set of events ad each event has an
unknown number of clusters. At training time we know the classes (clusters) of
each event. At test time we only know which instances belong to the same event.
Events share statistical properties which can be exploited.

From another point of view, the setup is transfer learning: each event is
a new task, and classed in the task share statistical properties.

User submissions implement a fit function that receives both event ids and
cluster ids (besides input covariates). At test time `test_submission`
receives only the coavariates and the event_id (assumed to be the first column
of `X_array`). It slices up `X_array` into single events, drops the event ids,
and sends the single event to the `predict_single_event` function implemented
by the users. This function returns a vector of labels (cluster assignments)
which is then joined back to the event id column and returned (to be passed
into `prediction_types.Clustering` and evaluated by
`score_types.clustering_efficiency`).
"""

# Author: Balazs Kegl <balazs.kegl@gmail.com>
# License: BSD 3 clause

import os

import numpy as np

from ..utils.importing import import_module_from_source


class Clusterer(object):
    def __init__(self, workflow_element_names=['clusterer']):
        self.element_names = workflow_element_names

    def train_submission(self, module_path, X_array, y_array, train_is=None):
        if train_is is None:
            train_is = slice(None, None, None)
        clusterer = import_module_from_source(
            os.path.join(module_path, self.element_names[0] + '.py'),
            self.element_names[0],
            sanitize=True
        )
        ctr = clusterer.Clusterer()
        ctr.fit(X_array[train_is], y_array[train_is])
        return ctr

    def test_submission(self, trained_model, X_array):
        ctr = trained_model
        unique_event_ids = np.unique(X_array[:, 0])
        cluster_ids = np.empty(len(X_array), dtype='int')

        for event_id in unique_event_ids:
            event_indices = (X_array[:, 0] == event_id)
            # select an event and drop event ids
            X_event = X_array[event_indices][:, 1:]
            cluster_ids[event_indices] = ctr.predict_single_event(X_event)

        return np.stack(
            (X_array[:, 0], cluster_ids), axis=-1).astype(dtype='int')
