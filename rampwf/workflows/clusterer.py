import numpy as np
from importlib import import_module


class Clusterer(object):
    def __init__(self, workflow_element_names=['clusterer']):
        self.workflow_element_names = workflow_element_names

    def train_submission(self, module_path, X_array, y_array, train_is=None):
        if train_is is None:
            train_is = slice(None, None, None)
        submitted_clusterer_module = '.{}'.format(
            self.workflow_element_names[0])
        clusterer = import_module(submitted_clusterer_module, module_path)
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


workflow = Clusterer()
