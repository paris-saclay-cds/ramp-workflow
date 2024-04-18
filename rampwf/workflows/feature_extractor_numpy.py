import os

import numpy as np

from ..utils.importing import import_module_from_source
from .base_workflow import BaseWorkflow


class FeatureExtractorNumpy(BaseWorkflow):
    """
    Feature extractor with numpy inputs and outputs.

    Relying only on numpy and not on pandas allows to gain on speed when
    calling the regressor multiple times sequentially (e.g. in MPC).
    """

    def __init__(self, workflow_element_names=['feature_extractor']):
        self.element_names = workflow_element_names

    def train_submission(self, module_path, X_array, y_array, train_is=None):
        if train_is is None:
            train_is = slice(None, None, None)
        feature_extractor = import_module_from_source(
            os.path.join(module_path, self.element_names[0] + '.py'),
            self.element_names[0],
            sanitize=True
        )
        fe = feature_extractor.FeatureExtractor()
        try:
            fe.fit(X_array[train_is], y_array[train_is])
        except (AttributeError, IndexError, ValueError):
            pass

        return fe

    def test_submission(self, trained_model, X_array):
        fe = trained_model
        X_array_tf = fe.transform(X_array)
        return X_array_tf
