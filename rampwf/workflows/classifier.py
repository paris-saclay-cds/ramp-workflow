import os

from ..utils.importing import import_module_from_source


class Classifier(object):
    def __init__(self, workflow_element_names=['classifier']):
        self.element_names = workflow_element_names
        # self.name = 'classifier_workflow'  # temporary

    def train_submission(self, module_path, X_array, y_array, train_is=None,
                         prev_trained_submission=None):
        if train_is is None:
            train_is = slice(None, None, None)
        classifier = import_module_from_source(
            os.path.join(module_path, self.element_names[0] + '.py'),
            self.element_names[0],
            sanitize=True
        )
        clf = classifier.Classifier()
        clf.fit(X_array[train_is], y_array[train_is], prev_trained_submission)
        return clf

    def test_submission(self, trained_model, X_array):
        clf = trained_model
        y_proba = clf.predict_proba(X_array)
        return y_proba
