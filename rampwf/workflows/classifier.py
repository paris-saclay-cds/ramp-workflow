from importlib import import_module


class Classifier(object):
    def __init__(self, workflow_element_names=['classifier']):
        self.workflow_element_names = workflow_element_names

    def train_submission(self, module_path, X_array, y_array, train_is=None):
        if train_is is None:
            train_is = range(len(y_array))
        submitted_classifier_module = '.{}'.format(
            self.workflow_element_names[0])
        classifier = import_module(submitted_classifier_module, module_path)
        clf = classifier.Classifier()
        clf.fit(X_array[train_is], y_array[train_is])
        return clf

    def test_submission(self, trained_model, X_array):
        clf = trained_model
        y_proba = clf.predict_proba(X_array)
        return y_proba

workflow = Classifier()
