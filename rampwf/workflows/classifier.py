import imp


class Classifier(object):
    def __init__(self, workflow_element_names=['classifier']):
        self.element_names = workflow_element_names
        # self.name = 'classifier_workflow'  # temporary

    def train_submission(self, module_path, X_array, y_array, train_idxs=None):
        if train_idxs is None:
            train_idxs = slice(None, None, None)
        submitted_classifier_file = '{}/{}.py'.format(
            module_path, self.element_names[0])
        classifier = imp.load_source('', submitted_classifier_file)
        clf = classifier.Classifier()
        clf.fit(X_array[train_idxs], y_array[train_idxs])
        return clf

    def test_submission(self, trained_model, X_array):
        clf = trained_model
        y_proba = clf.predict_proba(X_array)
        return y_proba
