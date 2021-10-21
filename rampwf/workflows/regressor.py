import os

from ..utils.importing import import_module_from_source


class Regressor(object):
    def __init__(self, workflow_element_names=['regressor']):
        self.element_names = workflow_element_names

    def train_submission(self, module_path, X_array, y_array, train_is=None,
                         prev_trained_model=None):
        if train_is is None:
            train_is = slice(None, None, None)
        regressor = import_module_from_source(
            os.path.join(module_path, self.element_names[0] + '.py'),
            self.element_names[0],
            sanitize=True
        )
        reg = regressor.Regressor()
        if prev_trained_model is None:
            reg.fit(X_array[train_is], y_array[train_is])
        else:
            reg.fit(
                X_array[train_is], y_array[train_is], prev_trained_model)
        return reg

    def test_submission(self, trained_model, X_array):
        reg = trained_model
        y_pred = reg.predict(X_array)
        return y_pred
