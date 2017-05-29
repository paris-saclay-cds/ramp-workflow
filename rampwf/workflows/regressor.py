import imp


class Regressor(object):
    def __init__(self, workflow_element_names=['regressor']):
        self.workflow_element_names = workflow_element_names

    def train_submission(self, module_path, X_array, y_array, train_is=None):
        if train_is is None:
            train_is = slice(None, None, None)
        submitted_regressor_file = '{}/{}.py'.format(
            module_path, self.workflow_element_names[0])
        regressor = imp.load_source('', submitted_regressor_file)
        reg = regressor.Regressor()
        reg.fit(X_array[train_is], y_array[train_is])
        return reg

    def test_submission(self, trained_model, X_array):
        reg = trained_model
        y_pred = reg.predict(X_array)
        return y_pred
