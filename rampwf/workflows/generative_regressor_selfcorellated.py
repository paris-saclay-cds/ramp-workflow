
from rampwf.utils.importing import import_file
import numpy as np

class GenerativeRegressorSelf(object):
    def __init__(self, target_column_name, nb_bins, workflow_element_names=['gen_regressor'], ):
        self.element_names = workflow_element_names
        self.target_column_name = target_column_name
        self.nb_bins = nb_bins

    def train_submission(self, module_path, X_array, y_array, train_is=None):
        if train_is is None:
            train_is = slice(None, None, None)
        gen_regressor = import_file(module_path, self.element_names[0])

        X_array = X_array.values[train_is,]

        regressors = []
        for i in range(len(self.target_column_name)):
            reg = gen_regressor.GenerativeRegressor(self.nb_bins)

            if i == 0 and y_array.shape[1] == 1:
                y = y_array[train_is]
            else:
                y = y_array.values[train_is, i]

            shape = y.shape
            if len(shape) == 1:
                y = y.reshape(-1, 1)
            elif len(shape) == 2:
                pass
            else:
                raise ValueError("More than two dims for y not supported")

            reg.fit(X_array, y)
            X_array = np.hstack([X_array, y])
            regressors.append(reg)
        return regressors

    def test_submission(self, trained_model, X_array):
        regressors = trained_model
        dims = []
        for i, reg in enumerate(regressors):
            to_drop = ["y_" + y for y in self.target_column_name[i:]]
            X = X_array.drop(columns=to_drop)
            X = X.values
            y_pred, bin_edges = reg.predict(X)
            dims.append(np.concatenate((bin_edges, y_pred), axis=2))

        # All the bins are the same in this classification setup
        return np.concatenate(dims, axis=1)

    def step(self, trained_model, X_array):
        """Careful, for now, for every x in the time dimension, we will sample a y"""
        regressors = trained_model
        y_sampled = []
        for i, reg in enumerate(regressors):
            X = X_array.copy()
            for j, predicted_dim in enumerate(np.array(y_sampled)):
                X["y_" + self.target_column_name[j]] = predicted_dim
            X = X.values
            y_pred, bin_edges = reg.predict(X)
            y_dim = []
            for prob, bins in zip(y_pred, bin_edges):
                prob = prob.ravel()
                bins = bins.ravel()
                selected = np.random.choice(list(range(len(prob))), p=prob)
                y_dim.append(np.random.uniform(low=bins[selected], high=bins[selected + 1]))
            y_sampled.append(y_dim)

        return np.array(y_sampled).swapaxes(0, 1)