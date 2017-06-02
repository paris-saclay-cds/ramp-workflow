class BaseScoreType(object):
    def check(self, y_true, y_pred):
        if self.n_columns == 0:
            assert len(y_true.shape) == 1
            assert len(y_pred.shape) == 1
        else:
            assert len(y_true.shape) == 2
            assert len(y_pred.shape) == 2
            assert y_true.shape[0] == self.n_columns
            assert y_pred.shape[0] == self.n_columns
        assert len(y_true) == len(y_pred)

    @property
    def worst(self):
        if self.is_lower_the_better:
            return self.maximum
        else:
            return self.minimum
