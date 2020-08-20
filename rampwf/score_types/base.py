class BaseScoreType(object):
    def check_y_pred_dimensions(self, y_true, y_pred):
        if len(y_true) != len(y_pred):
            raise ValueError(
                'Wrong y_pred dimensions: y_pred should have {} instances, '
                'instead it has {} instances'.format(
                    len(y_true), len(y_pred)))

    @property
    def worst(self):
        if self.is_lower_the_better:
            return self.maximum
        else:
            return self.minimum

    def score_function(self, ground_truths, predictions):
        y_true = ground_truths.y_pred
        y_pred = predictions.y_pred
        self.check_y_pred_dimensions(y_true, y_pred)
        return self.__call__(y_true, y_pred)
