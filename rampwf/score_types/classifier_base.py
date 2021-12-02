from .base import BaseScoreType


class ClassifierBaseScoreType(BaseScoreType):
    def score_function(self, ground_truths, predictions):
        self.label_names = ground_truths.label_names
        y_pred_label_index = predictions.y_pred_label_index
        y_true_label_index = ground_truths.y_pred_label_index
        self.check_y_pred_dimensions(y_true_label_index, y_pred_label_index)
        return self.__call__(y_true_label_index, y_pred_label_index)
