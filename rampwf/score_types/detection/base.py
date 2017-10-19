from __future__ import division

from ..base import BaseScoreType


class DetectionBaseScoreType(BaseScoreType):
    """Common abstract base type for detection scores.

    Implements `__call__` by selecting all prediction detections with
    a confidence higher than `conf_threshold`. It assumes that the child
    class implements `detection_score`.
    """

    conf_threshold = 0.5

    def __call__(self, y_true, y_pred, conf_threshold=None):
        if conf_threshold is None:
            conf_threshold = self.conf_threshold
        y_pred_above_confidence = [
            [detected_object[1:] for detected_object in single_detection
             if detected_object[0] > conf_threshold]
            for single_detection in y_pred]
        return self.detection_score(y_true, y_pred_above_confidence)


class DetectionBaseIOUScoreType(DetectionBaseScoreType):

    def __init__(self, name=None, precision=2, conf_threshold=0.5,
                 minipatch=None, iou_threshold=0.5):
        if name is None:
            self.name = '{name}(IOU={iou_threshold})'.format(
                name=self._name, iou_threshold=iou_threshold)
        else:
            self.name = name
        self.precision = precision
        self.conf_threshold = conf_threshold
        self.minipatch = minipatch
        self.iou_threshold = iou_threshold

