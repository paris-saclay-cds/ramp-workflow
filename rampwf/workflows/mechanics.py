
from .feature_extractor_regressor import FeatureExtractorRegressor


class Mechanics(FeatureExtractorRegressor):
    def __init__(self, workflow_element_names=[
            'feature_extractor', 'regressor']):
        super(Mechanics, self).__init__(workflow_element_names)
        self.element_names = workflow_element_names
