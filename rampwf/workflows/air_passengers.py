from .feature_extractor_regressor import FeatureExtractorRegressor


class AirPassengers(FeatureExtractorRegressor):
    def __init__(self, workflow_element_names=[
            'feature_extractor', 'regressor', 'external_data.csv']):
        super(AirPassengers, self).__init__(workflow_element_names[:2])
        self.element_names = workflow_element_names
