import os
import pandas as pd
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor


def _merge_weather(X):
    filepath = os.path.join(os.path.dirname(__file__),
                            'external_data_mini.csv')
    X_weather = pd.read_csv(filepath)
    X_merged = pd.merge(X, X_weather, how='left',
                        on=['DateOfDeparture', 'Arrival'], sort=False)
    return X_merged

def get_pipeline():
    merge_transformer = FunctionTransformer(_merge_weather, validate=False)
    to_encode = ['Arrival', 'Departure']
    transformer = ColumnTransformer(transformers=[
        ('onehotencode', make_pipeline(OneHotEncoder(handle_unknown='ignore')),
         to_encode),
        ('drop', 'drop', 'DateOfDeparture')
    ])

    pipeline = Pipeline(steps=[
        ('merge', merge_transformer),
        ('transfomer', transformer),
        ('regressor', RandomForestRegressor(n_estimators=10, max_depth=10,
                                            max_features=10)),
    ])
    return pipeline
