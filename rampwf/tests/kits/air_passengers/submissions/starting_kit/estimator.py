import os
import pandas as pd
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor


def _merge_external_data(X):
    filepath = os.path.join(os.path.dirname(__file__),
                            'external_data.csv')
    X_external = pd.read_csv(filepath)
    X_merged = pd.merge(X, X_external, how='left',
                        on=['DateOfDeparture', 'Arrival'], sort=False)
    return X_merged


def get_estimator():
    merge_transformer = FunctionTransformer(_merge_external_data,
                                            validate=False)
    categorical_cols = ['Arrival', 'Departure']
    drop_col = ['DateOfDeparture']
    preoprocessor = make_column_transformer(
        (OneHotEncoder(handle_unknown='ignore'), categorical_cols),
        ('drop', drop_col),
        remainder='passthrough'
    )
    pipeline = Pipeline(steps=[
        ('merge', merge_transformer),
        ('transfomer', preoprocessor),
        ('regressor', RandomForestRegressor(n_estimators=10, max_depth=10,
                                            max_features=10)),
    ])
    return pipeline
