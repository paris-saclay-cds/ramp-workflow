from sklearn.pipeline import Pipeline
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression


def get_estimator():

    numeric_cols = ['Pclass', 'Age', 'SibSp', 'Parch',
                    'Fare']
    preprocessor = make_column_transformer(
        (SimpleImputer(strategy='median'), numeric_cols),
    )
    pipeline = Pipeline(steps=[
        ('transformer', preprocessor),
        ('classifier', LogisticRegression()),
    ])
    return pipeline
