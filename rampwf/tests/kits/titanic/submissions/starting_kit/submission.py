from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression


def get_estimator():

    numeric_cols = ['Pclass', 'Age', 'SibSp', 'Parch',
                    'Fare']
    preprocessor = ColumnTransformer(transformers=[
        ('numeric', make_pipeline(SimpleImputer(strategy='median')),
         numeric_cols),
    ])
    pipeline = Pipeline(steps=[
        ('transformer', preprocessor),
        ('classifier', LogisticRegression()),
    ])
    return pipeline