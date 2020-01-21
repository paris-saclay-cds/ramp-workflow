from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression


def get_estimator():

    categorical_cols = ['Sex', 'Pclass', 'Embarked']
    numerical_cols = ['Age', 'SibSp', 'Parch', 'Fare']

    preprocessor = make_column_transformer(
        (OneHotEncoder(handle_unknown='ignore'), categorical_cols),
        (SimpleImputer(strategy='constant', fill_value=-1), numerical_cols),
    )

    pipeline = Pipeline([
        ('transformer', preprocessor),
        ('classifier', LogisticRegression()),
    ])

    return pipeline
