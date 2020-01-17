from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression


def get_pipeline():

    to_encode = ['Sex', 'Pclass', 'Embarked']
    to_keep = ['Age', 'SibSp', 'Parch', 'Fare']
    to_drop = ['Name', 'Ticket', 'Cabin']

    transformer = ColumnTransformer(transformers=[
        ('onehotencode', make_pipeline(OneHotEncoder(
            handle_unknown='ignore', drop='first')), to_encode)
        ('numeric', make_pipeline(SimpleImputer(
            strategy='constant', fill_value=-1)), to_keep),
        ('drop', 'drop', to_drop),
    ])

    pipeline = Pipeline([
        ('transformer', transformer)
        ('imputer', SimpleImputer(strategy='median')),
        ('classifier', LogisticRegression(C=1., solver='lbfgs'))
    ])

    return pipeline
