from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression


def get_pipeline():
    numeric_cols = ['Pclass', 'Age', 'SibSp', 'Parch',
                    'Fare']
    to_drop = ['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked']

    transformer = ColumnTransformer(transformers=[
        ('numeric', make_pipeline(SimpleImputer(strategy='median')),
         numeric_cols),
        ('drop', 'drop', to_drop),
    ])
    pipeline = Pipeline(steps=[
        ('transformer', transformer),
        ('classifier', LogisticRegression(C=1., solver='lbfgs'),)
    ])
    return pipeline