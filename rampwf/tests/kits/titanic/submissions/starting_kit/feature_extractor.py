from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer


def get_feature_extractor():
    numeric_cols = ['Pclass', 'Age', 'SibSp', 'Parch',
                    'Fare']
    to_drop = ['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked']

    transformer = ColumnTransformer(transformers=[
        ('numeric', make_pipeline(SimpleImputer(strategy='median')), numeric_cols),
        ('drop', 'drop', to_drop),
    ])

    return transformer
