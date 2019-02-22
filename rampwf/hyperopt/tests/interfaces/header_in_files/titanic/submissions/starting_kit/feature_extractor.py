import pandas as pd
import numpy as np
from rampwf.hyperopt import Hyperparameter

# RAMP START HYPERPARAMETERS
complex_features = Hyperparameter(default=True, values=[True, False])
# RAMP END HYPERPARAMETERS


class FeatureExtractor():
    def __init__(self):
        pass

    def fit(self, X_df, y):
        pass

    def transform(self, X_df):
        if int(complex_features):
            X_df = X_df.assign(LogFare=lambda x: np.log(x.Fare + 10.))
            X_df = X_df.assign(Cab=lambda x: x.Cabin == x.Cabin)

            X_df_new = pd.concat(
                [X_df.get(['Parch']),
                 X_df.assign(
                    LogFare=lambda x: 10 * np.log(x.Fare + 1.)**0.5).get(
                    ['LogFare']),
                 X_df.assign(SibSp=lambda x: np.exp(x.SibSp) + 0.6 * np.exp(
                     x.Parch)).get(['SibSp']),
                 X_df.assign(Age=lambda x: 10 * np.log(x.Age + 0.01)).get(
                    ['Age']),
                 pd.get_dummies(X_df.Sex, prefix='Sex', drop_first=False),
                 pd.get_dummies(
                    X_df.Pclass, prefix='Pclass', drop_first=False),
                 pd.get_dummies(
                    X_df.Embarked, prefix='Embarked', drop_first=True),
                 pd.get_dummies(X_df.Cab, prefix='Cab', drop_first=True)],
                axis=1)
        else:
            X_df_new = pd.concat(
                [X_df.get(['Fare', 'Age', 'SibSp', 'Parch']),
                 pd.get_dummies(X_df.Sex, prefix='Sex', drop_first=True),
                 pd.get_dummies(X_df.Pclass, prefix='Pclass', drop_first=True),
                 pd.get_dummies(
                     X_df.Embarked, prefix='Embarked', drop_first=True)],
                axis=1)

        X_df_new = X_df_new.fillna(-1)
        XX = X_df_new.values
        return XX
