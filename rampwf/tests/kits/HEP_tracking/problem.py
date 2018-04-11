import os
import pandas as pd
import rampwf as rw

problem_title = 'Particle tracking in the LHC ATLAS detector'
# A type (class) which will be used to create wrapper objects for y_pred
Predictions = rw.prediction_types.make_clustering()
# An object implementing the workflow
workflow = rw.workflows.Clusterer()
score_types = [
    rw.score_types.ClusteringEfficiency(name='efficiency', precision=3),
]
# validation folds don't cut into events
cv = rw.cvs.Clustering(n_cv=1, cv_test_size=0.5, random_state=57)
get_cv = cv.get_cv


def _read_data(path, f_name):
    df = pd.read_csv(os.path.join(path, 'data', f_name))
    y_df = df[['event_id', 'cluster_id']]
    X_df = df.drop(['cluster_id'], axis=1)
    return X_df.values, y_df.values


def get_train_data(path='.'):
    f_name = 'train.csv'
    return _read_data(path, f_name)


def get_test_data(path='.'):
    f_name = 'test.csv'
    return _read_data(path, f_name)
