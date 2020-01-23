from sklearn.ensemble import RandomForestClassifier


def get_estimator():
    clf = RandomForestClassifier(n_estimators=1, max_leaf_nodes=2,
                                 random_state=61)
    return clf
