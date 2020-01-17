from sklearn.ensemble import RandomForestClassifier


def get_pipeline():
    clf = RandomForestClassifier(
        n_estimators=10, max_leaf_nodes=10, random_state=61)
    return clf
