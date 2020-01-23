from sklearn.ensemble import RandomForestRegressor


def get_estimator():
    pipeline = RandomForestRegressor(
        n_estimators=2, max_leaf_nodes=2, random_state=61)
    return pipeline
