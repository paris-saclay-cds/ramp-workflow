import sys
import time
import inspect
import hashlib
import numpy as np
import pandas as pd
from pathlib import Path
from ..utils.importing import import_module_from_source

class FeatureExtractorClassifierWithEDA(object):
    def __init__(self, workflow_element_names=[
            'feature_extractor', 'classifier']):
        self.element_names = workflow_element_names
        self.cache_path = Path('.') / 'cache'
        self.cache_path.mkdir(parents=True, exist_ok=True)
        
    def _cache_transform(self, fe, X):
        data_hash = hashlib.sha256(
            np.ascontiguousarray(X.to_numpy())).hexdigest()
        cache_f_name = f'X_tr_{self.fe_hash}_{data_hash}.pkl'

        t0 = time.time()
        if hasattr(fe, 'to_cache') and fe.to_cache:
            try:
                X_tr = pd.read_pickle(self.cache_path / cache_f_name)
            except FileNotFoundError:
                X_tr = fe.transform(X)
                X_tr.to_pickle(self.cache_path / cache_f_name)
        else:
            X_tr = fe.transform(X)
        transform_time = time.time() - t0
#        print(f'size = {X.shape}, transform time = {transform_time}')
        return X_tr
        
    def train_submission(self, module_path, X_and_eda, y, train_is=None,
                         prev_trained_model=None):
        if train_is is None:
            train_is = slice(None, None, None)

        feature_extractor = import_module_from_source(
            Path(module_path) / f'{self.element_names[0]}.py',
            self.element_names[0])
        X = X_and_eda[0]
        eda = X_and_eda[1]
        X = X.copy()
        fe = feature_extractor.FeatureExtractor(eda)
        self.fe_hash = hashlib.sha256(
            inspect.getsource(feature_extractor).encode('utf-8')).hexdigest()
        
        t0 = time.time()
        fe.fit(X.iloc[train_is], y[train_is].ravel())        
        fit_time = time.time() - t0
#        print(f'size = {X.iloc[train_is].shape}, fit time = {fit_time}')

        X_tr = self._cache_transform(fe, X.iloc[train_is])

        classifier = import_module_from_source(
            Path(module_path) / f'{self.element_names[1]}.py',
            self.element_names[1],
        )
        clf = classifier.Classifier(eda)
        if prev_trained_model is None:
            clf.fit(X_tr, y[train_is].ravel())
        else:
            clf.fit(X_tr, y[train_is].ravel(), prev_trained_model[1])

        return fe, clf

    def test_submission(self, trained_model, X_and_eda):
        fe, clf = trained_model
        X = X_and_eda[0]
        try:
            X_tr = self._cache_transform(fe, X)
            y_proba = clf.predict_proba(X_tr)
        # sometimes the predictor crashes because cached X_tr is not
        # compatible with fitted X_tr, like new one hot columns
        # created for missing data
        except:
            X_tr = fe.transform(X)
            y_proba = clf.predict_proba(X_tr)            
        return y_proba
