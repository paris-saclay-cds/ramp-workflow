import sys
import time
import glob
import copy
import json
import inspect
import hashlib
import numpy as np
import pandas as pd
from pathlib import Path
from ..utils.importing import import_module_from_source
from . import BaseWorkflow
from typing import Tuple, Any, Optional
import types


class TabularClassifier(BaseWorkflow):
    def __init__(self):
        self.cache_path = Path(".") / "cache"
        self.cache_path.mkdir(parents=True, exist_ok=True)

    def set_element_names(self, submission_path):
        self.element_names = []
        i = 0
        while True:
            submissions_f_names = glob.glob(
                f'{submission_path}/data_preprocessor_{i}_*.py')
            if len(submissions_f_names) == 0:
                break
            data_preprocessor_path = submissions_f_names[0]
            self.element_names.append(Path(data_preprocessor_path).stem)
            i += 1
        self.feature_extractor_name = 'feature_extractor'
        self.element_names.append(self.feature_extractor_name)
        self.classifier_name = 'classifier'
        self.element_names.append(self.classifier_name)

    def _cache_transform(
        self, fe: Any, X: pd.DataFrame
    ) -> np.ndarray[Any, np.dtype[np.float64]]:

        t0 = time.time()
        if getattr(fe, "to_cache", False):
            data_hash = hashlib.sha256(np.ascontiguousarray(X.to_numpy())).hexdigest()
            cache_f_name = f"X_tr_{self.fe_hash}_{data_hash}.pkl"
            try:
                X_tr = pd.read_pickle(self.cache_path / cache_f_name)
            except FileNotFoundError:
                X_tr = fe.transform(X)
                X_tr.to_pickle(self.cache_path / cache_f_name)
        else:
            X_tr = fe.transform(X)
        transform_time = time.time() - t0
        return X_tr

    def _run_preprocessors(self, data_preprocessors, X_train, y_train, X_test, metadata):
        for dp in data_preprocessors:
            print(dp)
            X_train, y_train, X_test, metadata = dp.preprocess(
                X_train=X_train, y_train=y_train, X_test=X_test,
                metadata=metadata
            )
        # to defragment
        X_train = X_train.copy()
        X_test = X_test.copy()
        return X_train, y_train, X_test, metadata

    def preprocess_data(
        self,
        submission_path: str,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_test: pd.DataFrame,
    ) -> Tuple[
        pd.DataFrame,
        np.ndarray,
        pd.DataFrame,
    ]:
        """Preprocess the data through numbered data_preprocessor_<i>_*'s.

        Args:
            module_path (str): path of the submission
            X_train (pd.DataFrame): Train dataset
            y_train (np.ndarray[Any, np.dtype[np.float64]]): train target
            X_test (pd.DataFrame): test dataset

        Returns:
            Tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
                The preprocessed data X_train, y_train, X_test
        """
        self.metadata_after_dp = copy.deepcopy(self.metadata)
        data_preprocessor_names = [
            n for n in self.element_names if n[:18] == 'data_preprocessor_']
        data_preprocessors = []
        dp_hash = ''
        for data_preprocessor_name in data_preprocessor_names:
            data_preprocessor_path = Path(submission_path) / f'{data_preprocessor_name}.py'
            data_preprocessor = import_module_from_source(
                data_preprocessor_path, data_preprocessor_name
            )
            dp = data_preprocessor.DataPreprocessor()
            data_preprocessors.append(dp)
            dp_hash += inspect.getsource(data_preprocessor)
        to_cache = any([hasattr(dp, 'to_cache') and dp.to_cache for dp in data_preprocessors])
        if to_cache:
            dp_hash = hashlib.sha256(dp_hash.encode("utf-8")).hexdigest()
            X_train_hash = hashlib.sha256(np.ascontiguousarray(pd.util.hash_pandas_object(X_train))).hexdigest()
            X_test_hash = hashlib.sha256(np.ascontiguousarray(pd.util.hash_pandas_object(X_test))).hexdigest()
            y_train_hash = hashlib.sha256(np.ascontiguousarray(y_train)).hexdigest()
            metadata_hash = hashlib.sha256(json.dumps(self.metadata_after_dp, sort_keys=True).encode("utf-8")).hexdigest()
            X_train_cache_f_name = f"X_train_dp_{dp_hash}_{X_train_hash}.pkl"
            X_test_cache_f_name = f"X_test_dp_{dp_hash}_{X_test_hash}.pkl"
            y_train_cache_f_name = f"y_train_dp_{dp_hash}_{y_train_hash}.npy"
            metadata_cache_f_name = f"metadata_dp_{dp_hash}_{metadata_hash}.json"
            try:
                X_train_loaded = pd.read_pickle(self.cache_path / X_train_cache_f_name)
                X_test_loaded = pd.read_pickle(self.cache_path / X_test_cache_f_name)
                y_train_loaded = np.load(self.cache_path / y_train_cache_f_name)
                metadata_loaded = json.load(open(self.cache_path / metadata_cache_f_name))
                X_train = X_train_loaded
                X_test = X_test_loaded
                y_train = y_train_loaded
                self.metadata_after_dp = metadata_loaded
            except FileNotFoundError:
                X_train, y_train, X_test, self.metadata_after_dp = self._run_preprocessors(
                    data_preprocessors, X_train, y_train, X_test, self.metadata_after_dp)
                X_train.to_pickle(self.cache_path / X_train_cache_f_name)
                X_test.to_pickle(self.cache_path / X_test_cache_f_name)
                np.save(self.cache_path / y_train_cache_f_name, y_train)
                json.dump(self.metadata_after_dp, open(self.cache_path / metadata_cache_f_name, "w"))
        else:
            X_train, y_train, X_test, self.metadata_after_dp = self._run_preprocessors(
                data_preprocessors, X_train, y_train, X_test, self.metadata_after_dp)
        return X_train, y_train, X_test

    def train_submission(
        self,
        submission_path: str,
        X_train: pd.DataFrame,
        y_train: np.ndarray[Any, np.dtype[np.float64]],
        train_is: Optional[slice] = None,
        prev_trained_model: Optional[Tuple[Any, Any, Any]] = None,
    ) -> Tuple[Any, Any]:
        """Train the submission in module_path on the given dataset

        Args:
            submission_path (str): path of the submission
            X_train (pd.DataFrame): train dataset
            y_train (np.ndarray[Any, np.dtype[np.float64]]): train target
            train_is (Optional[slice], optional): List of training indeces. Defaults to None.
            prev_trained_model (Any, optional): previously trained model. Defaults to None.

        Returns:
            Tuple[Any, Any]: trained feature_extractor and classifier
        """
        if train_is is None:
            train_is = slice(None, None, None)

        X_train = X_train.copy()
        X_train = X_train.iloc[train_is]
        y_train = y_train[train_is]

        feature_extractor = import_module_from_source(
            Path(submission_path) / f"{self.feature_extractor_name}.py",
            self.feature_extractor_name,
        )
        self.metadata_after_fe = copy.deepcopy(self.metadata_after_dp)
        fe = feature_extractor.FeatureExtractor(self.metadata_after_fe)
        self.fe_hash = hashlib.sha256(
            inspect.getsource(feature_extractor).encode("utf-8")
        ).hexdigest()
        fe.fit(X_train, y_train)
        X_train = self._cache_transform(fe, X_train)

        classifier = import_module_from_source(
            Path(submission_path) / f"{self.classifier_name}.py",
            self.classifier_name,
        )
        clf = classifier.Classifier(self.metadata_after_fe)
        if prev_trained_model is None:
            clf.fit(X_train, y_train)
        else:
            clf.fit(X_train, y_train, prev_trained_model[1])

        return fe, clf

    def test_submission(
        self,
        trained_submission: Tuple[Any, Any],
        X: pd.DataFrame,
    ) -> np.ndarray[Any, np.dtype[np.float64]]:
        """Tests the trained submission

        Args:
            trained_submission (Tuple[Any, Any]): Trained submission consisting of [feature_extractor, model]
            X (pd.DataFrame): test dataset

        Returns:
            np.ndarray[Any, np.dtype[np.float64]]: Target predictions
        """
        fe, clf = trained_submission

        try:
            X = self._cache_transform(fe, X)
            y_proba = clf.predict_proba(X)
        # sometimes the predictor crashes because cached X is not
        # compatible with fitted X, like new one hot columns
        # created for missing data
        except:
            X = fe.transform(X)
            y_proba = clf.predict_proba(X)
        return y_proba
