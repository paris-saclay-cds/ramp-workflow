import os
import time
import inspect
import hashlib
import numpy as np
import pandas as pd
from pathlib import Path
from rampwf.utils.importing import import_module_from_source
from typing import Tuple, Any, Optional
import types
from rampwf.workflows import BaseWorkflow


class FeatureExtractorClassifierWithEDA(BaseWorkflow):
    def __init__(
        self,
        data_preprocessor_name: str = "data_preprocessor",
        feature_extractor_name: str = "feature_extractor",
        classifier_name: str = "classifier",
    ):
        self.data_preprocessor_name = data_preprocessor_name
        self.feature_extractor_name = feature_extractor_name
        self.classifier_name = classifier_name
        self.element_names = [
            data_preprocessor_name,
            feature_extractor_name,
            classifier_name,
        ]
        self.cache_path = Path(".") / "cache"
        self.cache_path.mkdir(parents=True, exist_ok=True)

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

    def preprocess_data(
        self,
        module_path: str,
        X_train: Tuple[pd.DataFrame, types.ModuleType],
        y_train: np.ndarray,
        X_test: Tuple[pd.DataFrame, types.ModuleType],
    ) -> Tuple[
        Tuple[pd.DataFrame, types.ModuleType],
        np.ndarray,
        Tuple[pd.DataFrame, types.ModuleType],
    ]:
        """This function preprocesses the data through the data_preprocessor

        Args:
            module_path (str): path of the submission
            X_train (Tuple[pd.DataFrame, types.ModuleType]): Train dataset with eda
            y_train (np.ndarray[Any, np.dtype[np.float64]]): train target
            X_test (Tuple[pd.DataFrame, types.ModuleType]): test dataset with eda

        Returns:
            Tuple[Tuple[pd.DataFrame, types.ModuleType], np.ndarray, Tuple[pd.DataFrame, types.ModuleType]]: The preprocessed data X_train_eda, y_train, X_test_eda
        """
        data_preprocessor_path = Path(module_path) / "data_preprocessor.py"
        if data_preprocessor_path.is_file():
            # Load preprocessor
            data_preprocessor = import_module_from_source(
                data_preprocessor_path, "data_preprocessor"
            )
            dp = data_preprocessor.DataPreprocessor()
            eda = X_train[1]

            X_train, y_train, X_test, eda = dp.preprocess(
                X=X_train[0], y=y_train, X_test=X_test[0], eda=eda
            )

            X_train = (X_train, eda)
            X_test = (X_test, eda)
        else:
            print(
                f"No preprocessor found in submission: {data_preprocessor_path.parent}"
            )
        return X_train, y_train, X_test

    def train_submission(
        self,
        module_path: str,
        X_and_eda: Tuple[pd.DataFrame, types.ModuleType],
        y: np.ndarray[Any, np.dtype[np.float64]],
        train_is: Optional[slice] = None,
        prev_trained_model: Optional[Tuple[Any, Any, Any]] = None,
    ) -> Tuple[Any, Any]:
        """Train the submission in module_path on the given dataset

        Args:
            module_path (str): path of the submission
            X_and_eda (Tuple[pd.DataFrame, str]): train dataset and eda info
            y (np.ndarray[Any, np.dtype[np.float64]]): target data
            train_is (Optional[slice], optional): List of training indeces. Defaults to None.
            prev_trained_model (Any, optional): previously trained model. Defaults to None.

        Returns:
            Tuple[Any, Any]: feature_extractor and classifier
        """
        if train_is is None:
            train_is = slice(None, None, None)

        X = X_and_eda[0]
        eda = X_and_eda[1]
        X = X.copy()

        X = X.iloc[train_is]
        y = y[train_is]

        # Perform feature extraction
        # ---------------------------
        feature_extractor = import_module_from_source(
            Path(module_path) / f"{self.feature_extractor_name}.py",
            self.feature_extractor_name,
        )
        fe = feature_extractor.FeatureExtractor(eda)
        self.fe_hash = hashlib.sha256(
            inspect.getsource(feature_extractor).encode("utf-8")
        ).hexdigest()

        fe.fit(X, y)
        X_tr = self._cache_transform(fe, X)
        # ---------------------------

        # Train model
        # ---------------------------
        classifier = import_module_from_source(
            Path(module_path) / f"{self.classifier_name}.py",
            self.classifier_name,
        )
        clf = classifier.Classifier(eda)
        if prev_trained_model is None:
            clf.fit(X_tr, y)
        else:
            clf.fit(X_tr, y, prev_trained_model[1])
        # ---------------------------

        return fe, clf

    def test_submission(
        self,
        trained_submission: Tuple[Any, Any],
        X_and_eda: Tuple[pd.DataFrame, types.ModuleType],
    ) -> np.ndarray[Any, np.dtype[np.float64]]:
        """Tests the trained submission

        Args:
            trained_submission (Tuple[Any, Any]): Trained submission consisting of [feature_extractor, model]
            X_and_eda (Tuple[pd.DataFrame, str]): Dataset and eda info

        Returns:
            np.ndarray[Any, np.dtype[np.float64]]: Target probabilities
        """
        fe, clf = trained_submission
        X = X_and_eda[0]
        eda = X_and_eda[1]

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
