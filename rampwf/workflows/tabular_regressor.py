import sys
import time
import glob
import inspect
import hashlib
import numpy as np
import pandas as pd
from pathlib import Path
from rampwf.utils.importing import import_module_from_source
from . import BaseWorkflow
from typing import Tuple, Any, Optional
import types


class TabularRegressor(BaseWorkflow):
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
            print(f'Adding {data_preprocessor_path} to workflow elements')
            self.element_names.append(Path(data_preprocessor_path).stem)
            i += 1
        self.feature_extractor_name = 'feature_extractor'
        self.element_names.append(self.feature_extractor_name)
        self.regressor_name = 'regressor'
        self.element_names.append(self.regressor_name)

    def _cache_transform(
        self, fe: Any, X: pd.DataFrame
    ) -> np.ndarray[Any, np.dtype[np.float64]]:
        data_hash = hashlib.sha256(np.ascontiguousarray(X.to_numpy())).hexdigest()
        cache_f_name = f"X_tr_{self.fe_hash}_{data_hash}.pkl"

        t0 = time.time()
        if hasattr(fe, "to_cache") and fe.to_cache:
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
        submission_path: str,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_test: pd.DataFrame,
    ) -> Tuple[
        pd.DataFrame,
        np.ndarray,
        pd.DataFrame,
    ]:
        """This function preprocesses the data through the data_preprocessor

        Args:
            module_path (str): path of the submission
            X_train (pd.DataFrame): Train dataset
            y_train (np.ndarray[Any, np.dtype[np.float64]]): train target
            X_test (pd.DataFrame): test dataset

        Returns:
            Tuple[pd.DataFrame, np.ndarray, pd.DataFrame]: The preprocessed data X_train, y_train, X_test
        """
        data_preprocessor_names = [n for n in self.element_names if n[:18] == 'data_preprocessor_']
        for data_preprocessor_name in data_preprocessor_names:
            data_preprocessor_path = Path(submission_path) / f'{data_preprocessor_name}.py'
            data_preprocessor = import_module_from_source(
                data_preprocessor_path, data_preprocessor_name
            )
            dp = data_preprocessor.DataPreprocessor()

            X_train, y_train, X_test, self.metadata = dp.preprocess(
                X_train=X_train, y_train=y_train, X_test=X_test,
                metadata=self.metadata
            )
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
            Tuple[Any, Any]: trained feature_extractor and regressor
        """
        if train_is is None:
            train_is = slice(None, None, None)

        X_train = X_train.copy()
        X_train = X_train.iloc[train_is]
        y_train = y_train[train_is]

        # Perform feature extraction
        # ---------------------------
        feature_extractor = import_module_from_source(
            Path(submission_path) / f"{self.feature_extractor_name}.py",
            self.feature_extractor_name,
        )
        fe = feature_extractor.FeatureExtractor(self.metadata)
        self.fe_hash = hashlib.sha256(
            inspect.getsource(feature_extractor).encode("utf-8")
        ).hexdigest()

        fe.fit(X_train, y_train)
        X_train = self._cache_transform(fe, X_train)

        regressor = import_module_from_source(
            Path(submission_path) / f"{self.regressor_name}.py",
            self.regressor_name,
        )
        reg = regressor.Regressor(self.metadata)
        if prev_trained_model is None:
            reg.fit(X_train, y_train)
        else:
            reg.fit(X_train, y_train, prev_trained_model[1])
        # ---------------------------

        return fe, reg

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
        fe, reg = trained_submission

        try:
            X = self._cache_transform(fe, X)
            y_pred = reg.predict(X)
        # sometimes the predictor crashes because cached X is not
        # compatible with fitted X, like new one hot columns
        # created for missing data
        except:
            X = fe.transform(X)
            y_pred = reg.predict(X)
        if len(y_pred.shape) == 1:
            y_pred = y_pred.reshape((len(y_pred), 1))
        return y_pred
