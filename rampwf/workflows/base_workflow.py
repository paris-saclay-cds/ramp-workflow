from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Tuple, Any, Optional
import types
from abc import ABC, abstractmethod, abstractstaticmethod


class BaseWorkflow(ABC):
    def set_metadata(self, metadata: Any) -> None:
        self.metadata = metadata
    
    def get_element_names(self) -> list[str]:
        return self.element_names
    
    def set_element_names(self, submission_path) -> None:
        pass

    def preprocess_data(
        self,
        submission_path: str,
        X_train: Any,
        y_train: Any,
        X_test: Any,
    ) -> Tuple[Any, Any, Any]:
        """This function preprocesses the data through the data_preprocessor.

        Default returns the input, can be overridden by derived workflows.

        Args:
            submission_path (str): path of the submission
            X_train (Any): train dataset, typically pd.DataFrame
            y_train (Any): train target, typically np.ndarray
            X_test (Any): test dataset, typically pd.DataFrame

        Returns:
            Tuple[Any, Any, Any]: The preprocessed data X_train, y_train, X_test
        """
        return X_train, y_train, X_test


    @abstractmethod
    def train_submission(
        self,
        submission_path: str,
        X_train: Any,
        y_train: Any,
        train_is: Optional[Any] = None,
        prev_trained_submission: Optional[Any] = None,
    ) -> Any:
        """Train the submission in module_path on the given dataset.

        Optionally indexed by CV slice.
        Optionally warm-started from prev_trained_submission.

        Args:
            submission_path (str): path of the submission
            X_train (Any): train dataset, typically pd.DataFrame
            y_train (Any): train target, typically np.ndarray
            train_is (Any, optional): CV training object
                Typically slice, list of training indices. Defaults to None.
            prev_trained_submission (Any, optional): previously trained model.
                Defaults to None.

        Returns:
            trained_submission (Any): trained trained_submission
        """

    @abstractmethod
    def test_submission(
        self,
        trained_submission: Any,
        X_test: Any,
    ) -> Any:
        """Tests the trained submission.

        Args:
            trained_submission (Any): trained submission
                Returned by train_submission.
            X_test (Any): test dataset, typically pd.DataFrame

        Returns:
            y_pred (Any): target predictions, typically np.ndarray
        """
