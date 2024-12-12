import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.utils.validation import check_is_fitted

from .fuzzycoco_base import FuzzyCocoBase


class FuzzyCocoRegressor(FuzzyCocoBase, RegressorMixin):
    def fit(
        self,
        X,
        y,
        output_filename: str = "fuzzySystem.ffs",
        script_file: str = "",
        verbose: bool = False,
        feature_names: list = None,
        target_name: str = "OUT",
    ):
        cdf = self._prepare_data(X, y, feature_names, target_name)
        self._run_script(cdf, output_filename, script_file, verbose)
        return self

    def predict(self, X, feature_names: list = None):
        check_is_fitted(self, "model_")
        cdf = self._prepare_data(X, None, feature_names)
        predictions = self.model_.smartPredict(cdf)
        predictions_list = predictions.to_list()

        # Convert predictions to floats
        result = [float(row[0]) for row in predictions_list]
        if isinstance(X, pd.DataFrame):
            return pd.Series(result, index=X.index)
        else:
            return np.array(result)

    def score(self, X, y, feature_names: list = None, target_name: str = "OUT"):
        check_is_fitted(self, "model_")
        if not isinstance(y, pd.Series):
            y = pd.Series(y, name=target_name)
        else:
            y = y.rename(target_name)

        y_pred = self.predict(X, feature_names=feature_names)
        # Convert to numpy arrays for calculation
        y_true = y.values
        y_pred = y_pred.values if isinstance(y_pred, pd.Series) else y_pred

        # Calculate R^2 (coefficient of determination)
        u = np.sum((y_true - y_pred) ** 2)
        v = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (u / v)
