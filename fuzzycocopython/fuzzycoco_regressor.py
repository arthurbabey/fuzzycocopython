import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.metrics import r2_score
from sklearn.utils.validation import (
    check_array,
    check_is_fitted,
    check_random_state,
    check_X_y,
)

from .fuzzycoco_base import FuzzyCocoBase
from .utils import parse_fuzzy_system


class FuzzyCocoRegressor(RegressorMixin, FuzzyCocoBase):
    def fit(
        self,
        X,
        y,
        output_filename: str = "fuzzySystem.ffs",
        feature_names: list = None,
        target_name: str = "OUT",
    ):

        # Validate inputs
        X, y = check_X_y(X, y, dtype="numeric", ensure_2d=True, ensure_all_finite=True)

        # handl random state
        self._rng = check_random_state(self.random_state)

        # Handle feature names from DataFrame or parameter
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = X.columns.tolist()
        elif feature_names is not None:
            self.feature_names_in_ = feature_names
        else:
            self.feature_names_in_ = [f"Feature_{i+1}" for i in range(X.shape[1])]
        self.n_features_in_ = len(self.feature_names_in_)

        cdf, combined = self._prepare_data(X, y, target_name)

        self._run_script(cdf, output_filename)
        self.variables_, self.rules_, self.default_rules_ = parse_fuzzy_system(
            output_filename
        )
        return self

    def predict(self, X, feature_names: list = None):
        check_is_fitted(self, ["model_", "feature_names_in_", "n_features_in_"])
        X = check_array(X, dtype=float, ensure_all_finite=True, ensure_2d=True)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, "
                f"but this {self.__class__.__name__} is expecting {self.n_features_in_} features as input."
            )

        cdf, _ = self._prepare_data(X, None, feature_names)
        predictions = self.model_.smartPredict(cdf)
        predictions_list = predictions.to_list()

        # Convert predictions to 1D array of shape (n_samples,)
        result = np.array([float(row[0]) for row in predictions_list])
        return result

    def score(self, X, y, feature_names: list = None, target_name: str = "OUT"):
        check_is_fitted(self, ["model_", "feature_names_in_", "n_features_in_"])
        if not isinstance(y, pd.Series):
            y = pd.Series(y, name=target_name)
        else:
            y = y.rename(target_name)

        y_pred = self.predict(X, feature_names=feature_names)
        y_true = y.values
        y_pred = y_pred.values if isinstance(y_pred, pd.Series) else y_pred

        return r2_score(y_true, y_pred)
