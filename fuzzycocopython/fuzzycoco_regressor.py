import os
import tempfile
import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.metrics import get_scorer
from sklearn.utils.validation import (
    check_array,
    check_is_fitted,
    check_random_state,
    check_X_y,
)

from .fuzzycoco_base import FuzzyCocoBase
from .fuzzycoco_plot_mixin import FuzzyCocoPlotMixin
from .utils import parse_fuzzy_system_from_model

class FuzzyCocoRegressor(FuzzyCocoPlotMixin, RegressorMixin, FuzzyCocoBase):
    def __init__(self, scoring='r2', **kwargs):
        super().__init__(scoring=scoring, **kwargs)

    def fit(
        self,
        X,
        y,
        feature_names: list = None,
        target_name: str = None,
    ):
        fd, tmp_ffs = tempfile.mkstemp(suffix=".ffs")
        os.close(fd)

        X, y = check_X_y(X, y, dtype="numeric", ensure_2d=True, ensure_all_finite=True)
        if X.shape[0] == 0:
            raise ValueError("No samples found in X. At least one sample is required.")
        self._rng = check_random_state(self.random_state)

        self.feature_names_in_ = (
            X.columns.tolist() if isinstance(X, pd.DataFrame) else
            feature_names if feature_names is not None else
            [f"Feature_{i+1}" for i in range(X.shape[1])]
        )
        self.target_name_in_ = (
            y.columns.tolist() if isinstance(y, (pd.Series, pd.DataFrame)) else
            target_name if target_name is not None else
            "OUT"
        )
        self.n_features_in_ = len(self.feature_names_in_)

        cdf, _ = self._prepare_data(X, y, self.target_name_in_)
        self._run_script(cdf, tmp_ffs)

        with open(tmp_ffs, "rb") as fh:
            self._ffs_bytes = fh.read()
        os.remove(tmp_ffs)

        self.variables_, self.rules_, self.default_rules_ = (
            parse_fuzzy_system_from_model(self.model_)
        )
        self._is_fitted = True
        return self

    
    def _predict(self, X):
        cdf, _ = self._prepare_data(X, None, None)
        predictions = self.model_.smartPredict(cdf).to_list()
        # Convert predictions to 1D array of shape (n_samples,)
        result = np.array([float(row[0]) for row in predictions])
        return result

    def predict(self, X):
        check_is_fitted(self, ["model_", "feature_names_in_", "n_features_in_"])
        X = check_array(
            X, dtype="numeric", ensure_all_finite=True, ensure_2d=True
        )
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X.shape[1]} features, but expected {self.n_features_in_}.")

        predictions = self._predict(X)
        return predictions

    def score(self, X, y):
        check_is_fitted(self, ["model_", "feature_names_in_", "n_features_in_", "target_name_in_"])
        if not isinstance(y, pd.Series):
            y = pd.Series(y, name=self.target_name_in_)
        else:
            y = y.rename(self.target_name_in_)

        y_pred = self.predict(X)
        y_true = y.values
        
        if isinstance(self.scoring, str):
            scorer = get_scorer(self.scoring)
            return scorer._score_func(
                y_true, y_pred
            )        
        elif callable(self.scoring):
            return self.scoring(y_true, y_pred)
        else:
            raise ValueError(f"Invalid scoring method: {self.scoring}")
        
    def predict_with_importances(self, X):
        check_is_fitted(self, ["rules_", "model_"])
        X_arr = check_array(X, dtype=float, ensure_all_finite=False, ensure_2d=False)
        single_sample = X_arr.ndim == 1
        if single_sample:
            X_arr = X_arr.reshape(1, -1)
        y_pred = self.predict(X_arr)
        all_rule_activations = []
        for row in X_arr:
            activations = self.model_.computeRulesFireLevels(row.tolist())
            all_rule_activations.append(activations)
        if single_sample:
            return y_pred[0], all_rule_activations[0]
        return y_pred, all_rule_activations