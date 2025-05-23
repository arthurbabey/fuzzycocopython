
import os
import tempfile
import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.utils.multiclass import check_classification_targets
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


class FuzzyCocoClassifier(FuzzyCocoPlotMixin, ClassifierMixin, FuzzyCocoBase):
    def fit(
        self,
        X,
        y,
        feature_names: list = None,
        target_name: str = None,
    ):
        fd, tmp_ffs = tempfile.mkstemp(suffix=".ffs")
        os.close(fd)

        X, y = check_X_y(X, y, dtype=float, ensure_2d=True, ensure_all_finite=True)
        check_classification_targets(y)
        if X.shape[0] == 0:
            raise ValueError("No samples found in X. At least one sample is required.")
        self.classes_ = np.unique(y)
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
        # here we set self.model_
        self._run_script(cdf, tmp_ffs)

        with open(tmp_ffs, "rb") as fh:
            self._ffs_bytes = fh.read()
        os.remove(tmp_ffs)

        self.variables_, self.rules_, self.default_rules_ = (
            parse_fuzzy_system_from_model(self.model_)
        )
        self._is_fitted = True
        return self


    def _predict(self, X_arr):
        cdf, _ = self._prepare_data(X=X_arr, y=None)
        preds = self.model_.smartPredict(cdf).to_list()
        return np.array([row[0] for row in preds])

    def predict(self, X):
        check_is_fitted(self, ["model_", "feature_names_in_", "n_features_in_"])
        X = check_array(X, dtype=float, ensure_all_finite=True, ensure_2d=True)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        # Verify the number of features matches
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X.shape[1]} features, but expected {self.n_features_in_}.")

        raw_vals = self._predict(X)
        y_pred = np.array([round(val) for val in raw_vals])  # Standard rounding
        y_pred = np.clip(
            y_pred, 0, len(self.classes_) - 1
        )  # Ensure indices stay in bounds
        y_mapped = np.asarray(self.classes_)[y_pred]

        return y_mapped

    def score(self, X, y):
        check_is_fitted(self, ["model_", "classes_", "feature_names_in_", "n_features_in_", "target_name_in_"])
        y_series = (
            pd.Series(y, name=self.target_name_in_)
            if not isinstance(y, pd.Series)
            else y.rename(self.target_name_in_)
        )
        y_pred = self.predict(X)

        # String scorer
        if isinstance(self.scoring, str):
            scorer = get_scorer(self.scoring)
            return scorer._score_func(y_series, y_pred)
        # Callable scorer
        elif callable(self.scoring):
            return self.scoring(y_series, y_pred)
        else:
            raise ValueError(f"Invalid scoring type: {self.scoring}")

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
