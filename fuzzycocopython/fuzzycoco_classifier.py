import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lfa_toolbox.view.mf_viewer import MembershipFunctionViewer
from sklearn.base import ClassifierMixin
from sklearn.metrics import accuracy_score
from sklearn.utils.multiclass import check_classification_targets
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
        output_filename: str = "./fuzzySystem.ffs",
        feature_names: list = None,
        target_name: str = "OUT",
        store_fitness_curve: bool = False,
    ):

        # Validate inputs
        X, y = check_X_y(X, y, dtype=float, ensure_2d=True, ensure_all_finite=True)
        check_classification_targets(y)
        if X.shape[0] == 0:
            raise ValueError("No samples found in X. At least one sample is required.")
        self.classes_ = np.unique(y)

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

        self.variables_, self.rules_, self.default_rules_ = (
            parse_fuzzy_system_from_model(self.model_)
        )
        self._set_logger()
        self._logger.flush()

        if store_fitness_curve:
            log_file = ".log.txt"
            try:
                with open(log_file, "r") as f:
                    lines = f.read().strip().splitlines()
                    if lines:
                        last_line = lines[-1]
                        parts = [s.strip() for s in last_line.split(",") if s.strip()]
                        self.fitness_curve_ = [float(val) for val in parts]
                    else:
                        self.fitness_curve_ = []
            except Exception as e:
                self.fitness_curve_ = []
                print(f"Error reading fitness curve from {log_file}: {e}")

        return self

    def _predict(self, X, feature_names: list = None):
        X_arr = check_array(
            X, dtype="numeric", ensure_all_finite=False, ensure_2d=False
        )
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)

        if X_arr.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X_arr.shape[1]} features, but model was trained with {self.n_features_in_}."
            )

        feat_names = (
            feature_names if feature_names is not None else self.feature_names_in_
        )
        cdf, _ = self._prepare_data(X_arr, None, feat_names)

        preds = self.model_.smartPredict(cdf).to_list()
        return np.array([row[0] for row in preds])

    def predict(self, X):
        check_is_fitted(self, ["model_", "feature_names_in_", "n_features_in_"])
        X = check_array(X, dtype=float, force_all_finite=True, ensure_2d=True)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, "
                f"but this {self.__class__.__name__} is expecting {self.n_features_in_} features as input."
            )
        raw_vals = self._predict(X, self.feature_names_in_)
        y_pred = np.array([round(val) for val in raw_vals])  # Standard rounding
        y_pred = np.clip(
            y_pred, 0, len(self.classes_) - 1
        )  # Ensure indices stay in bounds
        y_mapped = np.asarray(self.classes_)[y_pred]

        return y_mapped

    def score(self, X, y, target_name: str = "OUT"):
        check_is_fitted(
            self, ["model_", "classes_", "feature_names_in_", "n_features_in_"]
        )
        y_series = (
            pd.Series(y, name=target_name)
            if not isinstance(y, pd.Series)
            else y.rename(target_name)
        )
        y_pred = self.predict(X)
        return accuracy_score(y_series, y_pred)

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
            sample_activations = []
            for rule_obj, act in zip(self.rules_, activations):
                rule_dict = {
                    "antecedents": [
                        {"var_name": ant.lv_name.name, "set_name": ant.lv_value}
                        for ant in rule_obj.antecedents
                    ],
                    "consequents": [
                        {"var_name": cons.lv_name.name, "set_name": cons.lv_value}
                        for cons in rule_obj.consequents
                    ],
                    "activation": act,
                }
                sample_activations.append(rule_dict)
            all_rule_activations.append(sample_activations)
        if single_sample:
            return y_pred[0], all_rule_activations[0]
        return y_pred, all_rule_activations
