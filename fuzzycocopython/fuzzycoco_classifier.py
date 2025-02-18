import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lfa_toolbox.core.mf.triangular_mf import (
    LeftShoulderMF,
    RightShoulderMF,
    TriangularMF,
)
from lfa_toolbox.view.mf_viewer import MembershipFunctionViewer
from sklearn.base import ClassifierMixin
from sklearn.metrics import accuracy_score
from sklearn.utils.validation import check_array, check_is_fitted

from .fuzzycoco_base import FuzzyCocoBase
from .utils import parse_fuzzy_system


class FuzzyCocoClassifier(ClassifierMixin, FuzzyCocoBase):
    def fit(
        self,
        X,
        y,
        output_filename: str = "./fuzzySystem.ffs",
        feature_names: list = None,
        target_name: str = "OUT",
        store_fitness_curve: bool = False,
    ):
        # Handle feature names from DataFrame or parameter
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = X.columns.tolist()
            print(f"Feature names: {self.feature_names_in_}")
        elif feature_names is not None:
            self.feature_names_in_ = feature_names

        # Validate inputs
        X = check_array(X, ensure_2d=True, dtype="numeric", ensure_all_finite=False)
        y = check_array(y, ensure_2d=False, dtype="numeric", ensure_all_finite=False)

        if not hasattr(self, "feature_names_in_"):
            self.feature_names_in_ = (
                feature_names
                if feature_names is not None
                else [f"Feature_{i+1}" for i in range(X.shape[1])]
            )

        self.classes_ = np.unique(y)
        cdf, combined = self._prepare_data(X, y, self.feature_names_in_, target_name)
        self.n_features_in_ = (
            len(self.feature_names_in_) if y is not None else len(combined.columns)
        )

        self._run_script(cdf, output_filename)
        self.variables_, self.rules_, self.default_rules_ = parse_fuzzy_system(
            output_filename
        )
        self.logger.flush()

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

    def predict(self, X, feature_names: list = None):
        # Accept single sample (1D) or multiple samples (2D)
        X_arr = check_array(
            X, dtype="numeric", ensure_all_finite=False, ensure_2d=False
        )
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)
            single_sample = True
        else:
            single_sample = False

        if X_arr.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X_arr.shape[1]} features, but model was trained with {self.n_features_in_}."
            )

        feat_names = (
            feature_names if feature_names is not None else self.feature_names_in_
        )
        cdf, _ = self._prepare_data(X_arr, None, feat_names)
        predictions = self.model_.smartPredict(cdf).to_list()
        y_pred = np.array([int(row[0]) for row in predictions])
        y_mapped = np.asarray(self.classes_)[y_pred]

        if isinstance(X, pd.DataFrame):
            result = pd.Series(y_mapped, index=X.index, name="predictions")
        else:
            result = pd.Series(y_mapped, name="predictions")

        return result if not single_sample else result.iloc[0]

    def predict_with_importances(self, X, feature_names: list = None):

        # Ensure input is 2D; reshape if single sample.
        arr = check_array(X, dtype=float, ensure_all_finite=False, ensure_2d=False)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
            single_sample = True
        else:
            single_sample = False

        y_pred = self.predict(arr, feature_names=feature_names)
        # Ensure rules are fitted
        check_is_fitted(self, ["rules_"])

        all_activations = []
        for row in arr:
            acts = self.model_.computeRulesFireLevels(row.tolist())
            merged = []
            for rule_obj, act in zip(self.rules_, acts):
                # Build a dictionary representation from the FuzzyRule object.
                rule_dict = {
                    "antecedents": [
                        {"var_name": ant.lv_name.name, "set_name": ant.lv_value}
                        for ant in rule_obj.antecedents
                    ],
                    "consequents": [
                        {"var_name": cons.lv_name.name, "set_name": cons.lv_value}
                        for cons in rule_obj.consequents
                    ],
                }
                rule_dict["activation"] = act
                merged.append(rule_dict)
            all_activations.append(merged)

        if single_sample:
            return y_pred, all_activations[0]
        return y_pred, all_activations

    def score(self, X, y, feature_names: list = None, target_name: str = "OUT"):
        check_is_fitted(
            self, ["model_", "classes_", "feature_names_in_", "n_features_in_"]
        )
        y_series = (
            pd.Series(y, name=target_name)
            if not isinstance(y, pd.Series)
            else y.rename(target_name)
        )
        y_pred = self.predict(X, feature_names=feature_names)
        return accuracy_score(y_series, y_pred)

    def plot_fuzzy_sets(self):
        for lv in self.variables_:
            fig, ax = plt.subplots()
            ax.set_title(lv.name)
            for label, mf in lv.ling_values.items():
                # Display each membership function using the toolbox's viewer.
                MembershipFunctionViewer(mf, ax=ax, label=label)
            ax.legend()
            plt.show()
