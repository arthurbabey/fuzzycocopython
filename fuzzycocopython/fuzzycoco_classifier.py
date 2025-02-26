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

    def predict(self, X, feature_names: list = None):
        raw_vals = self._predict(X, feature_names)
        y_pred = np.array([round(val) for val in raw_vals])  # Standard rounding
        y_pred = np.clip(
            y_pred, 0, len(self.classes_) - 1
        )  # Ensure indices stay in bounds
        y_mapped = np.asarray(self.classes_)[y_pred]

        if isinstance(X, pd.DataFrame):
            return pd.Series(y_mapped, index=X.index, name="predictions")
        else:
            return pd.Series(y_mapped, name="predictions")

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

    def plot_fuzzification(self, sample, feature_names=None):
        """
        For each input linguistic variable used in the model, plot its membership functions
        and overlay the fuzzification for the corresponding crisp input value.

        Parameters:
        sample: an array-like of crisp input values (e.g., X_test[sample_index]).
        feature_names: (optional) a list of feature names corresponding to the columns of sample.
                        If not provided, generic names "Feature_1", "Feature_2", ... are assumed.

        Note:
        - The output variable (e.g., with lv.name "OUT") is skipped.
        - Only those fuzzy variables that are used in the model (i.e. appear in self.variables_)
            will be plotted.
        """

        # Build a mapping from feature name to crisp value.
        if feature_names is not None:
            sample_dict = {name: value for name, value in zip(feature_names, sample)}
        else:
            sample_dict = {f"Feature_{i+1}": value for i, value in enumerate(sample)}

        # Iterate over each linguistic variable in the model.
        for lv in self.variables_:
            # Skip the output variable.
            if lv.name.upper() == "OUT":
                continue
            # Check if this variable exists in the sample.
            if lv.name not in sample_dict:
                continue
            crisp_value = sample_dict[lv.name]

            fig, ax = plt.subplots()
            ax.set_title(f"{lv.name} (Input: {crisp_value})")

            # Plot each membership function and overlay the fuzzification.
            for label, mf in lv.ling_values.items():
                mvf = MembershipFunctionViewer(mf, ax=ax, label=label)
                mvf.fuzzify(crisp_value)
            ax.legend()
            plt.show()

    def plot_rule_activations(self, input_sample, feature_names=None):
        """
        Compute and plot the activation levels for each rule for a given input sample.

        Parameters:
            input_sample: array-like, a single sample of crisp input values (e.g., X_test[sample_index]).
            feature_names: (optional) list of feature names corresponding to the input sample.

        This method computes rule activations using self.model_.computeRulesFireLevels,
        matches them to self.rules_, and plots a bar chart showing each rule's activation level.
        """

        # Compute activations using the C++ wrapper.
        activations = self.model_.computeRulesFireLevels(input_sample.tolist())
        print(activations)

        # Generate rule labels; if rules have names, you could use them.
        rule_labels = [f"Rule {i+1}" for i in range(len(self.rules_))]

        # Plot the activations as a bar chart.
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(rule_labels, activations, color="skyblue")
        ax.set_xlabel("Rules")
        ax.set_ylabel("Activation Level")
        ax.set_title("Rule Activations for the Input Sample")
        ax.tick_params(axis="x", rotation=45)
        plt.tight_layout()
        plt.show()

    def plot_aggregated_output(self, input_sample, feature_names=None):
        """
        Visualize the aggregated fuzzy output for a given input sample,
        using a SingletonFIS to replicate the C++ singleton-based defuzzification.

        Parameters:
            input_sample: array-like, a single sample of crisp input values.
            feature_names: Optional list of feature names corresponding to the input.
        """
        import numpy as np
        from lfa_toolbox.core.fis.singleton_fis import SingletonFIS
        from lfa_toolbox.view.fis_viewer import FISViewer

        # Build a mapping for the input values.
        if feature_names is not None:
            input_dict = {
                name: value for name, value in zip(feature_names, input_sample)
            }
        else:
            input_dict = {
                f"Feature_{i+1}": value for i, value in enumerate(input_sample)
            }

        # Retrieve the output linguistic variable from the model (assuming "OUT").
        output_lv = next(
            (lv for lv in self.variables_ if lv.name.upper() == "OUT"), None
        )
        if output_lv is None:
            raise ValueError("Output linguistic variable 'OUT' not found.")

        # Create a SingletonFIS instance using the learned rules and default rule.
        fis = SingletonFIS(
            rules=self.rules_,
            default_rule=(self.default_rules_[0] if self.default_rules_ else None),
        )

        # Compute the prediction (crisp output) for the given input.
        result = fis.predict(input_dict)

        # Compare with the model's own prediction method (which uses the C++ engine) without rounding.
        result_ = self._predict(input_sample)
        print("Result (SingletonFIS):", result)
        print("Result (C++ Model):", result_)

        crisp_output = result.get("OUT")
        print("Crisp Output (SingletonFIS):", crisp_output)

        # Use FISViewer to display the system's aggregated fuzzy output.
        fisv = FISViewer(fis, figsize=(12, 10))
        fisv.show()
