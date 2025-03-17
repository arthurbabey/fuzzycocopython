import matplotlib.pyplot as plt
from lfa_toolbox.view.mf_viewer import MembershipFunctionViewer
from sklearn.utils.validation import check_is_fitted


class FuzzyCocoPlotMixin:
    """
    A mixin class providing plotting methods for FuzzyCoco estimators.

    Requires that the estimator using this mixin define:
      - self.variables_
      - self.rules_
      - self.model_ (if using plot_rule_activations or plot_aggregated_output)
      - self.default_rules_ (if referencing default rules, e.g. in plot_aggregated_output)
      - self._predict (if used in plot_aggregated_output)
    """

    def plot_fuzzy_sets(self):
        """
        Plot all membership functions associated with the learned fuzzy variables.
        """
        check_is_fitted(self, ["variables_"])
        for lv in self.variables_:
            fig, ax = plt.subplots()
            ax.set_title(lv.name)
            for label, mf in lv.ling_values.items():
                MembershipFunctionViewer(mf, ax=ax, label=label)
            ax.legend()
            plt.show()

    def plot_fuzzification(self, sample, feature_names=None):
        """
        For each input linguistic variable used in the model, plot its membership functions
        and overlay the fuzzification for the corresponding crisp input value.

        Parameters
        ----------
        sample : array-like
            Crisp input values (e.g., X_test[sample_index]).
        feature_names : list of str, optional
            Names corresponding to each input feature.
        """
        check_is_fitted(self, ["variables_"])

        # Build a mapping from feature name to crisp value.
        if feature_names is not None:
            sample_dict = {name: value for name, value in zip(feature_names, sample)}
        else:
            sample_dict = {f"Feature_{i+1}": value for i, value in enumerate(sample)}

        # Iterate over each linguistic variable in the model.
        for lv in self.variables_:
            # Skip the output variable if it's named "OUT".
            if lv.name.upper() == "OUT":
                continue

            # Only plot if the variable appears in sample_dict.
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

        Parameters
        ----------
        input_sample : array-like
            A single sample of crisp input values (e.g., X_test[sample_index]).
        feature_names : list of str, optional
            Names corresponding to the input sample.
        """
        check_is_fitted(self, ["rules_", "model_"])

        # If needed, reshape single-sample input:
        if hasattr(input_sample, "shape") and len(input_sample.shape) == 1:
            input_sample = input_sample.reshape(1, -1)

        # Compute rule activations using your underlying C++ engine.
        activations = self.model_.computeRulesFireLevels(input_sample.tolist()[0])

        rule_labels = [f"Rule {i+1}" for i in range(len(self.rules_))]

        # Plot the activations as a bar chart.
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(rule_labels, activations)
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

        Parameters
        ----------
        input_sample : array-like
            A single sample of crisp input values.
        feature_names : list of str, optional
            Names corresponding to the input features.
        """
        import numpy as np
        from lfa_toolbox.core.fis.singleton_fis import SingletonFIS
        from lfa_toolbox.view.fis_viewer import FISViewer

        check_is_fitted(self, ["rules_", "variables_", "default_rules_", "model_"])

        # Build a mapping for the input values.
        if feature_names is not None:
            input_dict = {
                name: value for name, value in zip(feature_names, input_sample)
            }
        else:
            input_dict = {
                f"Feature_{i+1}": value for i, value in enumerate(input_sample)
            }

        # Retrieve the output linguistic variable (assuming "OUT").
        output_lv = next(
            (lv for lv in self.variables_ if lv.name.upper() == "OUT"), None
        )
        if output_lv is None:
            raise ValueError(
                "Output linguistic variable 'OUT' not found in self.variables_."
            )

        # Create a SingletonFIS instance using the learned rules and default rule.
        fis = SingletonFIS(
            rules=self.rules_,
            default_rule=(self.default_rules_[0] if self.default_rules_ else None),
        )

        # Compute the prediction (crisp output) for the given input.
        result = fis.predict(input_dict)

        # Compare with the model's own prediction method (C++ engine) for debugging purposes.
        # NOTE: This assumes you have a method self._predict(...)
        result_ = self._predict(input_sample)

        # Show the aggregated fuzzy output via FISViewer.
        fisv = FISViewer(fis, figsize=(12, 10))
        fisv.show()
