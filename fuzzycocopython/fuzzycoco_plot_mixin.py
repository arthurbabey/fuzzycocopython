import matplotlib.pyplot as plt
import numpy as np
from lfa_toolbox.core.fis.singleton_fis import SingletonFIS
from lfa_toolbox.view.fis_viewer import FISViewer
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

    def plot_fuzzy_sets(self, **kwargs):
        """
        Plot all membership functions associated with the learned fuzzy variables.
        """
        check_is_fitted(self, ["variables_"])
        for lv in self.variables_:
            fig, ax = plt.subplots()
            ax.set_title(lv.name)
            for label, mf in lv.ling_values.items():
                MembershipFunctionViewer(mf, ax=ax, label=label, **kwargs)
            ax.legend()
            plt.show()

    def plot_fuzzification(self, sample, **kwargs):
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
        check_is_fitted(self, ["variables_", "feature_names_in_", "target_name_in_"])

        # Build a mapping from feature name to crisp value.
        sample_dict = {name: value for name, value in zip(self.feature_names_in_, sample)}
        

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
                mvf = MembershipFunctionViewer(mf, ax=ax, label=label, **kwargs)
                mvf.fuzzify(crisp_value)
            ax.legend()
            plt.show()

    def plot_rule_activations(self, input_sample, figsize=(10, 6)):
        """
        Compute and plot the activation levels for each fuzzy rule for a given input sample.

        Parameters
        ----------
        input_sample : array-like
            A single sample of crisp input values (e.g., X_test[sample_index]).
        feature_names : list of str, optional
            Names corresponding to the input features (not used in this plot).
        """
        import matplotlib.pyplot as plt
        import numpy as np
        from sklearn.utils.validation import check_is_fitted

        # Check if required attributes are fitted
        check_is_fitted(self, ["rules_", "model_"])

        # Ensure the input sample is 2D (one row)
        if hasattr(input_sample, "shape") and len(input_sample.shape) == 1:
            input_sample = input_sample.reshape(1, -1)

        # Compute rule activations using the underlying C++ engine
        activations = self.model_.computeRulesFireLevels(input_sample.tolist()[0])
        # Ensure activations are floats for plotting
        activations = [float(a) for a in activations]

        # Generate labels: "Rule 1", "Rule 2", etc.
        rule_labels = [f"Rule {i+1}" for i in range(len(self.rules_))]

        # Create a bar chart for rule activations
        fig, ax = plt.subplots(figsize=figsize)
        bars = ax.bar(rule_labels, activations, color="skyblue", edgecolor="black")
        ax.set_xlabel("Rules", fontsize=12)
        ax.set_ylabel("Activation Level", fontsize=12)
        ax.set_title("Rule Activations for the Input Sample", fontsize=14)
        ax.tick_params(axis="x", rotation=45)

        # Annotate each bar with its activation value
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),  # vertical offset
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=10,
            )

        plt.tight_layout()
        plt.show()

    def plot_aggregated_output(self, input_sample, figsize=(12, 10)):
        """
        Visualize the aggregated fuzzy output for a given input sample,
        using a SingletonFIS to replicate the C++ singleton-based defuzzification.

        Parameters
        ----------
        input_sample : array-like
            A single sample of crisp input values.
        figsize : tuple, optional
            Size of the figure for the plot.
        """

        check_is_fitted(
            self,
            ["rules_", "variables_", "default_rules_", "model_", "feature_names_in_", "feature_names_in_", "target_name_in_"],
        )

        # Build a mapping for the input values.
        input_dict = {
            name: value for name, value in zip(self.feature_names_in_, input_sample)
        }

        # Retrieve the output linguistic variable
        output_lv = next(
            (lv for lv in self.variables_ if lv.name.upper() == self.target_name_in_.upper()), None
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

        result = fis.predict(input_dict)
        result_cpp = self._predict(input_sample)

        if not np.isclose(float(result.get(self.target_name_in_)), float(result_cpp[0])):
            raise ValueError(
                f"Python and C++ defuzzification results do not match: {result} vs. {result_cpp}"
            )
        # Show the aggregated fuzzy output via FISViewer.
        fisv = FISViewer(fis, figsize=figsize)
        fisv.show()
