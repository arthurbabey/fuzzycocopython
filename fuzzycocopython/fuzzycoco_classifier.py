import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.metrics import accuracy_score
from sklearn.utils.validation import check_array, check_is_fitted

from .fuzzycoco_base import FuzzyCocoBase
from .utils import extract_fuzzy_rules, parse_fuzzy_rules


class FuzzyCocoClassifier(ClassifierMixin, FuzzyCocoBase):
    def fit(
        self,
        X,
        y,
        output_filename: str = "./fuzzySystem.ffs",
        feature_names: list = None,
        target_name: str = "OUT",
    ):

        # If X is a DataFrame, preserve its columns before calling check_array
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = X.columns.tolist()
        else:
            # If it's not a DataFrame, rely on given feature_names or generate them after check_array
            if feature_names is not None:
                self.feature_names_in_ = feature_names

        # Convert X and y to arrays (loses DataFrame info if any)
        X = check_array(X, ensure_2d=True, dtype="numeric", ensure_all_finite=False)
        y = check_array(y, ensure_2d=False, dtype="numeric", ensure_all_finite=False)

        # If we didn't set feature_names_in_ yet and we have no DataFrame columns:
        if not hasattr(self, "feature_names_in_"):
            if feature_names is not None:
                self.feature_names_in_ = feature_names
            else:
                self.feature_names_in_ = [f"Feature_{i+1}" for i in range(X.shape[1])]

        self.classes_ = np.unique(y)

        # Now that we have feature_names_in_, prepare data
        cdf, combined = self._prepare_data(X, y, self.feature_names_in_, target_name)

        # If y is provided, last column is target
        if y is not None:
            self.n_features_in_ = len(self.feature_names_in_)
        else:
            self.n_features_in_ = len(combined.columns)

        self._run_script(cdf, output_filename)

        # initialize self.rules_ with the parsed rules
        self.rules_ = parse_fuzzy_rules(output_filename)
        return self

    def predict(self, X, feature_names: list = None):
        check_is_fitted(
            self, ["model_", "classes_", "n_features_in_", "feature_names_in_"]
        )

        # Convert X to array
        X = check_array(X, ensure_2d=True, dtype="numeric", ensure_all_finite=False)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but this model was trained with {self.n_features_in_} features."
            )

        # Use stored feature_names_in_ if none given
        if feature_names is None:
            feature_names = self.feature_names_in_

        cdf, _ = self._prepare_data(X, None, feature_names)

        predictions = self.model_.smartPredict(cdf)
        predictions_list = predictions.to_list()
        y_pred = np.array([int(row[0]) for row in predictions_list])

        if not np.all((y_pred >= 0) & (y_pred < len(self.classes_))):
            raise ValueError(
                f"Invalid prediction indices: {y_pred}. Predictions must be in the range [0, {len(self.classes_) - 1}]."
            )

        y_mapped = np.asarray(self.classes_)[y_pred]

        # Return a series with the same index if X was originally a DataFrame with an index
        if isinstance(X, pd.DataFrame):
            return pd.Series(y_mapped, index=X.index, name="predictions")
        else:
            return pd.Series(y_mapped, name="predictions")

    def predict_with_importances(self, X, feature_names=None):
        """Returns predictions and rule activation levels for one or multiple samples."""

        # Get predictions first
        y_mapped = self.predict(X, feature_names=feature_names)

        # Ensure X is a NumPy array
        X = np.asarray(X, dtype=float)

        # Get rule definitions
        rules_df = extract_fuzzy_rules(self)  # Get structured DataFrame of rules

        # Compute rule activations
        if X.ndim == 1:  # Single instance
            fire_levels = self.model_.computeRulesFireLevels(X.tolist())
            rules_df["Activation"] = fire_levels
            return y_mapped, rules_df

        # Multiple samples
        fire_levels = [self.model_.computeRulesFireLevels(row.tolist()) for row in X]
        activations_df = pd.DataFrame(
            fire_levels,
            columns=[f"Activation_sample{i+1}" for i in range(len(fire_levels[0]))],
        )

        return y_mapped, pd.concat([rules_df, activations_df], axis=1)

    def score(self, X, y, feature_names: list = None, target_name: str = "OUT"):
        check_is_fitted(
            self, ["model_", "classes_", "feature_names_in_", "n_features_in_"]
        )

        if not isinstance(y, pd.Series):
            y = pd.Series(y, name=target_name)
        else:
            y = y.rename(target_name)

        y_pred = self.predict(X, feature_names=feature_names)
        return accuracy_score(y, y_pred)

    def _rules(self):
        """Prints the extracted rules in a structured format."""
        rules_df = extract_fuzzy_rules(self)
        print(rules_df)
