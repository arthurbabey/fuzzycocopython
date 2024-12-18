import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.metrics import accuracy_score
from sklearn.utils.validation import check_array, check_is_fitted

from .fuzzycoco_base import FuzzyCocoBase


class FuzzyCocoClassifier(ClassifierMixin, FuzzyCocoBase):
    def fit(
        self,
        X,
        y,
        output_filename: str = "./fuzzySystem.ffs",
        script_file: str = "",
        verbose: bool = False,
        feature_names: list = None,
        target_name: str = "OUT",
    ):
        X = check_array(X, ensure_2d=True, dtype="numeric", ensure_all_finite=False)
        y = check_array(y, ensure_2d=False, dtype="numeric", ensure_all_finite=False)

        self.classes_ = np.unique(y)
        cdf, combined = self._prepare_data(X, y, feature_names, target_name)
        if y is not None:
            self.feature_names_in_ = combined.columns[:-1].tolist()
            self.n_features_in_ = len(self.feature_names_in_)
        else:
            self.feature_names_in_ = combined.columns.tolist()
            self.n_features_in_ = len(self.feature_names_in_)

        self._run_script(cdf, output_filename, script_file, verbose)
        return self

    def predict(self, X, feature_names: list = None):
        """Predict class labels for samples in X."""
        # Check if the model and attributes are fitted
        check_is_fitted(self, ["model_", "classes_", "n_features_in_"])

        # Validate input X
        X = check_array(X, ensure_2d=True, dtype="numeric", ensure_all_finite=False)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but this model was trained with {self.n_features_in_} features."
            )

        # Prepare data
        cdf, _ = self._prepare_data(X, None, feature_names)

        # Predict using the model
        predictions = self.model_.smartPredict(cdf)
        predictions_list = predictions.to_list()

        # Ensure predictions are valid integers
        y_pred = np.array([int(row[0]) for row in predictions_list])

        # Validate prediction indices
        if not np.all((y_pred >= 0) & (y_pred < len(self.classes_))):
            raise ValueError(
                f"Invalid prediction indices: {y_pred}. Predictions must be in the range [0, {len(self.classes_) - 1}]."
            )

        # Map predictions to class labels
        y_mapped = np.asarray(self.classes_)[y_pred]

        # Return predictions
        if isinstance(X, pd.DataFrame):
            return pd.Series(y_mapped, index=X.index, name="predictions")
        else:
            return pd.Series(y_mapped, name="predictions")

    def score(self, X, y, feature_names: list = None, target_name: str = "OUT"):
        """Return the mean accuracy of predictions."""
        # Check if the model and classes are fitted
        check_is_fitted(
            self, ["model_", "classes_", "feature_names_in_", "n_features_in_"]
        )

        # Prepare y to ensure alignment
        if not isinstance(y, pd.Series):
            y = pd.Series(y, name=target_name)
        else:
            y = y.rename(target_name)

        y_pred = self.predict(X, feature_names=feature_names)
        return accuracy_score(y, y_pred)
