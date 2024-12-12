import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.utils.validation import check_is_fitted

from .fuzzycoco_base import FuzzyCocoBase


class FuzzyCocoClassifier(FuzzyCocoBase, ClassifierMixin):
    def fit(
        self,
        X,
        y,
        output_filename: str = "fuzzySystem.ffs",
        script_file: str = "",
        verbose: bool = False,
        feature_names: list = None,
        target_name: str = "OUT",
    ):
        cdf = self._prepare_data(X, y, feature_names, target_name)
        self._run_script(cdf, output_filename, script_file, verbose)
        return self

    def predict(self, X, feature_names: list = None):
        check_is_fitted(self, "model_")
        cdf = self._prepare_data(X, None, feature_names)
        predictions = self.model_.smartPredict(cdf)
        predictions_list = predictions.to_list()

        # Return a Series with the same index as X if X is a DataFrame
        if isinstance(X, pd.DataFrame):
            return pd.Series([row[0] for row in predictions_list], index=X.index)
        else:
            return pd.Series([row[0] for row in predictions_list])

    def score(self, X, y, feature_names: list = None, target_name: str = "OUT"):
        check_is_fitted(self, "model_")

        # Prepare y to ensure it aligns with predictions
        if not isinstance(y, pd.Series):
            y = pd.Series(y, name=target_name)
        else:
            y = y.rename(target_name)

        y_pred = self.predict(X, feature_names=feature_names)
        return (y_pred == y).mean()
