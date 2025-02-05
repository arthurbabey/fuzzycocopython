import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.metrics import r2_score
from sklearn.utils.validation import check_array, check_is_fitted

from .fuzzycoco_base import FuzzyCocoBase
from .utils import parse_fuzzy_rules


class FuzzyCocoRegressor(RegressorMixin, FuzzyCocoBase):
    def fit(
        self,
        X,
        y,
        output_filename: str = "fuzzySystem.ffs",
        feature_names: list = None,
        target_name: str = "OUT",
    ):
        X = check_array(X, ensure_2d=True, dtype="numeric", ensure_all_finite=False)
        y = check_array(y, ensure_2d=False, dtype="numeric", ensure_all_finite=False)
        cdf, combined = self._prepare_data(X, y, feature_names, target_name)
        # Set attributes after data is known
        if y is not None:
            self.feature_names_in_ = combined.columns[:-1].tolist()
            self.n_features_in_ = len(self.feature_names_in_)
        else:
            self.feature_names_in_ = combined.columns.tolist()
            self.n_features_in_ = len(self.feature_names_in_)

        self._run_script(cdf, output_filename)
        self.rules_ = parse_fuzzy_rules(output_filename)
        return self

    def predict(self, X, feature_names: list = None):
        check_is_fitted(self, ["model_", "feature_names_in_", "n_features_in_"])
        X = check_array(X, ensure_2d=True, dtype="numeric", ensure_all_finite=False)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but this model was trained with {self.n_features_in_} features."
            )
        cdf, _ = self._prepare_data(X, None, feature_names)
        predictions = self.model_.smartPredict(cdf)
        predictions_list = predictions.to_list()

        # Convert predictions to floats
        result = [float(row[0]) for row in predictions_list]
        if isinstance(X, pd.DataFrame):
            return pd.Series(result, index=X.index)
        else:
            return np.array(result)

    def score(self, X, y, feature_names: list = None, target_name: str = "OUT"):
        check_is_fitted(self, ["model_", "feature_names_in_", "n_features_in_"])
        if not isinstance(y, pd.Series):
            y = pd.Series(y, name=target_name)
        else:
            y = y.rename(target_name)

        y_pred = self.predict(X, feature_names=feature_names)
        y_true = y.values
        y_pred = y_pred.values if isinstance(y_pred, pd.Series) else y_pred

        return r2_score(y_true, y_pred)
