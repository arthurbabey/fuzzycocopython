import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.metrics import accuracy_score, r2_score
from sklearn.utils.validation import (
    check_X_y,
    check_array,
    check_is_fitted,
)
from sklearn.metrics import get_scorer

from .utils import _build_cpp_params, _make_dataframe

import math
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import (
    check_X_y,
    check_array,
    check_is_fitted,
)
from sklearn.metrics import accuracy_score

from ._fuzzycoco_core import (
    FuzzyCoco,
    make_rng,
)



class FuzzyCocoBase(BaseEstimator):
    def __init__(self,
        # ---------- global ----------
        nb_rules,
        nb_max_var_per_rule,
        max_generations=100,
        max_fitness=1.0,
        nb_cooperators=2,
        influence_rules_initial_population=False,
        influence_evolving_ratio=0.8,
        # ---------- vars (required bits may be None → auto) ----------
        nb_sets_in=3,
        nb_bits_vars_in=None,
        nb_bits_sets_in=None,
        nb_bits_pos_in=8,   # default 8-bit resolution
        nb_sets_out=3,
        nb_bits_vars_out=None,
        nb_bits_sets_out=None,
        nb_bits_pos_out=8,  # default 8-bit resolution
        # ---------- GA ----------
        rules_pop_size=100,
        mfs_pop_size=100,
        elite_size=5,
        cx_prob=0.5,
        mut_flip_genome=0.5,
        mut_flip_bit=0.025,
        # ---------- fitness ----------
        threshold=0.5,
        metrics_weights=None,
        features_weights=None,
        # ---------- misc ----------
        random_state=None,
    ):
        # store for get_params / cloning
        self.nb_rules = nb_rules
        self.nb_max_var_per_rule = nb_max_var_per_rule
        self.max_generations = max_generations
        self.max_fitness = max_fitness
        self.nb_cooperators = nb_cooperators
        self.influence_rules_initial_population = influence_rules_initial_population
        self.influence_evolving_ratio = influence_evolving_ratio

        self.nb_sets_in  = nb_sets_in
        self.nb_bits_vars_in  = nb_bits_vars_in
        self.nb_bits_sets_in  = nb_bits_sets_in
        self.nb_bits_pos_in   = nb_bits_pos_in

        self.nb_sets_out = nb_sets_out
        self.nb_bits_vars_out = nb_bits_vars_out
        self.nb_bits_sets_out = nb_bits_sets_out
        self.nb_bits_pos_out  = nb_bits_pos_out

        self.rules_pop_size = rules_pop_size
        self.mfs_pop_size   = mfs_pop_size
        self.elite_size     = elite_size
        self.cx_prob        = cx_prob
        self.mut_flip_genome = mut_flip_genome
        self.mut_flip_bit    = mut_flip_bit

        self.threshold = threshold
        self.metrics_weights  = metrics_weights
        self.features_weights = features_weights

        self.random_state = random_state
        self.model_ = None
        
        
    def _prepare_dataframes(self, X, y=None, *, target_name="OUT"):
        X = np.asarray(X, dtype=float)

        if X.ndim == 1:
            X = X.reshape(1, -1)

        if X.shape[1] != len(self.feature_names_in_):
            raise ValueError(
                f"X has {X.shape[1]} features, but expected {len(self.feature_names_in_)}: "
                f"{self.feature_names_in_}"
            )

        header_x = self.feature_names_in_
        dfin = _make_dataframe(X, header_x)

        if y is None:
            return dfin, None

        if isinstance(y, pd.Series):
            y_arr = y.to_numpy(dtype=float).reshape(-1, 1)
        else:
            y_arr = np.asarray(y, dtype=float).reshape(-1, 1)

        if y_arr.shape[0] != X.shape[0]:
            raise ValueError("X and y must have the same number of samples")

        dfout = _make_dataframe(y_arr, [target_name])
        return dfin, dfout

    def _fit(self, X, y, feature_names=None, target_name="OUT"):
        X, y = check_X_y(X, y, dtype=float, ensure_2d=True)
        y = y.reshape(-1, 1)
        
        if X.shape[0] == 0:
            raise ValueError("No samples found in X. At least one sample is required.")
        self.classes_ = np.unique(y)
        
        self.feature_names_in_ = (
            X.columns.tolist() if isinstance(X, pd.DataFrame) else
            feature_names if feature_names is not None else
            [f"Feature_{i+1}" for i in range(X.shape[1])]
        )
        self.target_name_in_ = target_name
        self.n_features_in_ = len(self.feature_names_in_)
        
        params = _build_cpp_params(
            n_features=self.n_features_in_,
            nb_rules=self.nb_rules,
            nb_max_var_per_rule=self.nb_max_var_per_rule,
            max_generations=self.max_generations,
            max_fitness=self.max_fitness,
            nb_cooperators=self.nb_cooperators,
            influence_rules_initial_population=self.influence_rules_initial_population,
            influence_evolving_ratio=self.influence_evolving_ratio,
            nb_sets_in=self.nb_sets_in,
            nb_bits_vars_in=self.nb_bits_vars_in,
            nb_bits_sets_in=self.nb_bits_sets_in,
            nb_bits_pos_in=self.nb_bits_pos_in,
            nb_sets_out=self.nb_sets_out,
            nb_bits_vars_out=self.nb_bits_vars_out,
            nb_bits_sets_out=self.nb_bits_sets_out,
            nb_bits_pos_out=self.nb_bits_pos_out,
            rules_pop_size=self.rules_pop_size,
            mfs_pop_size=self.mfs_pop_size,
            elite_size=self.elite_size,
            cx_prob=self.cx_prob,
            mut_flip_genome=self.mut_flip_genome,
            mut_flip_bit=self.mut_flip_bit,
            threshold=self.threshold,
            metrics_weights=self.metrics_weights,
            features_weights=self.features_weights,
        )

        dfin, dfout = self._prepare_dataframes(X, y)

        rng = make_rng(-1 if self.random_state is None else self.random_state)
        self.model_ = FuzzyCoco(dfin, dfout, params, rng)
        self.model_.run()
        self.model_.select_best()
        return self

    def _predict(self, X):
        check_is_fitted(self, ["model_", "n_features_in_", "feature_names_in_"])
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but the model was trained with {self.model_.n_features} features."
            )
        X = check_array(X, dtype=float, ensure_all_finite=True, ensure_2d=True)
        
        dfin, _ = self._prepare_dataframes(X)
        preds = np.asarray(self.model_.predict(dfin).to_list(), float).ravel()
        return preds

    def score(self, X, y):
        pass



class FuzzyCocoClassifier(FuzzyCocoBase, ClassifierMixin):
    def __init__(self, scoring='accuracy', **kwargs):
        super().__init__(**kwargs)
        self.scoring = scoring
    def fit(self, X, y, feature_names=None, target_name="OUT"):
        super()._fit(X, y, feature_names, target_name)
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        raw_vals = self._predict(X)
        y_pred = np.array([round(val) for val in raw_vals])
        y_pred = np.clip(y_pred, 0, len(self.classes_) - 1)
        return np.asarray(self.classes_)[y_pred]

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

    def score(self, X, y):
        check_is_fitted(self, ["model_", "feature_names_in_", "n_features_in_", "target_name_in_"])
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


class FuzzyCocoRegressor(FuzzyCocoBase, RegressorMixin):
    def __init__(self, scoring='r2', **kwargs):
        super().__init__(**kwargs)
        self.scoring = scoring
    def fit(self, X, y, feature_names=None, target_name="OUT"):
        super()._fit(X, y, feature_names, target_name)
        return self

    def predict(self, X):
        return self._predict(X)

    def score(self, X, y):
        check_is_fitted(self, ["model_", "feature_names_in_", "n_features_in_", "target_name_in_"])
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

