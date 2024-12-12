# simplifying basic import
from .fuzzycoco_base import FuzzyCocoBase
from .fuzzycoco_classifier import FuzzyCocoClassifier
from .fuzzycoco_regressor import FuzzyCocoRegressor
from .params import Params

__all__ = ["FuzzyCocoClassifier", "FuzzyCocoRegressor", "FuzzyCocoBase", "Params"]
