# load the compiled c++ extension
# from . import fuzzycoco_core

# simplifying basic import
from .fuzzycoco_base import FuzzyCocoBase
from .fuzzycoco_classifier import FuzzyCocoClassifier
from .fuzzycoco_regressor import FuzzyCocoRegressor

__all__ = ["FuzzyCocoClassifier", "FuzzyCocoRegressor", "FuzzyCocoBase"]
