import numpy as np
from sklearn.utils.estimator_checks import parametrize_with_checks

from fuzzycocopython import FuzzyCocoClassifier, FuzzyCocoRegressor


@parametrize_with_checks([FuzzyCocoClassifier()])
def test_sklearn_compatible_classifier(estimator, check):
    check(estimator)


@parametrize_with_checks([FuzzyCocoRegressor()])
def test_sklearn_compatible_regressor(estimator, check):
    check(estimator)


def test_remove_file():
    import os

    os.remove("fuzzySystem.ffs")
    assert not os.path.exists("fuzzySystem.ffs")
