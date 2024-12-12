import numpy as np
import pandas as pd
import pytest

from fuzzycocopython import FuzzyCocoClassifier, FuzzyCocoRegressor, Params


@pytest.fixture
def random_seed():
    np.random.seed(42)


@pytest.fixture
def classifier_params():
    return Params(seed=42)


@pytest.fixture
def regressor_params():
    return Params(seed=42)


def test_classifier_with_pandas(random_seed, classifier_params, tmp_path):
    # Generate a small classification dataset
    X = pd.DataFrame(np.random.rand(20, 3), columns=["A", "B", "C"])
    y = pd.Series(np.random.randint(0, 2, size=20), name="Target")

    output_filename = tmp_path / "fuzzySystem.ffs"
    model = FuzzyCocoClassifier(classifier_params)  # No output_filename here
    model.fit(X, y, output_filename=str(output_filename))  # Provide it to fit
    preds = model.predict(X)
    score = model.score(X, y)

    assert len(preds) == len(y)
    assert 0.0 <= score <= 1.0


def test_classifier_with_numpy_no_names(random_seed, classifier_params, tmp_path):
    # Generate a small classification dataset
    X = np.random.rand(20, 3)
    y = np.random.randint(0, 2, size=20)
    output_filename = tmp_path / "fuzzySystem.ffs"
    model = FuzzyCocoClassifier(classifier_params)
    # No feature_names, no target_name
    model.fit(X, y, output_filename=str(output_filename))
    preds = model.predict(X)
    score = model.score(X, y)

    assert len(preds) == len(y)
    assert 0.0 <= score <= 1.0


def test_classifier_with_numpy_with_names(random_seed, classifier_params, tmp_path):
    # Generate a small classification dataset
    X = np.random.rand(20, 3)
    y = np.random.randint(0, 2, size=20)
    feature_names = ["Feat1", "Feat2", "Feat3"]
    output_filename = tmp_path / "fuzzySystem.ffs"
    model = FuzzyCocoClassifier(classifier_params)
    model.fit(
        X,
        y,
        feature_names=feature_names,
        target_name="Class",
        output_filename=str(output_filename),
    )
    preds = model.predict(X, feature_names=feature_names)
    score = model.score(X, y, feature_names=feature_names, target_name="Class")

    assert len(preds) == len(y)
    assert 0.0 <= score <= 1.0


def test_regressor_with_pandas(random_seed, regressor_params, tmp_path):
    # Generate a small regression dataset
    X = pd.DataFrame(np.random.rand(20, 3), columns=["A", "B", "C"])
    y = pd.Series(np.random.rand(20), name="Target")
    output_filename = tmp_path / "fuzzySystem.ffs"
    model = FuzzyCocoRegressor(regressor_params)
    model.fit(X, y, output_filename=str(output_filename))
    preds = model.predict(X)
    score = model.score(X, y)

    assert len(preds) == len(y)
    assert isinstance(score, float)


def test_regressor_with_numpy_no_names(random_seed, regressor_params, tmp_path):
    X = np.random.rand(20, 3)
    y = np.random.rand(20)
    output_filename = tmp_path / "fuzzySystem.ffs"
    model = FuzzyCocoRegressor(regressor_params)
    model.fit(X, y, output_filename=str(output_filename))
    preds = model.predict(X)
    score = model.score(X, y)

    assert len(preds) == 20
    assert isinstance(score, float)


def test_regressor_with_numpy_with_names(random_seed, regressor_params, tmp_path):
    X = np.random.rand(20, 3)
    y = np.random.rand(20)
    feature_names = ["Var1", "Var2", "Var3"]
    output_filename = tmp_path / "fuzzySystem.ffs"
    model = FuzzyCocoRegressor(regressor_params)
    model.fit(
        X,
        y,
        feature_names=feature_names,
        target_name="Y",
        output_filename=str(output_filename),
    )
    preds = model.predict(X, feature_names=feature_names)
    score = model.score(X, y, feature_names=feature_names, target_name="Y")

    assert len(preds) == 20
    assert isinstance(score, float)
