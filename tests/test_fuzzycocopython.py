import numpy as np
import pandas as pd

from fuzzycocopython import FuzzyCocoClassifier, FuzzyCocoRegressor


def test_classifier_with_pandas(tmp_path):
    # Generate a small classification dataset
    X = pd.DataFrame(np.random.rand(20, 3), columns=["A", "B", "C"])
    y = pd.Series(np.random.randint(0, 2, size=20), name="Target")

    output_filename = tmp_path / "fuzzySystem.ffs"
    model = FuzzyCocoClassifier(random_state=123)  # No output_filename here
    model.fit(X, y, output_filename=str(output_filename))  # Provide it to fit
    preds = model.predict(X)
    score = model.score(X, y)

    assert len(preds) == len(y)
    assert 0.0 <= score <= 1.0


def test_classifier_with_numpy_no_names(tmp_path):
    # Generate a small classification dataset
    X = np.random.rand(20, 3)
    y = np.random.randint(0, 2, size=20)
    output_filename = tmp_path / "fuzzySystem.ffs"
    model = FuzzyCocoClassifier(random_state=123)
    model.fit(X, y, output_filename=str(output_filename))
    preds = model.predict(X)
    score = model.score(X, y)

    assert len(preds) == len(y)
    assert 0.0 <= score <= 1.0


def test_classifier_with_numpy_with_names(tmp_path):
    # Generate a small classification dataset
    X = np.random.rand(20, 3)
    y = np.random.randint(0, 2, size=20)
    feature_names = ["Feat1", "Feat2", "Feat3"]
    output_filename = tmp_path / "fuzzySystem.ffs"
    model = FuzzyCocoClassifier(random_state=123)
    model.fit(
        X,
        y,
        feature_names=feature_names,
        target_name="Class",
        output_filename=str(output_filename),
    )
    preds = model.predict(X)
    score = model.score(X, y)
    #model.plot_aggregated_output(X[1])

    assert len(preds) == len(y)
    assert 0.0 <= score <= 1.0


def test_regressor_with_pandas(tmp_path):
    # Generate a small regression dataset
    X = pd.DataFrame(np.random.rand(20, 3), columns=["A", "B", "C"])
    y = pd.Series(np.random.rand(20), name="Target")
    output_filename = tmp_path / "fuzzySystem.ffs"
    model = FuzzyCocoRegressor(random_state=123)
    model.fit(X, y, output_filename=str(output_filename))
    preds = model.predict(X)
    score = model.score(X, y)

    assert len(preds) == len(y)
    assert isinstance(score, float)


def test_regressor_with_numpy_no_names(tmp_path):
    X = np.random.rand(20, 3)
    y = np.random.rand(20)
    output_filename = tmp_path / "fuzzySystem.ffs"
    model = FuzzyCocoRegressor(random_state=123)
    model.fit(X, y, output_filename=str(output_filename))
    preds = model.predict(X)
    score = model.score(X, y)

    assert len(preds) == 20
    assert isinstance(score, float)


def test_regressor_with_numpy_with_names(tmp_path):
    X = np.random.rand(20, 3)
    y = np.random.rand(20)
    feature_names = ["Var1", "Var2", "Var3"]
    output_filename = tmp_path / "fuzzySystem.ffs"
    model = FuzzyCocoRegressor(random_state=123)
    model.fit(
        X,
        y,
        feature_names=feature_names,
        target_name="Y",
        output_filename=str(output_filename),
    )
    preds = model.predict(X)
    score = model.score(X, y, target_name="Y")

    assert len(preds) == 20
    assert isinstance(score, float)
