# FuzzyCocoPython

Python bindings and scikit-learn style estimators for [fuzzycoco](https://github.com/arthurbabey/fuzzycoco),
an evolutionary fuzzy rule learning engine written in C++. This package wraps the C++ core as a Python module
and exposes `FuzzyCocoClassifier` and `FuzzyCocoRegressor` with a familiar fit/predict API.

## Features
- Train fuzzy rule-based classifiers and regressors using `fit`, `predict`, and `score`
- Inspect learned linguistic variables, rules, and activation statistics from Python
- Persist trained estimators with `save`/`load` helpers based on `joblib`
- Ships with a demo notebook (`demo.ipynb`) showing the main API in action

## Installation

Clone the repository, initialize the `fuzzycoco` submodule, and install the Python package in editable mode:

```bash
git clone https://github.com/arthurbabey/fuzzycocopython.git
cd fuzzycocopython
git submodule update --init --recursive
pip install -e .
```

The build requires a C++17 toolchain to compile the bundled bindings. Refer to the
[`fuzzycoco` project](https://github.com/arthurbabey/fuzzycoco) for more background on the underlying engine.

## Quick start

```python
import pandas as pd
from sklearn.datasets import load_iris
from fuzzycocopython import FuzzyCocoClassifier

data = load_iris(as_frame=True)
clf = FuzzyCocoClassifier(random_state=0)
clf.fit(data.data, data.target)

preds = clf.predict(data.data)
score = clf.score(data.data, data.target)

print(f"Accuracy: {score:.3f}")
print(clf.rules_df_.head())
```

For a guided tour, open `demo.ipynb`. Additional usage examples live in `tests/test_fuzzycocopython.py`.

## Contributing

Issues and pull requests are welcome. Please run the test suite (`pytest`) after making changes to ensure
compatibility.
