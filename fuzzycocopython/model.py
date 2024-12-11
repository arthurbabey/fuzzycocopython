import os
import subprocess

import numpy as np
import pandas as pd
from fuzzycoco_core import (
    CocoScriptRunnerMethod,
    DataFrame,
    FuzzyCocoScriptRunner,
    FuzzySystem,
    NamedList,
    slurp,
)
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

from .params import Params


class FuzzyModel(BaseEstimator, ClassifierMixin):
    def __init__(self, params: Params):
        self.params = params
        self.is_fitted = False

    def fit(
        self,
        X,
        y,
        output_filename: str = "fuzzySystem.ffs",
        script_file: str = "",
        verbose: bool = False,
    ):

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if not isinstance(y, pd.Series):
            y = pd.Series(y, name="OUT")
        else:
            y = y.rename("OUT")

        combined = pd.concat([X, y], axis=1)
        # this is done to match requirement of DataFrame constructor
        header = list(combined.columns)
        data_list = [header] + combined.astype(str).values.tolist()
        cdf = DataFrame(data_list, False)

        # use a script file if provided, otherwise generate one
        if script_file:
            script = slurp(script_file)
        else:
            generated_file = self.params.generate_md_file()
            script = slurp(generated_file)
            os.remove(generated_file)

        runner = CocoScriptRunnerMethod(cdf, self.params.seed, output_filename)
        scripter = FuzzyCocoScriptRunner(runner)
        # work around to suppress output from FuzyCocoScriptRunner::run
        # not working yet as fuzzySystem is not save to a file
        if True:
            scripter.evalScriptCode(script)
        else:
            self._run_in_subprocess(scripter.evalScriptCode, script)

        self.model_ = self._load(output_filename)
        return self

    def predict(self, X):
        check_is_fitted(self, "model_")
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        header = list(X.columns)
        data_list = [header] + X.astype(str).values.tolist()
        cdf = DataFrame(data_list, False)

        predictions = self.model_.smartPredict(cdf)
        predictions_list = predictions.to_list()
        # Return as a Series for scikit-learn compatibility
        return pd.Series([row[0] for row in predictions_list], index=X.index)

    def score(self, X, y):
        check_is_fitted(self, "model_")
        y_pred = self.predict(X)
        return (y_pred == y).mean()

    def _load(self, filename: str):
        desc = NamedList.parse(filename)
        return FuzzySystem.load(desc.get_list("fuzzy_system"))

    def _run_in_subprocess(self, func, *args):
        script_path = "/tmp/temp_script.py"
        with open(script_path, "w") as f:
            f.write(
                f"from fuzzycoco_core import *\n"
                f"scripter = {func.__self__.__class__.__name__}(*{args})\n"
                f"scripter.{func.__name__}(*{args})\n"
            )
        subprocess.run(
            ["python", script_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        os.remove(script_path)
