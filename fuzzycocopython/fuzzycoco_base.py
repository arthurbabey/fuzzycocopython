import os
import subprocess

import pandas as pd
from sklearn.base import BaseEstimator

from .fuzzycoco_core import (
    CocoScriptRunnerMethod,
    DataFrame,
    FuzzyCocoScriptRunner,
    FuzzySystem,
    NamedList,
    slurp,
)
from .params import Params


class FuzzyCocoBase(BaseEstimator):
    def __init__(self, params: Params):
        self.params = params
        self.model_ = None

    def _prepare_data(self, X, y=None, feature_names=None, target_name="OUT"):
        # Handle X
        if not isinstance(X, pd.DataFrame):
            if feature_names is None:
                feature_names = [f"Feature_{i+1}" for i in range(X.shape[1])]
            X = pd.DataFrame(X, columns=feature_names)

        if y is not None:
            # Handle target
            if not isinstance(y, pd.Series):
                y = pd.Series(y, name=target_name)
            else:
                y = y.rename(target_name)
            combined = pd.concat([X, y], axis=1)
        else:
            combined = X

        header = list(combined.columns)
        data_list = [header] + combined.astype(str).values.tolist()
        cdf = DataFrame(data_list, False)
        return cdf

    def _run_script(self, cdf, output_filename, script_file, verbose):
        if script_file:
            script = slurp(script_file)
        else:
            generated_file = self.params.generate_md_file()
            script = slurp(generated_file)
            os.remove(generated_file)

        runner = CocoScriptRunnerMethod(cdf, self.params.seed, output_filename)
        scripter = FuzzyCocoScriptRunner(runner)

        # if verbose:
        # workaround to avoid cerr output from FuzzyCocoScriptRunner::run
        # workaround disable because then fuzzySystem is not saved at all yet by FuzzyCocoScriptRunner::run
        if True:
            scripter.evalScriptCode(script)
        else:
            self._run_in_subprocess(scripter.evalScriptCode, script)

        self.model_ = self._load(output_filename)

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
