# model.py
import pandas as pd
import os
from fuzzycoco_core import DataFrame, CocoScriptRunnerMethod, FuzzyCocoScriptRunner, slurp
from .params import Params

class FuzzyModel:
    def __init__(self, params: Params):
        self.params = params

    def fit(self, df: pd.DataFrame, output_filename: str = "", script_file: str = ""):
        cdf = DataFrame(df.values.tolist())
        if script_file:
            runner = CocoScriptRunnerMethod(cdf, seed=self.params.seed, output_filename=output_filename)
            script = slurp(script_file)
        else:
            generated_file = self.params.generate_md_file()
            runner = CocoScriptRunnerMethod(cdf, seed=self.params.seed, output_filename=output_filename)
            script = slurp(generated_file)
            os.remove(generated_file)
        scripter = FuzzyCocoScriptRunner(runner)
        scripter.evalScriptCode(script)
        