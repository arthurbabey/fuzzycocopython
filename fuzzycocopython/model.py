# model.py
import pandas as pd
import os
from fuzzycoco_core import DataFrame, CocoScriptRunnerMethod, FuzzyCocoScriptRunner, slurp, NamedList, FuzzySystem
from .params import Params

class FuzzyModel:
    def __init__(self, params: Params):
        self.params = params
        self.model = None

    def fit(self, X: pd.DataFrame, output_filename: str = "fuzzysystem.ffs", script_file: str = ""):
        data_list = X.astype(str).values.tolist()
        cdf = DataFrame(data_list, True)
        runner = CocoScriptRunnerMethod(cdf, self.params.seed, output_filename)
        if script_file:
            script = slurp(script_file)
        else:
            generated_file = self.params.generate_md_file()
            script = slurp(generated_file)
            os.remove(generated_file)
        scripter = FuzzyCocoScriptRunner(runner)
        scripter.evalScriptCode(script)
        self.model = self._load(output_filename)
        return self

    def predict(self, X: pd.DataFrame):
        if self.model is None:
            raise RuntimeError("Model not fitted yet")
        data_list = X.astype(str).values.tolist()
        cdf = DataFrame(data_list, True)
        predictions = self.model.smartPredict(cdf)
        predictions_list = predictions.to_list()
        return pd.DataFrame(predictions_list)

    def score(self, X: pd.DataFrame, y: pd.Series):
        y_pred = self.predict(X).iloc[:, 0]
        return (y_pred == y).mean()

    def evaluate(self, X: pd.DataFrame, y: pd.Series):
        y_pred = self.predict(X).iloc[:, 0]
        return {
            "accuracy": (y_pred == y).mean()
        }

    def save(self, filename: str):
        pass

    def _load(self, filename: str):
        desc = NamedList.parse(filename)
        return FuzzySystem.load(desc.get_list("fuzzy_system"))
