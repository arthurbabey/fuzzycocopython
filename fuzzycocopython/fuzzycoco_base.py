import os
import subprocess

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_array

from .fuzzycoco_core import (
    CocoScriptRunnerMethod,
    DataFrame,
    FuzzyCocoScriptRunner,
    FuzzySystem,
    NamedList,
    slurp,
)
from .utils import generate_md_file


class FuzzyCocoBase(BaseEstimator):
    def __init__(
        self,
        random_state=1,
        nbRules=3,
        nbMaxVarPerRule=3,
        nbOutVars=1,
        nbInSets=2,
        nbOutSets=2,
        inVarsCodeSize=7,
        outVarsCodeSize=1,
        inSetsCodeSize=2,
        outSetsCodeSize=1,
        inSetsPosCodeSize=8,
        outSetPosCodeSize=1,
        maxGenPop1=150,
        maxFitPop1=0.999,
        elitePop1=10,
        popSizePop1=350,
        cxProbPop1=0.9,
        mutFlipIndPop1=0.2,
        mutFlipBitPop1=0.01,
        elitePop2=10,
        popSizePop2=350,
        cxProbPop2=0.6,
        mutFlipIndPop2=0.4,
        mutFlipBitPop2=0.01,
        sensitivityW=1.0,
        specificityW=1.0,
        accuracyW=0.0,
        ppvW=0.0,
        rmseW=0.5,
        rrseW=0.0,
        raeW=0.0,
        mxeW=0.0,
        distanceThresholdW=0.01,
        distanceMinThresholdW=0.0,
        dontCareW=0.35,
        overLearnW=0.0,
        threshold=0.5,
        threshActivated=True,
        script_file=None,
        verbose=False,
    ):

        self.random_state = random_state
        # params of FuzzyCoco
        self.nbRules = nbRules
        self.nbMaxVarPerRule = nbMaxVarPerRule
        self.nbOutVars = nbOutVars
        self.nbInSets = nbInSets
        self.nbOutSets = nbOutSets
        self.inVarsCodeSize = inVarsCodeSize
        self.outVarsCodeSize = outVarsCodeSize
        self.inSetsCodeSize = inSetsCodeSize
        self.outSetsCodeSize = outSetsCodeSize
        self.inSetsPosCodeSize = inSetsPosCodeSize
        self.outSetPosCodeSize = outSetPosCodeSize
        self.maxGenPop1 = maxGenPop1
        self.maxFitPop1 = maxFitPop1
        self.elitePop1 = elitePop1
        self.popSizePop1 = popSizePop1
        self.cxProbPop1 = cxProbPop1
        self.mutFlipIndPop1 = mutFlipIndPop1
        self.mutFlipBitPop1 = mutFlipBitPop1
        self.elitePop2 = elitePop2
        self.popSizePop2 = popSizePop2
        self.cxProbPop2 = cxProbPop2
        self.mutFlipIndPop2 = mutFlipIndPop2
        self.mutFlipBitPop2 = mutFlipBitPop2
        self.sensitivityW = sensitivityW
        self.specificityW = specificityW
        self.accuracyW = accuracyW
        self.ppvW = ppvW
        self.rmseW = rmseW
        self.rrseW = rrseW
        self.raeW = raeW
        self.mxeW = mxeW
        self.distanceThresholdW = distanceThresholdW
        self.distanceMinThresholdW = distanceMinThresholdW
        self.dontCareW = dontCareW
        self.overLearnW = overLearnW
        self.threshold = threshold
        self.threshActivated = threshActivated
        self.script_file = script_file
        self.verbose = verbose

    def _prepare_data(self, X, y=None, feature_names=None, target_name="OUT"):
        # Handle X
        if not isinstance(X, pd.DataFrame):
            X = check_array(X, ensure_2d=True)
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
        return cdf, combined

    def _run_script(self, cdf, output_filename, script_file, verbose):
        if script_file:
            script = slurp(script_file)
        else:
            generated_file = self._generate_md_files()
            script = slurp(generated_file)
            os.remove(generated_file)

        runner = CocoScriptRunnerMethod(cdf, self.random_state, output_filename)
        scripter = FuzzyCocoScriptRunner(runner)

        # if verbose:
        # workaround to avoid cerr output from FuzzyCocoScriptRunner::run
        # workaround disable because then fuzzySystem is not saved at all yet by FuzzyCocoScriptRunner::run
        if True:
            scripter.evalScriptCode(script)
        else:
            self._run_in_subprocess(scripter.evalScriptCode, script)

        self.model_ = self._load(output_filename)

    def _generate_md_files(self):
        return generate_md_file(
            self.nbRules,
            self.nbMaxVarPerRule,
            self.nbOutVars,
            self.nbInSets,
            self.nbOutSets,
            self.inVarsCodeSize,
            self.outVarsCodeSize,
            self.inSetsCodeSize,
            self.outSetsCodeSize,
            self.inSetsPosCodeSize,
            self.outSetPosCodeSize,
            self.maxGenPop1,
            self.maxFitPop1,
            self.elitePop1,
            self.popSizePop1,
            self.cxProbPop1,
            self.mutFlipIndPop1,
            self.mutFlipBitPop1,
            self.elitePop2,
            self.popSizePop2,
            self.cxProbPop2,
            self.mutFlipIndPop2,
            self.mutFlipBitPop2,
            self.sensitivityW,
            self.specificityW,
            self.accuracyW,
            self.ppvW,
            self.rmseW,
            self.rrseW,
            self.raeW,
            self.mxeW,
            self.distanceThresholdW,
            self.distanceMinThresholdW,
            self.dontCareW,
            self.overLearnW,
            self.threshold,
            self.threshActivated,
        )

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
