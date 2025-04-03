import os
import pickle

import pandas as pd
from sklearn.base import BaseEstimator
from ._fuzzycoco_core import (
    CocoScriptRunnerMethod,
    DataFrame,
    FuzzyCocoScriptRunner,
    FuzzySystem,
    NamedList,
    slurp,
)
from .utils import generate_fs_file


class FuzzyCocoBase(BaseEstimator):
    def __init__(
        self,
        random_state=None,
        scoring='accuracy',
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
        maxGenPop1=100,
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
        verbose=False,
    ):
        self.random_state = random_state
        self.scoring = scoring
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
        self.verbose = verbose

    def _prepare_data(self, X, y=None, target_name="OUT"):

        # Convert X to DataFrame if not already
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names_in_)

        # Combine X and y if y is provided
        if y is not None:
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



    def _run_script(self, cdf, output_filename):

        generated_file = self._generate_fs_files()
        script = slurp(generated_file)
        os.remove(generated_file)

        seed = self._rng.randint(0, 1e6)
        runner = CocoScriptRunnerMethod(cdf, seed, output_filename)
        scripter = FuzzyCocoScriptRunner(runner)

        # Run the script
        scripter.evalScriptCode(script)

        self.fitness_history_ = runner.get_fitness_history()
        self.model_ = self._load(output_filename)

    def _generate_fs_files(self):
        return generate_fs_file(
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



    def __getstate__(self):
        """
        1) Save initialization parameters (so we can call __init__ later).
        2) Copy the full __dict__ but remove the unpickleable model_.
        3) Also remove alpha, model_filename, etc. from the dict copy so
        they don't conflict with the values we pass to __init__.
        """
        state = {}
        # 1) Capture init params
        state['init_params'] = self.get_params()

        # 2) Copy everything from __dict__ as 'attributes'
        #    Then drop unpickleable or conflicting items.
        attributes = self.__dict__.copy()
        if 'model_' in attributes:
            del attributes['model_']  # remove unpickleable object

        # Now 'attributes' still holds e.g. n_features_in_, feature_names_in_, classes_, etc.
        # (Those are all pickleable Python objects.)
        state['attributes'] = attributes
        return state

    def __setstate__(self, state):
        """
        1) Re-run __init__ with init_params.
        2) Restore all the other learned attributes from 'attributes'.
        3) Reload the model_ from the file if it exists.
        """
        # 1) Reconstruct estimator from the saved init params
        init_params = state.get('init_params', {})
        self.__init__(**init_params)

        # 2) Restore everything else (like n_features_in_, feature_names_in_, etc.)
        self.__dict__.update(state.get('attributes', {}))

        # 3) If we have a valid filename, re-load the unpickleable model
        if self.model_filename_:
            self.model_ = self._load(self.model_filename_)
            self._is_fitted = True
        else:
            self._is_fitted = False
            

    def save(self, filepath):
        """Save the estimator (excluding the unpickleable C++ model)."""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    def load_weights(self, filepath):
        with open(filepath, 'rb') as f:
            loaded = pickle.load(f)
        # Copy over learned attributes, but not init
        learned_keys = [k for k in loaded.__dict__ if k not in self.get_params()]
        for k in learned_keys:
            setattr(self, k, getattr(loaded, k))
        return self

    @classmethod
    def load(cls, filepath):
        with open(filepath, 'rb') as f:
            return pickle.load(f)