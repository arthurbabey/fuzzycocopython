import json
import os
import uuid

import numpy as np
from lfa_toolbox.core.lv.linguistic_variable import LinguisticVariable
from lfa_toolbox.core.mf.lin_piece_wise_mf import (
    LeftShoulderMF,
    RightShoulderMF,
    TriangularMF,
)
from lfa_toolbox.core.mf.singleton_mf import SingletonMF
from lfa_toolbox.core.rules.default_fuzzy_rule import DefaultFuzzyRule
from lfa_toolbox.core.rules.fuzzy_rule import FuzzyRule
from lfa_toolbox.core.rules.fuzzy_rule_element import Antecedent, Consequent


def parse_fuzzy_system(ffs_file):
    """
    Parse a fuzzy system file (JSON format) to extract:
      - a list of LinguisticVariable objects,
      - a list of FuzzyRule objects,
      - a list of DefaultFuzzyRule objects.

    For INPUT variables, membership functions are built as follows:
      - If there is only one set: use a LeftShoulderMF from pos to pos+1.
      - If there are two sets: the first is LeftShoulderMF and the second is RightShoulderMF.
      - If more than two sets: use LeftShoulderMF for the first, RightShoulderMF for the last,
        and TriangularMF for the intermediate sets.

    For OUTPUT variables, membership functions are built as Singletons.

    The antecedent activation function and the implication function are fixed to 'min'
    (equivalent to FIS.AND_min).

    :param ffs_file: Either a file path to the fuzzy system file or a JSON string.
    :return: (linguistic_variables, fuzzy_rules, default_rules)
    """

    def object_pairs_hook_with_duplicates(pairs):
        d = {}
        for key, value in pairs:
            if key in d:
                if isinstance(d[key], list):
                    d[key].append(value)
                else:
                    d[key] = [d[key], value]
            else:
                d[key] = value
        return d

    def create_input_linguistic_variable(var_name, set_items):
        set_items = sorted(set_items, key=lambda s: s.get("position", 0))
        n = len(set_items)
        ling_values_dict = {}

        for i, s in enumerate(set_items):
            pos = s.get("position")
            set_name = s.get("name")

            if n == 1:
                mf = LeftShoulderMF(pos, pos + 1)
            elif n == 2:
                if i == 0:
                    mf = LeftShoulderMF(pos, set_items[i + 1]["position"])
                else:
                    mf = RightShoulderMF(set_items[i - 1]["position"], pos)
            else:
                if i == 0:
                    mf = LeftShoulderMF(pos, set_items[i + 1]["position"])
                elif i == n - 1:
                    mf = RightShoulderMF(set_items[i - 1]["position"], pos)
                else:
                    mf = TriangularMF(
                        set_items[i - 1]["position"], pos, set_items[i + 1]["position"]
                    )

            ling_values_dict[set_name] = mf

        return LinguisticVariable(var_name, ling_values_dict)

    def create_output_linguistic_variable(var_name, set_items):
        """
        Build membership functions for output variables as singletons.
        """
        set_items = sorted(set_items, key=lambda s: s.get("position", 0))
        ling_values_dict = {}
        for s in set_items:
            pos = s.get("position")
            set_name = s.get("name")
            mf = SingletonMF(pos)
            ling_values_dict[set_name] = mf
        return LinguisticVariable(var_name, ling_values_dict)

    # Load JSON
    if isinstance(ffs_file, str):
        if os.path.isfile(ffs_file):
            with open(ffs_file, "r") as f:
                data = json.load(f, object_pairs_hook=object_pairs_hook_with_duplicates)
        else:
            data = json.loads(
                ffs_file, object_pairs_hook=object_pairs_hook_with_duplicates
            )
    else:
        data = ffs_file

    fuzzy_sys = data.get("fuzzy_system", {})
    variables_section = fuzzy_sys.get("variables", {})
    linguistic_variables = []
    lv_dict = {}

    for var_type in ["input", "output"]:
        for var_key, var_data in variables_section.get(var_type, {}).items():
            var_name = var_data.get("name", var_key)
            sets_data = var_data.get("Sets")
            if not sets_data:
                continue
            set_items = sets_data.get("Set")
            if not isinstance(set_items, list):
                set_items = [set_items]

            if var_type == "input":
                lv = create_input_linguistic_variable(var_name, set_items)
            else:  # "output"
                lv = create_output_linguistic_variable(var_name, set_items)

            linguistic_variables.append(lv)
            lv_dict[var_name] = lv

    fixed_act_func = (np.min, "AND_min")
    fixed_impl_func = (np.min, "AND_min")

    fuzzy_rules = []
    rules_section = fuzzy_sys.get("rules", {})
    for rule_name, rule in rules_section.items():
        antecedents_raw = rule.get("antecedents", {}).get("antecedent", {})
        if isinstance(antecedents_raw, list):
            antecedents_list = antecedents_raw
        else:
            antecedents_list = [antecedents_raw]

        consequents_raw = rule.get("consequents", {}).get("consequent", {})
        if isinstance(consequents_raw, list):
            consequents_list = consequents_raw
        else:
            consequents_list = [consequents_raw]

        ants = []
        for ant_data in antecedents_list:
            var_key = ant_data.get("var_name")
            lv = lv_dict.get(var_key, var_key)
            ants.append(
                Antecedent(
                    lv_name=lv,
                    lv_value=ant_data.get("set_name"),
                    is_not=False,
                )
            )

        cons = []
        for cons_data in consequents_list:
            var_key = cons_data.get("var_name")
            lv = lv_dict.get(var_key, var_key)
            cons.append(
                Consequent(
                    lv_name=lv,
                    lv_value=cons_data.get("set_name"),
                )
            )

        frule = FuzzyRule(ants, fixed_act_func, cons, fixed_impl_func)
        fuzzy_rules.append(frule)

    default_rules = []
    default_rules_section = fuzzy_sys.get("default_rules", {})
    for rule_name, rule in default_rules_section.items():
        var_key = rule.get("var_name")
        lv = lv_dict.get(var_key, var_key)
        cons = Consequent(
            lv_name=lv,
            lv_value=rule.get("set_name"),
        )
        dfrule = DefaultFuzzyRule([cons], fixed_impl_func)
        default_rules.append(dfrule)

    return linguistic_variables, fuzzy_rules, default_rules


def generate_fs_file(
    nbRules,
    nbMaxVarPerRule,
    nbOutVars,
    nbInSets,
    nbOutSets,
    inVarsCodeSize,
    outVarsCodeSize,
    inSetsCodeSize,
    outSetsCodeSize,
    inSetsPosCodeSize,
    outSetPosCodeSize,
    maxGenPop1,
    maxFitPop1,
    elitePop1,
    popSizePop1,
    cxProbPop1,
    mutFlipIndPop1,
    mutFlipBitPop1,
    elitePop2,
    popSizePop2,
    cxProbPop2,
    mutFlipIndPop2,
    mutFlipBitPop2,
    sensitivityW,
    specificityW,
    accuracyW,
    ppvW,
    rmseW,
    rrseW,
    raeW,
    mxeW,
    distanceThresholdW,
    distanceMinThresholdW,
    dontCareW,
    overLearnW,
    threshold,
    threshActivated,
):

    unique_id = str(uuid.uuid4())
    filename = f"temp_file_{unique_id}.fs"

    with open(filename, "w") as f:
        f.write('experiment_name = "placeholder";\n')
        f.write('savePath = "placeholder";\n')
        f.write("nbRules = " + str(nbRules) + ";\n")
        f.write("nbMaxVarPerRule = " + str(nbMaxVarPerRule) + ";\n")
        f.write("nbOutVars = " + str(nbOutVars) + ";\n")
        f.write("nbInSets = " + str(nbInSets) + ";\n")
        f.write("nbOutSets = " + str(nbOutSets) + ";\n")
        f.write("inVarsCodeSize = " + str(inVarsCodeSize) + ";\n")
        f.write("outVarsCodeSize = " + str(outVarsCodeSize) + ";\n")
        f.write("inSetsCodeSize = " + str(inSetsCodeSize) + ";\n")
        f.write("outSetsCodeSize = " + str(outSetsCodeSize) + ";\n")
        f.write("inSetsPosCodeSize = " + str(inSetsPosCodeSize) + ";\n")
        f.write("outSetPosCodeSize = " + str(outSetPosCodeSize) + ";\n")
        f.write("maxGenPop1 = " + str(maxGenPop1) + ";\n")
        f.write("maxFitPop1 = " + str(maxFitPop1) + ";\n")
        f.write("elitePop1 = " + str(elitePop1) + ";\n")
        f.write("popSizePop1 = " + str(popSizePop1) + ";\n")
        f.write("cxProbPop1 = " + str(cxProbPop1) + ";\n")
        f.write("mutFlipIndPop1 = " + str(mutFlipIndPop1) + ";\n")
        f.write("mutFlipBitPop1 = " + str(mutFlipBitPop1) + ";\n")
        f.write("elitePop2 = " + str(elitePop2) + ";\n")
        f.write("popSizePop2 = " + str(popSizePop2) + ";\n")
        f.write("cxProbPop2 = " + str(cxProbPop2) + ";\n")
        f.write("mutFlipIndPop2 = " + str(mutFlipIndPop2) + ";\n")
        f.write("mutFlipBitPop2 = " + str(mutFlipBitPop2) + ";\n")
        f.write("sensitivityW = " + str(sensitivityW) + ";\n")
        f.write("specificityW = " + str(specificityW) + ";\n")
        f.write("accuracyW = " + str(accuracyW) + ";\n")
        f.write("ppvW = " + str(ppvW) + ";\n")
        f.write("rmseW = " + str(rmseW) + ";\n")
        f.write("rrseW = " + str(rrseW) + ";\n")
        f.write("raeW = " + str(raeW) + ";\n")
        f.write("mxeW = " + str(mxeW) + ";\n")
        f.write("distanceThresholdW = " + str(distanceThresholdW) + ";\n")
        f.write("distanceMinThresholdW = " + str(distanceMinThresholdW) + ";\n")
        f.write("dontCareW = " + str(dontCareW) + ";\n")
        f.write("overLearnW = " + str(overLearnW) + ";\n")
        f.write("threshold = " + str(threshold) + ";\n")
        f.write("threshActivated = " + str(threshActivated).lower() + ";\n")
        f.write(
            "function doSetParams()\n{\n    setParams(nbRules, nbMaxVarPerRule, nbOutVars, nbInSets, nbOutSets, inVarsCodeSize, outVarsCodeSize, inSetsCodeSize, outSetsCodeSize, inSetsPosCodeSize, outSetPosCodeSize, maxGenPop1, maxFitPop1, elitePop1, popSizePop1, cxProbPop1, mutFlipIndPop1, mutFlipBitPop1, elitePop2, popSizePop2, cxProbPop2, mutFlipIndPop2,mutFlipBitPop2, sensitivityW, specificityW, accuracyW, ppvW, rmseW, rrseW, raeW, mxeW, distanceThresholdW, distanceMinThresholdW, dontCareW, overLearnW, threshold, threshActivated)\n}\n"
        )
        f.write("function doRun()\n{\n    doSetParams();\n    runEvo();\n}\ndoRun()\n")
    return filename
