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


def create_input_linguistic_variable(var_name, set_items):
    set_items = sorted(set_items, key=lambda s: s.get("position", 0))
    n = len(set_items)
    ling_values_dict = {}
    for i, s in enumerate(set_items):
        pos = s.get("position")
        # We ignore the original "name" and use the provided generic label.
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
    set_items = sorted(set_items, key=lambda s: s.get("position", 0))
    ling_values_dict = {}
    for s in set_items:
        pos = s.get("position")
        set_name = s.get("name")
        mf = SingletonMF(pos)
        ling_values_dict[set_name] = mf
    return LinguisticVariable(var_name, ling_values_dict)


def generate_generic_labels(n):
    match n:
        case 1:
            return ["Medium"]
        case 2:
            return ["Low", "High"]
        case 3:
            return ["Low", "Medium", "High"]
        case 4:
            return ["Very Low", "Low", "High", "Very High"]
        case 5:
            return ["Very Low", "Low", "Medium", "High", "Very High"]
        case 6:
            return [
                "Very Low",
                "Low",
                "Slightly Low",
                "Slightly High",
                "High",
                "Very High",
            ]
        case 7:
            return [
                "Very Low",
                "Low",
                "Slightly Low",
                "Medium",
                "Slightly High",
                "High",
                "Very High",
            ]
        case _:
            return [f"Set {i+1}" for i in range(n)]


# --- Main helper that parses the fuzzy system from your model ---


def parse_fuzzy_system_from_model(model):
    """
    Given a fuzzy system model (with methods:
       - get_input_variables()
       - get_output_variables()
       - get_rules()
    ),
    parse the fuzzy system and build:
      - a list of LinguisticVariable objects,
      - a list of FuzzyRule objects,
      - a list of DefaultFuzzyRule objects.

    For both input and output variables, the membership set names are replaced with generic labels
    (e.g., "Low"/"High" for 2 sets, "Low"/"Medium"/"High" for 3 sets, etc.).
    The rules are updated to use these generic labels.
    """
    linguistic_variables = []
    lv_dict = {}  # key: variable name, value: LinguisticVariable object
    label_mapping = {}  # key: variable name, value: {original_set_name: generic_label}

    # --- Process input variables ---
    in_vars = (
        model.get_input_variables()
    )  # e.g., [{'name': 'Feature_3', 'sets': [...]}, ...]
    for var in in_vars:
        var_name = var["name"]
        sets = var["sets"]
        generic_labels = generate_generic_labels(len(sets))
        mapping = {}
        new_sets = []
        for label, s in zip(generic_labels, sets):
            orig = s.get("name")
            mapping[orig] = label
            new_s = dict(s)  # copy the original set dictionary
            new_s["name"] = label
            new_sets.append(new_s)
        label_mapping[var_name] = mapping
        lv = create_input_linguistic_variable(var_name, new_sets)
        linguistic_variables.append(lv)
        lv_dict[var_name] = lv

    # --- Process output variables ---
    out_vars = model.get_output_variables()
    for var in out_vars:
        var_name = var["name"]
        sets = var["sets"]
        # Optionally, you may also want generic labels for output variables.
        generic_labels = generate_generic_labels(len(sets))
        mapping = {}
        new_sets = []
        for label, s in zip(generic_labels, sets):
            orig = s.get("name")
            mapping[orig] = label
            new_s = dict(s)
            new_s["name"] = label
            new_sets.append(new_s)
        label_mapping[var_name] = mapping
        lv = create_output_linguistic_variable(var_name, new_sets)
        linguistic_variables.append(lv)
        lv_dict[var_name] = lv

    # --- Process rules ---
    fixed_act_func = (np.min, "AND_min")
    fixed_impl_func = (np.min, "AND_min")
    fuzzy_rules = []
    default_rules = []
    rules_data = (
        model.get_rules()
    )  # e.g., list of dicts with keys "antecedents" and "consequents"
    for rule in rules_data:
        ants_data = rule.get("antecedents", [])
        cons_data = rule.get("consequents", [])
        ants = []
        cons = []
        for ant in ants_data:
            var_name = ant["var_name"]
            orig_set = ant["set_name"]
            # Substitute with the generic label if available:
            set_name = label_mapping.get(var_name, {}).get(orig_set, orig_set)
            lv = lv_dict.get(var_name)
            if lv is not None:
                ants.append(Antecedent(lv, set_name, is_not=False))
        for con in cons_data:
            var_name = con["var_name"]
            orig_set = con["set_name"]
            set_name = label_mapping.get(var_name, {}).get(orig_set, orig_set)
            lv = lv_dict.get(var_name)
            if lv is not None:
                cons.append(Consequent(lv, set_name))
        if ants:
            fuzzy_rules.append(FuzzyRule(ants, fixed_act_func, cons, fixed_impl_func))
        else:
            default_rules.append(DefaultFuzzyRule(cons, fixed_impl_func))

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
