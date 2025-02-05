import json
import uuid

import pandas as pd


def extract_fuzzy_rules(model):
    """Extract fuzzy rules from a fitted FuzzyCocoClassifier model."""
    if not hasattr(model, "rules_"):
        raise ValueError("Model must be fitted before extracting rules.")

    rule_data = []

    for rule in model.rules_:  # Directly iterate over the list of rules
        rule_data.append(
            {
                "Rule": rule["name"],
                "Antecedent": f"{rule['antecedent']['variable']} IS {rule['antecedent']['set']}",
                "Consequent": f"{rule['consequent']['variable']} IS {rule['consequent']['set']}",
                # "Antecedent Position": rule["antecedent"]["position"],
                # "Consequent Position": rule["consequent"]["position"]
            }
        )

    return pd.DataFrame(rule_data)


def parse_fuzzy_rules(ffs_path):
    """Extract fuzzy rules from a .ffs file."""
    with open(ffs_path, "r") as file:
        data = json.load(file)

    rules = []
    fuzzy_system = data.get("fuzzy_system", {})

    for rule_name, rule_data in fuzzy_system.get("rules", {}).items():
        antecedents = rule_data.get("antecedents", {})
        consequents = rule_data.get("consequents", {})

        antecedent = {
            "variable": antecedents["antecedent"].get("var_name"),
            "set": antecedents["antecedent"].get("set_name"),
            "position": antecedents["antecedent"].get("set_position"),
        }

        consequent = {
            "variable": consequents["consequent"].get("var_name"),
            "set": consequents["consequent"].get("set_name"),
            "position": consequents["consequent"].get("set_position"),
        }

        rules.append(
            {"name": rule_name, "antecedent": antecedent, "consequent": consequent}
        )

    return rules


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
