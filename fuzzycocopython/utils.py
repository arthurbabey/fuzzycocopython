def generate_md_file(
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
    script_filename,
):

    filename = script_filename
    with open(filename, "w") as f:
        f.write(
            "/* FUGE-LC Reference script\n     Note: the name of the functions cannot be changed\n*/\n"
        )
        f.write('experiment_name = "Mile";\n')
        f.write('savePath = "/Users/mer/Desktop/Donnee_Karl/results";\n')
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
