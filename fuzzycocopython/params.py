class Params:
    def __init__(
        self,
        seed,
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
    ):
        self.seed = seed
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

    def generate_md_file(self, filename="script.md"):
        with open(filename, "w") as f:
            f.write(
                "/* FUGE-LC Reference script\n     Note: the name of the functions cannot be changed\n*/\n"
            )
            f.write('experiment_name = "Mile";\n')
            f.write('savePath = "/Users/mer/Desktop/Donnee_Karl/results";\n')
            f.write("nbRules = " + str(self.nbRules) + ";\n")
            f.write("nbMaxVarPerRule = " + str(self.nbMaxVarPerRule) + ";\n")
            f.write("nbOutVars = " + str(self.nbOutVars) + ";\n")
            f.write("nbInSets = " + str(self.nbInSets) + ";\n")
            f.write("nbOutSets = " + str(self.nbOutSets) + ";\n")
            f.write("inVarsCodeSize = " + str(self.inVarsCodeSize) + ";\n")
            f.write("outVarsCodeSize = " + str(self.outVarsCodeSize) + ";\n")
            f.write("inSetsCodeSize = " + str(self.inSetsCodeSize) + ";\n")
            f.write("outSetsCodeSize = " + str(self.outSetsCodeSize) + ";\n")
            f.write("inSetsPosCodeSize = " + str(self.inSetsPosCodeSize) + ";\n")
            f.write("outSetPosCodeSize = " + str(self.outSetPosCodeSize) + ";\n")
            f.write("maxGenPop1 = " + str(self.maxGenPop1) + ";\n")
            f.write("maxFitPop1 = " + str(self.maxFitPop1) + ";\n")
            f.write("elitePop1 = " + str(self.elitePop1) + ";\n")
            f.write("popSizePop1 = " + str(self.popSizePop1) + ";\n")
            f.write("cxProbPop1 = " + str(self.cxProbPop1) + ";\n")
            f.write("mutFlipIndPop1 = " + str(self.mutFlipIndPop1) + ";\n")
            f.write("mutFlipBitPop1 = " + str(self.mutFlipBitPop1) + ";\n")
            f.write("elitePop2 = " + str(self.elitePop2) + ";\n")
            f.write("popSizePop2 = " + str(self.popSizePop2) + ";\n")
            f.write("cxProbPop2 = " + str(self.cxProbPop2) + ";\n")
            f.write("mutFlipIndPop2 = " + str(self.mutFlipIndPop2) + ";\n")
            f.write("mutFlipBitPop2 = " + str(self.mutFlipBitPop2) + ";\n")
            f.write("sensitivityW = " + str(self.sensitivityW) + ";\n")
            f.write("specificityW = " + str(self.specificityW) + ";\n")
            f.write("accuracyW = " + str(self.accuracyW) + ";\n")
            f.write("ppvW = " + str(self.ppvW) + ";\n")
            f.write("rmseW = " + str(self.rmseW) + ";\n")
            f.write("rrseW = " + str(self.rrseW) + ";\n")
            f.write("raeW = " + str(self.raeW) + ";\n")
            f.write("mxeW = " + str(self.mxeW) + ";\n")
            f.write("distanceThresholdW = " + str(self.distanceThresholdW) + ";\n")
            f.write(
                "distanceMinThresholdW = " + str(self.distanceMinThresholdW) + ";\n"
            )
            f.write("dontCareW = " + str(self.dontCareW) + ";\n")
            f.write("overLearnW = " + str(self.overLearnW) + ";\n")
            f.write("threshold = " + str(self.threshold) + ";\n")
            f.write("threshActivated = " + str(self.threshActivated).lower() + ";\n")
            f.write(
                "function doSetParams()\n{\n    setParams(nbRules, nbMaxVarPerRule, nbOutVars, nbInSets, nbOutSets, inVarsCodeSize, outVarsCodeSize, inSetsCodeSize, outSetsCodeSize, inSetsPosCodeSize, outSetPosCodeSize, maxGenPop1, maxFitPop1, elitePop1, popSizePop1, cxProbPop1, mutFlipIndPop1, mutFlipBitPop1, elitePop2, popSizePop2, cxProbPop2, mutFlipIndPop2,mutFlipBitPop2, sensitivityW, specificityW, accuracyW, ppvW, rmseW, rrseW, raeW, mxeW, distanceThresholdW, distanceMinThresholdW, dontCareW, overLearnW, threshold, threshActivated)\n}\n"
            )
            f.write(
                "function doRun()\n{\n    doSetParams();\n    runEvo();\n}\ndoRun()\n"
            )
        return filename
