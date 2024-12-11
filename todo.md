## TODO :

1. Save a fuzzzy system : right now a fuzzysystem is save directly after run aka when FuzzyCocoScriptRunner::evalScript is executed. The python API then directly read that file to store it ; not so pythonic i guess it would be a nice to have a self.save separate from the training itself

2. FuzzyCocoScriptRunner::evalScript yield cerr and prints params ; it would be nice to modulate verbose behaviour directly from python. The workaround yet work to avoid printing but also does not save the .ffs which is atm required

3. Right now python API use CocoScriptRunnerMethod from fuzzy_coco_executable.cpp which was (probably) written mainly for the executable. Maybe good practice would be to declare a similar class that match closer our python needs in a proper file. This class would need to inherit from ScriptRunnerMethod and overide run method

3. To match scikit-learn API one way is to create two classes : classier & regressor. Need to undersand how fuzzycoco is managing that to integrate it in the python API

4. Extensive test on the python code and benchmark time between cpp exec and python api
