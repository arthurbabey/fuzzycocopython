/**
  * @file   fuzzy_coco_script_runner.h
  * @author Karl Forner <karl.forner@gmail.com>
  * @author Lonza
  * @date   09.2024
  * @class FuzzyCoco
  *
  * @brief Base class to execute the javascript Fuzzy Coco script using duktape . Adapted from scriptmanager.h
  * by HEIGH-VD
  */

#ifndef FUZZY_COCO_SCRIPT_RUNNER_H
#define FUZZY_COCO_SCRIPT_RUNNER_H

#include <string>
using namespace std;

#include "fuzzy_coco_params.h"
#include "fuzzy_coco.h"

struct ScriptParams {
  FuzzyCocoParams coco;
  // addititional params
  // the number of output vars in the dataset.
  // by convention the output vars are the last columns from the dataset
  int nbOutputVars = 1;


};

class ScriptRunnerMethod {
public:
  virtual void run(const ScriptParams& params) = 0;
};

// ARTHUR: move CocoScriptRunnerMethod from exec to here to expose it in python, and add a getter for the fitness history
class CocoScriptRunnerMethod : public ScriptRunnerMethod {
  public:
      CocoScriptRunnerMethod(const DataFrame& df, int seed, const std::string& output_filename);
      void run(const ScriptParams& params) override;

      // ARTHUR: Getter to retrieve the fitness history from the trained FuzzyCoco instance.
      const std::vector<double>& getFitnessHistory() const;

  private:
      const DataFrame& _df;
      int _seed;
      std::string _output_filename;
      // NEW: This member will store the FuzzyCoco instance after training.
      std::unique_ptr<FuzzyCoco> _fuzzyCoco;
  };

class FuzzyCocoScriptRunner
{
public:
    FuzzyCocoScriptRunner(ScriptRunnerMethod& runner);
    ~FuzzyCocoScriptRunner();

    void readScript(string fileName);
    // void runScript();
    // bool isScriptReady() const;
    // void evalScriptFile(string fileName);
    void evalScriptCode(const string& code);

    void evalSimpleCodeThatReturnsAString(const string& code);
    void setParams(const ScriptParams& params);

    // main function: to override
    void run();


    ScriptParams _params;
private:
  ScriptRunnerMethod& _runner;
};

#endif // FUZZY_COCO_SCRIPT_RUNNER_H
