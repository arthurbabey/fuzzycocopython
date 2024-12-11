#include <filesystem>
#include "fuzzy_coco.h"
#include "fuzzy_coco_script_runner.h"
#include "logging_logger.h"
#include "file_utils.h"
using namespace logging;

struct Params {
    string datasetFile;
    string scriptFile;
    string fuzzyFile;
    string ouputPath;

    bool verbose = false;
    bool eval = false;
    bool predict = false;
    int seed = -1;
};

/**
  * Displays the command line help.
  */
void showHelp()
{
    cerr << endl << "Valid parameters are :" << endl << endl;
    cerr << " --verbose : Verbose output" << endl << endl;
    cerr << " --evaluate : Perform an evaluation of the given fuzzy system on the specified database" << endl << endl;
    cerr << " --predict : Perform a prediction of the given fuzzy system on the specified database" << endl << endl;
    cerr << " --seed : Seed for the random generator" << endl << endl;
    cerr << "       Value : Seed value" << endl << endl;
    cerr << " -d  : Dataset  (REQUIRED)" << endl;
    cerr << "       Value : Path to the dataset" << endl << endl;
    cerr << " -s  : Script   (REQUIRED for fuzzy system inference)" << endl;
    cerr << "       Value : Path to the execution script" << endl << endl;;
    cerr << " -f  : Fuzzy system   (REQUIRED for evaluation/prediction)" << endl;
    cerr << "       Value : Path to the fuzzy system file" << endl << endl;;
    cerr << " -o  : Output Path" << endl;
    cerr << "       Value : Path to the output file" << endl << endl;
    cerr << " --help : this message" << endl << endl;;
}

/**
  * Prompts the invalid usage error message.
  *
  * @param progName Name of this executable.
  */
void invalidUsage(string progName)
{
    cerr << endl << "ERROR : Invalid parameters format !" << endl << endl;
    cerr << "Usage : " << progName << " -p1 value -p2 value ..." << endl << endl;
    cerr << "For parameter list run with --help" << endl << endl;
}

/**
  * Prompts the invalid parameter error message.
  */
void invalidParam(string param)
{
    cerr << endl << "ERROR : Invalid parameter '" << param << "'!" << endl;
    showHelp();
    exit(1);
}

void missingParamValue(string param)
{
    cerr << endl << "ERROR : parameter '" << param << " requires a value'!" << endl;
    showHelp();
    exit(1);
}

void error(const string& message) {
    cerr << endl << "ERROR :" << message << endl << endl;
    showHelp();
    exit(1);
}

/**
  * Parse the command line arguments.
  *
  * @param args Arguments passed in a StringList structure.
  */
Params parseArguments(const vector<string>& args)
{
    Params params;
    // Look if we run directly from cmdline
    if (args.size() <= 1) {
        return params;
    }

    // Help parameter
    if (args.at(1) == "--help") {
        showHelp();
    }
    int i = 1;
    const int nb = args.size();
    while(i  < nb) {
        // Look for the argument marker : '-'
        const string& arg = args.at(i);

        if (arg.at(0) != '-')  invalidUsage(args[0]);
        if (arg.length() < 2) invalidParam(arg);

        // options that do not take values
        if (arg == "--verbose") {
            params.verbose = true;
        }
        else if (arg == "--evaluate") {
            params.eval = true;
        }
        else if (arg == "--predict") {
            params.predict = true;
        } else { // from there  we need a value
            if (i >= nb - 1) missingParamValue(arg);
                     // Dataset file parameter
            if (arg.at(1) == 'd') {
                params.datasetFile = args[i+1];
            }
                // Script file parameter
            else if (arg.at(1) == 's') {
                params.scriptFile = args[i+1];
            }
                // Fuzzy file parameter
            else if (arg.at(1) == 'f') {
                params.fuzzyFile = args[i+1];
            }

                // Output file parameter
            else if (arg.at(1) == 'o') {
                params.ouputPath = args[i+1];
            }
                // Seed parameter
            else if(arg == "--seed") {
                params.seed = stoi(args.at(i+1));
            } else {
                invalidParam(arg);
            }
            i++; // value param
        }
        i++; // param
    } // while

    return params;
}

void check_file(const string& filename) {
    if (filename.empty()) return;
    if (!filesystem::is_regular_file(filename))  {
        cerr << "ERROR : file '" << filename << "' not found !" << endl << endl;
        error("file not found");
    }
}


void check_params(const Params& params) {
    if (params.eval || params.predict) {
        if (params.eval && params.predict)
            error("you cannot perform both a prediction and a evaluation !");
        if (params.fuzzyFile.empty())
            error("you must specify a fuzzy system to perform a evaluation/prediction !");
        if (params.datasetFile.empty())
            error("you must specify a dataset to perform a evaluation/prediction !");
    } else {
        if (params.datasetFile.empty() || params.scriptFile.empty()) {
            error("you must load a dataset AND a script to compute a FuzzySystem");
        }
    }

    check_file(params.datasetFile);
    check_file(params.scriptFile);
    check_file(params.fuzzyFile);

}

// this the function that will execute fuzzy coco
class CocoScriptRunnerMethod : public ScriptRunnerMethod {
public:
    CocoScriptRunnerMethod(const DataFrame& df, int seed, const path& output_filename)
        : _df(df), _seed(seed), _output_filename(output_filename) {}

    void run(const ScriptParams& params) override {
        logger() << L_time << "CocoScriptRunnerMethod::run()\n";
        const int nb_out_vars = params.nbOutputVars;
        assert(nb_out_vars >= 1);
        DataFrame dfin = _df.subsetColumns(0, _df.nbcols() - nb_out_vars - 1);
        DataFrame dfout = _df.subsetColumns(_df.nbcols() - nb_out_vars, _df.nbcols() - 1);
        // cerr << dfin << dfout;
        // fix defuzz_thresholds: TODO: improve this
        ScriptParams fixed_params = params;
        auto& defuzz_thresholds = fixed_params.coco.output_vars_defuzz_thresholds;
        int nb_thresholds = defuzz_thresholds.size();
        assert(nb_thresholds > 0);
        if (nb_thresholds < nb_out_vars) {
            int nb_missing = nb_out_vars - nb_thresholds;
            for (int i = 0; i < nb_missing; i++) {
                defuzz_thresholds.push_back(defuzz_thresholds.back());
            }
        }

        RandomGenerator rng(_seed);
        FuzzyCoco coco(dfin, dfout, fixed_params.coco, rng);
        logger() << coco;
        // TODO: rename pop1 in rules
        auto gen = coco.run();
        auto desc = coco.describeBestFuzzySystem();

        // save fuzzy system !!
        if (_output_filename.empty()) {
            cout << desc;
        } else {
            FileUtils::mkdir_if_needed(_output_filename);
            fstream output_file(_output_filename, ios::out);
            if (!output_file.is_open()) throw runtime_error(string("unable to open file ") + _output_filename);
            output_file << desc;
        }
    }

private:
    const DataFrame& _df;
    int _seed;
    string _output_filename;
};


void launch(const Params& params) {

    // read dataset
    vector<vector<string>> tokens;
    tokens.reserve(1000);
    FileUtils::parseCSV(path(params.datasetFile), tokens);
    DataFrame df(tokens, true);

    if (params.eval || params.predict) {
        // we need to load a fuzzy system
        ifstream in(params.fuzzyFile);
        NamedList desc = NamedList::parse(in);
        FuzzySystem fs = FuzzySystem::load(desc.get_list("fuzzy_system"));

        const auto& predicted = fs.smartPredict(df);

        if (params.predict) {
            FileUtils::writeCSV(cout, predicted);
        } else {
            auto weights = FuzzySystemMetrics::load(desc.get_list("fitness_metrics_weights"));

            auto thresh_desc = desc.get_list("defuzz_thresholds");
            const int nb = thresh_desc.size();
            vector<double> thresholds(nb);
            for (int i = 0; i < nb; i++)
                thresholds[i] = thresh_desc[i].value().get_double();

            DataFrame actual = df.subsetColumns(predicted.colnames());
            FuzzySystemWeightedFitness fitter(weights);
            FuzzySystemMetricsComputer computer;
            auto metrics = computer.compute(predicted, actual, thresholds);
            double fitness = fitter.fitness(metrics);
            NamedList eval;
            eval.add("fitness", fitness);
            eval.add("metrics", metrics.describe());
            eval.add("weights", weights.describe());
            cout << eval;
        }

    } else {
        // let's compute the fuzzy system using the script manager
        CocoScriptRunnerMethod runner(df, params.seed, params.ouputPath);
        FuzzyCocoScriptRunner scripter(runner);

        string script = FileUtils::slurp(params.scriptFile);

        logger() << "evaluating script...\n";
        scripter.evalScriptCode(script);
    }

}


/**
  * Main function.
  */

int main(int argc, char *argv[])
{
    vector<string> args;
    args.reserve(argc);
    for (int i = 0; i < argc; i++)
        args.push_back({argv[i]});

    Params params = parseArguments(args);
    check_params(params);

    if (params.verbose) logger().activate();
    logger() << L_allwaysFlush << L_time << "Fuzzy Coco started\n";

    launch(params);
}
