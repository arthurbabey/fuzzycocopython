#include <pybind11/pybind11.h>
#include <pybind11/stl.h>       // For automatic conversion of STL containers
#include <pybind11/stl/filesystem.h>
#include "dataframe.h"
#include "file_utils.h"         // For static utility functions
#include "fuzzy_coco_script_runner.h"
#include "fuzzy_coco.h"         // For ScriptParams if needed
#include "matrix.h"             // Include Matrix class
#include "fuzzy_system.h"       // Include FuzzySystem
#include <fstream>

namespace py = pybind11;

PYBIND11_MODULE(fuzzycoco_core, m) {
    m.doc() = "Python bindings for the FuzzyCoco project";


    //  Bind DataFrame
    py::class_<DataFrame>(m, "DataFrame")
        //.def(py::init<int, int>(), "Initialize with number of rows and columns")
        //.def(py::init<const std::string&, bool>(), "Initialize from a CSV file with row names flag")
        .def(py::init<const std::vector<std::vector<std::string>>&, bool>(), "Initialize from a 2D vector of strings with row names flag")
        .def("nbcols", &DataFrame::nbcols, "Get the number of columns")
        .def("nbrows", &DataFrame::nbrows, "Get the number of rows")
        .def("to_list", [](const DataFrame &df) {
            std::vector<std::vector<double>> data;
            for (int row = 0; row < df.nbrows(); row++) {
               data.push_back(df.fetchRow(row));
            }
            return data;
            }, "Convert DataFrame to a list of lists");

    //  Bind FileUtils
    m.def("parseCSV", py::overload_cast<const std::string&, std::vector<std::vector<std::string>>&, char>(&FileUtils::parseCSV),
          "Parse CSV from string content into tokens");
    m.def("parseCSV", py::overload_cast<std::istream&, std::vector<std::vector<std::string>>&, char>(&FileUtils::parseCSV),
          "Parse CSV from input stream into tokens");
    m.def("parseCSV", py::overload_cast<const std::filesystem::path&, std::vector<std::vector<std::string>>&, char>(&FileUtils::parseCSV),
          "Parse CSV from file into tokens");
    m.def("slurp", &FileUtils::slurp, "Read entire file content into a string");

    //  Bind CocoScriptRunnerMethod
    py::class_<CocoScriptRunnerMethod>(m, "CocoScriptRunnerMethod")
        .def(py::init<const DataFrame&, int, const std::string&>(),
             "Initialize with DataFrame, seed, and output path")
        .def("get_fitness_history", [](const CocoScriptRunnerMethod &runner) -> std::vector<double> {
        return runner.getFitnessHistory();
    }, "Return the history of computed fitness values for the last training session");


    //  Bind FuzzyCocoScriptRunner
    py::class_<FuzzyCocoScriptRunner>(m, "FuzzyCocoScriptRunner")
        .def(py::init<CocoScriptRunnerMethod&>(),
             "Initialize with a CocoScriptRunnerMethod instance")
        .def("evalScriptCode", &FuzzyCocoScriptRunner::evalScriptCode,
             "Evaluate the fuzzy system script from a string");

    //  Bind NamedList
    py::class_<NamedList>(m, "NamedList")
        .def_static("parse", [](const std::string &file_path) {
            std::ifstream in(file_path);
            if (!in.is_open()) throw std::runtime_error("File not found: " + file_path);
            return NamedList::parse(in);
        }, "Parse a NamedList from a file")
        .def("get_list", &NamedList::get_list, "Get a sublist by name");

    //  Bind Matrix<double>
    py::class_<Matrix<double>>(m, "MatrixDouble")
        .def(py::init<int, int>(), "Initialize Matrix with rows and columns")
        .def("nbrows", &Matrix<double>::nbrows, "Get number of rows")
        .def("nbcols", &Matrix<double>::nbcols, "Get number of columns")
        .def("reset", &Matrix<double>::reset, "Reset all values to zero")
        .def("__repr__", [](const Matrix<double> &mat) {
            std::ostringstream oss;
            oss << mat;
            return oss.str();
        })
        .def("to_list", [](const Matrix<double> &mat) {
            std::vector<std::vector<double>> list(mat.nbrows(), std::vector<double>(mat.nbcols()));
            for (int i = 0; i < mat.nbrows(); i++) {
                for (int j = 0; j < mat.nbcols(); j++) {
                    list[i][j] = mat[i][j];  // Properly access elements
                }
            }
            return list;
        }, "Convert Matrix to a nested list of lists");

    //  Bind the base FuzzySystem class
    py::class_<FuzzySystem>(m, "FuzzySystem")
        .def_static("load", &FuzzySystem::load, "Load a FuzzySystem from a NamedList")
        .def("smartPredict", &FuzzySystem::smartPredict, "Perform a smart prediction")
        .def("computeRulesFireLevels",
             [](FuzzySystem &self, const std::vector<double> &input_values) {
                 return self.computeRulesFireLevels(input_values);
             },
             "Compute and return the firing strengths of each rule.")
        .def("computeRulesImplications",
             [](FuzzySystem &self, const std::vector<double> &rule_fire_levels) {
                 Matrix<double> results(rule_fire_levels.size(), self.getNbRules());
                 self.computeRulesImplications(rule_fire_levels, results);
                 std::vector<std::vector<double>> output;
                 output.reserve(results.nbrows());
                 for (int i = 0; i < results.nbrows(); i++) {
                     std::vector<double> row;
                     row.reserve(results.nbcols());
                     for (int j = 0; j < results.nbcols(); j++) {
                         row.push_back(results[i][j]);
                     }
                     output.push_back(std::move(row));
                 }
                 return output;
             },
             "Compute implications from rule fire levels and return them as a nested list.")
        .def("get_rules", [](FuzzySystem &fs) -> py::list {
             py::list rules;
             int nb_rules = fs.getNbRules();
             auto inputConds = fs.getInputRulesConditions();
             auto outputConds = fs.getOutputRulesConditions();
             for (int i = 0; i < nb_rules; i++) {
                 FuzzyRule rule(fs.getDB());
                 rule.setConditions(inputConds[i], outputConds[i]);
                 NamedList ruleDesc = rule.describe();
                 py::dict d;
                 py::list antecedents;
                 for (auto &ant : ruleDesc.get_list("antecedents")) {
                     py::dict ant_d;
                     ant_d["var_name"] = ant->get_string("var_name");
                     ant_d["set_name"] = ant->get_string("set_name");
                     antecedents.append(ant_d);
                 }
                 d["antecedents"] = antecedents;
                 py::list consequents;
                 for (auto &con : ruleDesc.get_list("consequents")) {
                     py::dict con_d;
                     con_d["var_name"] = con->get_string("var_name");
                     con_d["set_name"] = con->get_string("set_name");
                     consequents.append(con_d);
                 }
                 d["consequents"] = consequents;
                 rules.append(d);
             }
             // Append default rules:
             auto defRules = fs.getDefaultRulesOutputSets();
             int nb_out = fs.getDB().getNbOutputVars();
             for (int var_idx = 0; var_idx < nb_out; var_idx++) {
                 py::dict d;
                 d["antecedents"] = py::list(); // empty antecedents
                 py::list consequents;
                 py::dict con_d;
                 auto outVar = fs.getDB().getOutputVariable(var_idx);
                 con_d["var_name"] = outVar.getName();
                 int set_idx = defRules[var_idx];
                 auto set_ptr = outVar.getSet(set_idx);
                 NamedList setDesc = set_ptr->describe();
                 con_d["set_name"] = setDesc.get_string("name");
                 consequents.append(con_d);
                 d["consequents"] = consequents;
                 rules.append(d);
             }
             return rules;
         }, "Return the fuzzy system rules as a list of dictionaries.")
        .def("get_input_variables", [](FuzzySystem &fs) -> std::vector<py::dict> {
             std::vector<py::dict> result;
             int nb = fs.getDB().getNbInputVars();
             for (int i = 0; i < nb; i++) {
                 auto var = fs.getDB().getInputVariable(i);
                 py::dict d;
                 d["name"] = var.getName();
                 std::vector<py::dict> sets;
                 for (const auto &s : var.getSets()) {
                     py::dict sd;
                     NamedList desc = s.describe();
                     sd["name"] = desc.get_string("name");
                     sd["position"] = desc.get_double("position");
                     sets.push_back(sd);
                 }
                 d["sets"] = sets;
                 result.push_back(d);
             }
             return result;
         }, "Return a list of input variables (with their name and sets) as dictionaries.")
        .def("get_output_variables", [](FuzzySystem &fs) -> std::vector<py::dict> {
             std::vector<py::dict> result;
             int nb = fs.getDB().getNbOutputVars();
             for (int i = 0; i < nb; i++) {
                 auto var = fs.getDB().getOutputVariable(i);
                 py::dict d;
                 d["name"] = var.getName();
                 std::vector<py::dict> sets;
                 for (const auto &s : var.getSets()) {
                     py::dict sd;
                     NamedList desc = s.describe();
                     sd["name"] = desc.get_string("name");
                     sd["position"] = desc.get_double("position");
                     sets.push_back(sd);
                 }
                 d["sets"] = sets;
                 result.push_back(d);
             }
             return result;
         }, "Return a list of output variables (with their name and sets) as dictionaries.");
}
