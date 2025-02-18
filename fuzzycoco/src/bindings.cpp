#include <pybind11/pybind11.h>
#include <pybind11/stl.h>       // For automatic conversion of STL containers
#include <pybind11/stl/filesystem.h>
#include "dataframe.h"
#include "file_utils.h"         // For static utility functions
#include "fuzzy_coco_script_runner.h"
#include "fuzzy_coco.h"         // For ScriptParams if needed
#include "fuzzy_coco_executable.cpp"  // Ensure this is the correct path
#include "matrix.h"             // Include Matrix class
#include "fuzzy_system.h"       // Include FuzzySystem

namespace py = pybind11;

PYBIND11_MODULE(fuzzycoco_core, m) {
    m.doc() = "Python bindings for the FuzzyCoco project";


    py::class_<logging::Logger>(m, "Logger")
        // activate, deactivate, flush, log, etc.
        .def("activate", &logging::Logger::activate)
        .def("deactivate", [](logging::Logger &self){ self.activate(false); })
        .def("flush", &logging::Logger::flush)
        .def("log", [](logging::Logger &self, const std::string &msg){ self << msg; return &self; },
             py::return_value_policy::reference);

    // The EXACT same logger the C++ code calls:
    m.def("get_logger", &logging::logger,
          py::return_value_policy::reference,
          "Get the global logger instance (which now points to a File_Logger).");

    // ==========================
    //  Bind DataFrame
    // ==========================
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

    // ==========================
    //  Bind FileUtils
    // ==========================
    m.def("parseCSV", py::overload_cast<const std::string&, std::vector<std::vector<std::string>>&, char>(&FileUtils::parseCSV),
          "Parse CSV from string content into tokens");
    m.def("parseCSV", py::overload_cast<std::istream&, std::vector<std::vector<std::string>>&, char>(&FileUtils::parseCSV),
          "Parse CSV from input stream into tokens");
    m.def("parseCSV", py::overload_cast<const std::filesystem::path&, std::vector<std::vector<std::string>>&, char>(&FileUtils::parseCSV),
          "Parse CSV from file into tokens");
    m.def("slurp", &FileUtils::slurp, "Read entire file content into a string");

    // ==========================
    //  Bind CocoScriptRunnerMethod
    // ==========================
    py::class_<CocoScriptRunnerMethod>(m, "CocoScriptRunnerMethod")
        .def(py::init<const DataFrame&, int, const std::string&>(),
             "Initialize with DataFrame, seed, and output path");

    // ==========================
    //  Bind FuzzyCocoScriptRunner
    // ==========================
    py::class_<FuzzyCocoScriptRunner>(m, "FuzzyCocoScriptRunner")
        .def(py::init<CocoScriptRunnerMethod&>(),
             "Initialize with a CocoScriptRunnerMethod instance")
        .def("evalScriptCode", &FuzzyCocoScriptRunner::evalScriptCode,
             "Evaluate the fuzzy system script from a string");

    // ==========================
    //  Bind NamedList
    // ==========================
    py::class_<NamedList>(m, "NamedList")
        .def_static("parse", [](const std::string &file_path) {
            std::ifstream in(file_path);
            if (!in.is_open()) throw std::runtime_error("File not found: " + file_path);
            return NamedList::parse(in);
        }, "Parse a NamedList from a file")
        .def("get_list", &NamedList::get_list, "Get a sublist by name");

    // ==========================
    //  Bind Matrix<double>
    // ==========================
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

    // ==========================
    //  Bind FuzzySystem
    // ==========================
    py::class_<FuzzySystem>(m, "FuzzySystem")
        .def_static("load", &FuzzySystem::load, "Load a FuzzySystem from a NamedList")
        .def("smartPredict", &FuzzySystem::smartPredict, "Perform a smart prediction")
        .def("computeRulesFireLevels",
             [](FuzzySystem &self, const std::vector<double> &input_values) {
                 return self.computeRulesFireLevels(input_values);  // Use correct function signature
             },
             "Compute and return the firing strengths of each rule.")
        .def("computeRulesImplications",
             [](FuzzySystem &self, const std::vector<double> &rule_fire_levels) {
                 Matrix<double> results(rule_fire_levels.size(), self.getNbRules());
                 self.computeRulesImplications(rule_fire_levels, results);

                 // Convert Matrix to Python list of lists
                 std::vector<std::vector<double>> output;
                 for (int i = 0; i < results.nbrows(); i++) {
                     std::vector<double> row;
                     for (int j = 0; j < results.nbcols(); j++) {
                         row.push_back(results[i][j]);  // Access elements correctly
                     }
                     output.push_back(row);
                 }
                 return output;
             },
             "Compute implications from rule fire levels and return them as a nested list.");
}
