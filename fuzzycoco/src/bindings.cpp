#include <pybind11/pybind11.h>
#include <pybind11/stl.h>       // For automatic conversion of STL containers
#include "dataframe.h"
#include "file_utils.h"         // For static utility functions
#include "fuzzy_coco_script_runner.h"
#include "fuzzy_coco.h"         // For ScriptParams if needed
#include "fuzzy_coco_executable.cpp"  // Ensure this is the correct path

namespace py = pybind11;

PYBIND11_MODULE(fuzzycoco_core, m) {
    m.doc() = "Python bindings for the FuzzyCoco project";

    // Bind DataFrame
    py::class_<DataFrame>(m, "DataFrame")
        .def(py::init<int, int>(), "Initialize with number of rows and columns")
        .def(py::init<const std::string&, bool>(), "Initialize from a CSV file with row names flag")
        .def(py::init<const std::vector<std::vector<std::string>>&, bool>(), "Initialize from a 2D vector of strings with row names flag")
        .def("nbcols", &DataFrame::nbcols, "Get the number of columns")
        .def("nbrows", &DataFrame::nbrows, "Get the number of rows")
        .def("subsetColumns", 
             py::overload_cast<int, int>(&DataFrame::subsetColumns, py::const_), 
             "Subset columns by range (col1 to col2)")
        .def("subsetColumns", 
             py::overload_cast<const std::vector<int>&>(&DataFrame::subsetColumns, py::const_), 
             "Subset columns by indices")
        .def("subsetColumns", 
             py::overload_cast<const std::vector<std::string>&>(&DataFrame::subsetColumns, py::const_), 
             "Subset columns by names");

    // Bind FileUtils (namespace functions as static bindings)
    m.def("mkdir_if_needed", &FileUtils::mkdir_if_needed, "Create directory if it doesn't exist");
    m.def("parseCSV", py::overload_cast<const std::string&, std::vector<std::vector<std::string>>&, char>(&FileUtils::parseCSV), 
          "Parse CSV from string content into tokens");
    m.def("parseCSV", py::overload_cast<std::istream&, std::vector<std::vector<std::string>>&, char>(&FileUtils::parseCSV), 
          "Parse CSV from input stream into tokens");
    m.def("parseCSV", py::overload_cast<const std::filesystem::path&, std::vector<std::vector<std::string>>&, char>(&FileUtils::parseCSV), 
          "Parse CSV from file into tokens");
    m.def("writeCSV", &FileUtils::writeCSV, "Write DataFrame to output stream as CSV");
    m.def("slurp", &FileUtils::slurp, "Read entire file content into a string");

    // Bind CocoScriptRunnerMethod
    py::class_<CocoScriptRunnerMethod>(m, "CocoScriptRunnerMethod")
        .def(py::init<const DataFrame&, int, const std::string&>(),
             "Initialize with DataFrame, seed, and output path")
        .def("run", &CocoScriptRunnerMethod::run,
             "Run the fuzzy system");

    // Bind FuzzyCocoScriptRunner
    py::class_<FuzzyCocoScriptRunner>(m, "FuzzyCocoScriptRunner")
        .def(py::init<CocoScriptRunnerMethod&>(), 
             "Initialize with a CocoScriptRunnerMethod instance")
        .def("evalScriptCode", &FuzzyCocoScriptRunner::evalScriptCode, 
             "Evaluate the fuzzy system script from a string");
}
