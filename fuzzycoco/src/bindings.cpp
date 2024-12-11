#include <pybind11/pybind11.h>
#include <pybind11/stl.h>       // For automatic conversion of STL containers
#include <pybind11/stl/filesystem.h>
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
        .def("to_list", [](const DataFrame &df) {
            std::vector<std::vector<double>> data;
            for (int row = 0; row < df.nbrows(); row++) {
               data.push_back(df.fetchRow(row));
            }
            return data;
            }, "Convert DataFrame to a list of lists");
             

    // Bind FileUtils (namespace functions as static bindings)
    m.def("parseCSV", py::overload_cast<const std::string&, std::vector<std::vector<std::string>>&, char>(&FileUtils::parseCSV), 
          "Parse CSV from string content into tokens");
    m.def("parseCSV", py::overload_cast<std::istream&, std::vector<std::vector<std::string>>&, char>(&FileUtils::parseCSV), 
          "Parse CSV from input stream into tokens");
    m.def("parseCSV", py::overload_cast<const std::filesystem::path&, std::vector<std::vector<std::string>>&, char>(&FileUtils::parseCSV), 
          "Parse CSV from file into tokens");
    m.def("slurp", &FileUtils::slurp, "Read entire file content into a string");

    // Bind CocoScriptRunnerMethod
    py::class_<CocoScriptRunnerMethod>(m, "CocoScriptRunnerMethod")
        .def(py::init<const DataFrame&, int, const std::string&>(),
             "Initialize with DataFrame, seed, and output path");

    // Bind FuzzyCocoScriptRunner
    py::class_<FuzzyCocoScriptRunner>(m, "FuzzyCocoScriptRunner")
        .def(py::init<CocoScriptRunnerMethod&>(), 
             "Initialize with a CocoScriptRunnerMethod instance")
        .def("evalScriptCode", &FuzzyCocoScriptRunner::evalScriptCode, 
             "Evaluate the fuzzy system script from a string");

     // Bind NamedList 
     py::class_<NamedList>(m, "NamedList")
        .def_static("parse", [](const std::string &file_path) {
            std::ifstream in(file_path);
            if (!in.is_open()) throw std::runtime_error("File not found: " + file_path);
            return NamedList::parse(in);
        }, "Parse a NamedList from a file")
        .def("get_list", &NamedList::get_list, "Get a sublist by name");

     // FuzzySystem bindings
     py::class_<FuzzySystem>(m, "FuzzySystem")
        .def_static("load", &FuzzySystem::load, "Load a FuzzySystem from a NamedList")
        .def("smartPredict", &FuzzySystem::smartPredict, "Perform a smart prediction");
        
}
