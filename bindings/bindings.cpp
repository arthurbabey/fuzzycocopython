#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <sstream>

#include "dataframe.h"
#include "fuzzy_coco.h"
#include "fuzzy_coco_params.h"
#include "named_list.h"
#include "random_generator.h"

namespace py = pybind11;
using fuzzy_coco::NamedList;
using fuzzy_coco::Scalar;
using fuzzy_coco::DataFrame;
using fuzzy_coco::GlobalParams;
using fuzzy_coco::VarsParams;
using fuzzy_coco::EvolutionParams;
using fuzzy_coco::FitnessParams;
using fuzzy_coco::FuzzyCocoParams;
using fuzzy_coco::FuzzyCoco;
using fuzzy_coco::RandomGenerator;

// -----------------------------------------------------------------------------
// Helper to convert NamedList recursively to Python objects
// -----------------------------------------------------------------------------
py::object named_list_to_python(const NamedList &nl) {
    if (nl.is_scalar()) {
        const Scalar &v = nl.scalar();
        if (v.is_bool())    return py::bool_(v.get_bool());
        if (v.is_int())     return py::int_(v.get_int());
        if (v.is_double())  return py::float_(v.get_double());
        if (v.is_string())  return py::str(v.get_string());
        return py::none();
    }
    py::dict d;
    for (const auto &child : nl) d[py::str(child.name())] = named_list_to_python(child);
    return d;
}

// -----------------------------------------------------------------------------
PYBIND11_MODULE(_fuzzycoco_core, m) {
    m.doc() = "Python bindings for the FuzzyCoco library";

    // ------------------------ DataFrame ----------------------------------
    py::class_<DataFrame>(m, "DataFrame")
        .def(py::init<const std::vector<std::vector<std::string>>&, bool>())
        .def("nbcols", &DataFrame::nbcols)
        .def("nbrows", &DataFrame::nbrows)
        .def("to_list", [](const DataFrame &df) {
            std::vector<std::vector<double>> rows;
            for (int r = 0; r < df.nbrows(); ++r) rows.push_back(df.fetchRow(r));
            return rows;
        });

    // ------------------------ Parameter structs --------------------------
    py::class_<GlobalParams>(m, "GlobalParams")
        .def(py::init<>())
        .def_readwrite("nb_rules", &GlobalParams::nb_rules)
        .def_readwrite("nb_max_var_per_rule", &GlobalParams::nb_max_var_per_rule)
        .def_readwrite("max_generations", &GlobalParams::max_generations)
        .def_readwrite("max_fitness", &GlobalParams::max_fitness)
        .def_readwrite("nb_cooperators", &GlobalParams::nb_cooperators)
        .def_readwrite("influence_rules_initial_population", &GlobalParams::influence_rules_initial_population)
        .def_readwrite("influence_evolving_ratio", &GlobalParams::influence_evolving_ratio);

    py::class_<VarsParams>(m, "VarsParams")
        .def(py::init<>())
        .def_readwrite("nb_sets", &VarsParams::nb_sets)
        .def_readwrite("nb_bits_vars", &VarsParams::nb_bits_vars)
        .def_readwrite("nb_bits_sets", &VarsParams::nb_bits_sets)
        .def_readwrite("nb_bits_pos", &VarsParams::nb_bits_pos);

    py::class_<EvolutionParams>(m, "EvolutionParams")
        .def(py::init<>())
        .def_readwrite("pop_size", &EvolutionParams::pop_size)
        .def_readwrite("elite_size", &EvolutionParams::elite_size)
        .def_readwrite("cx_prob", &EvolutionParams::cx_prob)
        .def_readwrite("mut_flip_genome", &EvolutionParams::mut_flip_genome)
        .def_readwrite("mut_flip_bit", &EvolutionParams::mut_flip_bit);

    py::class_<FitnessParams>(m, "FitnessParams")
        .def(py::init<>())
        .def("fix_output_thresholds",
            &FitnessParams::fix_output_thresholds,
            py::arg("nb_out_vars"))
        .def_readwrite("output_vars_defuzz_thresholds",
                    &FitnessParams::output_vars_defuzz_thresholds)
        .def_readwrite("metrics_weights", &FitnessParams::metrics_weights)
        .def_readwrite("features_weights", &FitnessParams::features_weights);

    py::class_<FuzzyCocoParams>(m, "FuzzyCocoParams")
        .def(py::init<>())
        .def_readwrite("global_params", &FuzzyCocoParams::global_params)
        .def_readwrite("input_vars_params", &FuzzyCocoParams::input_vars_params)
        .def_readwrite("output_vars_params", &FuzzyCocoParams::output_vars_params)
        .def_readwrite("rules_params", &FuzzyCocoParams::rules_params)
        .def_readwrite("mfs_params", &FuzzyCocoParams::mfs_params)
        .def_readwrite("fitness_params", &FuzzyCocoParams::fitness_params);

    // ------------------------ RandomGenerator ----------------------------
    py::class_<RandomGenerator>(m, "RandomGenerator")
        .def(py::init<uint32_t>(), py::arg("seed"));

    // ------------------------ FuzzyCoco core -----------------------------
    py::class_<FuzzyCoco>(m, "FuzzyCoco")
        .def(py::init<const DataFrame&, const DataFrame&, const FuzzyCocoParams&, RandomGenerator&>(),
             py::keep_alive<1, 2>(),
             py::keep_alive<1, 3>())
        .def("run", [](FuzzyCoco &self) {
            self.run();
        })
        .def("select_best", &FuzzyCoco::selectBestFuzzySystem)
        .def("describe", [](FuzzyCoco &self) {
            return named_list_to_python(self.describeBestFuzzySystem());
        })
        .def("predict", &FuzzyCoco::predict)
        .def("rules_fire_from_values",
             [](FuzzyCoco &self, const std::vector<double>& x) {
                 return self.getFuzzySystem().computeRulesFireLevels(x);
             },
             py::arg("input_values"));
}
