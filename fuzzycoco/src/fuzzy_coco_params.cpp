#include <cmath>
#include "fuzzy_coco_params.h"
#include "dataframe.h"

ostream& operator<<(ostream& out, const VarsParams& p) {
    DataFrame df(1, 4);
    df.colnames({"nb_sets", "nb_bits_vars", "nb_bits_sets", "nb_bit_pos"});
    auto D = [](int i) { return is_na(i) ? MISSING_DATA_DOUBLE : double(i); };
    vector<double> row = {D(p.nb_sets), D(p.nb_bits_vars), D(p.nb_bits_sets), D(p.nb_bits_pos)};
    df.fillRow(0, row);
    out << df;

    return out;
}

bool VarsParams::operator==(const VarsParams& p) const {
    return 
        nb_sets == p.nb_sets && 
        nb_bits_vars == p.nb_bits_vars && 
        nb_bits_sets == p.nb_bits_sets &&
        nb_bits_pos == p.nb_bits_pos;
}

void VarsParams::evaluate_missing(int nb_vars) {
    if (is_na(nb_bits_vars))
        nb_bits_vars = evaluate_nb_bits_vars(nb_vars);
    if (is_na(nb_bits_sets))
        nb_bits_sets = evaluate_nb_bits_for_nb(nb_sets);
}

bool GlobalParams::operator==(const GlobalParams& p) const {
    return 
        nb_rules == p.nb_rules && 
        nb_max_var_per_rule == p.nb_max_var_per_rule && 
        max_generations == p.max_generations &&
        max_fitness == p.max_fitness&&
        nb_cooperators == p.nb_cooperators;
}

ostream& operator<<(ostream& out, const GlobalParams& p) {
    DataFrame df(1, 5);
    df.colnames({"nb_rules", "nb_max_var_per_rule", "max_generations", "max_fitness", "nb_cooperators"});
    auto D = [](int i) { return is_na(i) ? MISSING_DATA_DOUBLE : double(i); };
    vector<double> row = {D(p.nb_rules), D(p.nb_max_var_per_rule), D(p.max_generations), D(p.max_fitness), D(p.nb_cooperators)};
    df.fillRow(0, row);
    out << df;

    return out;
}

bool FuzzyCocoParams::has_missing() const {
    return 
        global_params.has_missing() ||
        input_vars_params.has_missing() ||
        output_vars_params.has_missing() ||
        output_vars_defuzz_thresholds.empty() ||
        rules_params.has_missing() ||
        mfs_params.has_missing();
}

bool FuzzyCocoParams::operator==(const FuzzyCocoParams& p) const {
    return 
        global_params == p.global_params &&
        input_vars_params == p.input_vars_params &&
        output_vars_params == p.output_vars_params &&
        output_vars_defuzz_thresholds == p.output_vars_defuzz_thresholds &&
        defuzz_threshold_activated == p.defuzz_threshold_activated &&
        rules_params == p.rules_params &&
        mfs_params == p.mfs_params &&
        metrics_weights == p.metrics_weights;
}

ostream& operator<<(ostream& out, const FuzzyCocoParams& p) {
    out << "Fuzzy Coco Params:" << endl
        << "===================" << endl
        << "Global params:" << endl << p.global_params
        << "Input Variables:" << endl << p.input_vars_params
        << "Output Variables:" << endl << p.output_vars_params
        << "defuzz threshold:" << (p.defuzz_threshold_activated ? "ACTIVATED" : "DISABLED") << endl
        << "output_vars_defuzz_thresholds:" << p.output_vars_defuzz_thresholds << endl
        << "Rules params"  << endl << p.rules_params
        << "MFs (Membership Functions) params"  << endl << p.mfs_params
        << "Fitness Metric Weights:" << endl << p.metrics_weights << endl;
    return out;
}