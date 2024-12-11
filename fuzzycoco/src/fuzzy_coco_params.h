#ifndef FUZZY_COCO_PARAMS_H
#define FUZZY_COCO_PARAMS_H

// class CoEvolution;
#include <string>
#include <iostream>
#include <cmath>

using namespace std;
#include "fuzzy_system_metrics.h"

#include "evolution_params.h"


struct VarsParams {
    // the number of fuzzy sets to use for the Membership function associated with the variables
    int nb_sets = 3;
    // the nb of bits to encode the variable index. N.B: for input var, at least one extra bit must be added to account for the DontCar
    int nb_bits_vars = MISSING_DATA_INT;
    // the nb of bits to encode the set index. Directy related to nb_sets
    int nb_bits_sets = MISSING_DATA_INT;
    // the nb of bits to encode the variable position/value. a low value means that the variable range is heavily discretized (can take only few values)
    int nb_bits_pos = MISSING_DATA_INT;

    static int evaluate_nb_bits_for_nb(int nb) {  return nb < 2 ? 0 : ceil(log2(nb)); }
    static int evaluate_nb_bits_vars(int nb_vars) {  return evaluate_nb_bits_for_nb(nb_vars) + 1; }

    // N.B: nb_input_vars comes from the dataset
    void evaluate_missing(int nb_vars);

    bool has_missing() const { return is_na(nb_sets) || is_na(nb_bits_vars) || is_na(nb_bits_vars) || is_na(nb_bits_pos); }

    bool operator!=(const VarsParams& p) const { return ! (*this == p); }
    bool operator==(const VarsParams& p) const;

    friend ostream& operator<<(ostream& out, const VarsParams& p);
};

struct GlobalParams {
    // N.B: default values, either NA --> must be evaluated from other values or given, or a value
    // ================ global =======================
    // the number of rules for inferring the fuzzy system. Note that some rules may be discarded because of the "DontCare" mechanism
    int nb_rules = MISSING_DATA_INT;
    // the number of input variables "slots" using by the antecendents of rules
    // default to all input variables. note that some vars may be discarded because of the "DontCare" mechanism
    int nb_max_var_per_rule = MISSING_DATA_INT;
    // the maximum number of coevolution generations to compute. also cf max_fit
    int max_generations = 100;
    // the fitness theshold to stop the evolution. N.B: 2 means that it will never early-stop
    double max_fitness = 2;
    // the number of cooperators to use to evaluate the fitness in the coevolution algorithm
    int nb_cooperators = 2;

    bool has_missing() const { 
        return is_na(nb_rules) || is_na(nb_max_var_per_rule) || is_na(max_generations) || is_na(max_fitness) || is_na(nb_cooperators); 
    }

    bool operator!=(const GlobalParams& p) const { return ! (*this == p); }
    bool operator==(const GlobalParams& p) const;

    friend ostream& operator<<(ostream& out, const GlobalParams& p);
};

struct FuzzyCocoParams
{
    GlobalParams global_params;
    // variables
    VarsParams input_vars_params;
    VarsParams output_vars_params;
    vector<double> output_vars_defuzz_thresholds;
    bool defuzz_threshold_activated = true;
    // =========== rules pop ===============
    EvolutionParams rules_params;
    // =========== Membership function positions pop
    EvolutionParams mfs_params;
    // weights. N.B: we use a FuzzySystemMetrics struct to store the weights
    // instead of the values
    FuzzySystemMetrics metrics_weights;

    // // N.B: not used any longer
    // bool fixedVars = false;

    // ====================================================================================
    FuzzyCocoParams() {
        // shortcut (N.B: all other values are set to 0)
        FuzzySystemMetrics& w = metrics_weights;
        w.sensitivity = 1.0;
        w.specificity = 0.8;
    }

    void evaluate_missing(int nb_input_vars, int nb_output_vars) {
        input_vars_params.evaluate_missing(nb_input_vars);
        output_vars_params.evaluate_missing(nb_output_vars);
    }

    bool has_missing() const;
    bool operator!=(const FuzzyCocoParams& p) const { return ! (*this == p); }
    bool operator==(const FuzzyCocoParams& p) const;

    friend ostream& operator<<(ostream& out, const FuzzyCocoParams& p);
};

#endif // FUZZY_COCO_PARAMS_H
