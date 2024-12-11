#include "fuzzy_system.h"
#include <unordered_map>

FuzzySystem FuzzySystem::load(const NamedList& desc) {
    const auto& vars = desc.get_list("variables");
    auto db = FuzzyVariablesDB::load(vars);
    FuzzySystem fs(db);

    // transform rules list to ConditionIndexes
    // map var names to index
    unordered_map<string, int> map_in;
    const auto& input_vars = vars.get_list("input");
    const int nb_input_vars = input_vars.size();
    map_in.reserve(nb_input_vars);
    for (int  i = 0; i < nb_input_vars; i++) {
        map_in[input_vars[i].name()] = i;
    }

    unordered_map<string, int> map_out;
    const auto& output_vars = vars.get_list("output");
    const int nb_output_vars = output_vars.size();
    map_out.reserve(nb_output_vars);
    for (int  i = 0; i < nb_output_vars; i++) {
        map_out[output_vars[i].name()] = i;
    }

    const auto& rules = desc.get_list("rules");
    const int nb_rules = rules.size();
    vector<ConditionIndexes> input_cond_lst;
    vector<ConditionIndexes> output_cond_lst;
    input_cond_lst.reserve(nb_rules);
    output_cond_lst.reserve(nb_rules);
    for (int i = 0; i < nb_rules; i++) {
        const auto& rule = rules[i];

        const auto& antecedents = rule.get_list("antecedents");
        ConditionIndexes cis_in;
        cis_in.reserve(antecedents.size());
        for (const auto& ant : antecedents)
            cis_in.push_back({map_in[ant->get_string("var_name")], ant->get_int("set_index")});

        const auto& consequents = rule.get_list("consequents");
        ConditionIndexes cis_out;
        cis_out.reserve(consequents.size());
        for (const auto& cons : consequents)
            cis_out.push_back({map_out[cons->get_string("var_name")], cons->get_int("set_index")});

        input_cond_lst.push_back(move(cis_in));
        output_cond_lst.push_back(move(cis_out));
    }
    
    fs.setRulesConditions(input_cond_lst, output_cond_lst);
  
    // default rules
    const auto& defrules = desc.get_list("default_rules");
    vector<int> default_rules;
    default_rules.resize(defrules.size());
    for (const auto& defrule : defrules) {
        default_rules[map_in[defrule->get_string("var_name")]] = defrule->get_int("set_index");
    }
    fs.setDefaultRulesConditions(default_rules);

    return fs;
}

FuzzySystem::FuzzySystem(const FuzzyVariablesDB& db) 
    : _vars_db(db), _state(db.getNbInputVars(), db.getNbOutputVars(), db.getNbOutputSets()) 
{
    _default_rules_out_sets.resize(_vars_db.getNbOutputVars(), 0);
}



FuzzySystem::FuzzySystem(
        int nb_input_vars, int nb_input_sets, 
        int nb_output_vars, int nb_output_sets) 
    : FuzzySystem(
        build_default_var_names(nb_input_vars, "in"), 
        build_default_var_names(nb_output_vars, "out"),
        nb_input_sets, nb_output_sets) {}

FuzzySystem::FuzzySystem(const vector<string>& input_var_names, const vector<string>& output_var_names, 
int nb_input_sets, int nb_output_sets)
    : 
    _vars_db(input_var_names, nb_input_sets, output_var_names, nb_output_sets),                                                      
    _state(input_var_names.size(), output_var_names.size(), nb_output_sets)
{

    // init the default rules to the first set for each output var
    _default_rules_out_sets.resize(_vars_db.getNbOutputVars(), 0);

}


vector<string> FuzzySystem::build_default_var_names(int nbvars, const string& base_name)
{
  vector<string> names;
  names.reserve(nbvars);
  for (int i = 0; i < nbvars; i++)
    names.push_back(base_name + '_' + to_string(i + 1));
  return names;
}

// N.B: if the set idx of a default rule is out of range --> set it to 0
void FuzzySystem::setDefaultRulesConditions(const vector<int>& out_sets) {
    // need to check the output set indexes, and set to 0 if wrong
    _default_rules_out_sets = out_sets;
    const int nb = getDB().getNbOutputSets();
    for (auto& i : _default_rules_out_sets)
        if (i >= nb) i = 0;
}

void FuzzySystem::setRulesConditions(const vector<ConditionIndexes>& input_cond_lst, const vector<ConditionIndexes>& output_cond_lst)
{
    const int MAX_NB_RULES = input_cond_lst.size();
    assert(input_cond_lst.size() == MAX_NB_RULES);

    // filter out the abnormal rule conditions (out-of-range, repetitions...)
    // and rules which have empty either input or output condictions
    _input_rules_idx.clear();
    _input_rules_idx.reserve(MAX_NB_RULES);
    _output_rules_idx.clear();
    _output_rules_idx.reserve(MAX_NB_RULES);
    const int nb_input_vars = getDB().getNbInputVars();
    const int nb_input_sets = getDB().getNbInputSets();
    const int nb_output_vars = getDB().getNbOutputVars();
    const int nb_output_sets = getDB().getNbOutputSets();

    for (int i = 0; i < MAX_NB_RULES; i++) {
        auto cis_in = FuzzyRule::filterConditionIndexes(nb_input_vars, nb_input_sets, input_cond_lst[i]);
        if (!cis_in.empty()) {
            auto cis_out = FuzzyRule::filterConditionIndexes(nb_output_vars, nb_output_sets, output_cond_lst[i]);
            if (!cis_out.empty()) {
                _input_rules_idx.push_back(cis_in);
                _output_rules_idx.push_back(cis_out);
            }
        }
    }
}

int FuzzySystem::computeTotalInputVarsUsedInRules() const {
    const auto& v = getInputRulesConditions();
    int total = 0;
    for (const auto& cis : v)
        total += cis.size();
    return total;
}

vector<int> FuzzySystem::getUsedInputVariables() const
{
    const int nb = getDB().getNbInputVars();
    const auto& v = getInputRulesConditions();
    vector<bool> used(nb, false);
    for (const auto& cis : v)
        for (const auto& ci: cis)
            used[ci.var_idx] = true;
    vector<int> res;
    res.reserve(nb);
    for (int i = 0; i < nb; i++)
        if (used[i])
            res.push_back(i);
    return res;
}

vector<int> FuzzySystem::getUsedOutputVariables() const
{
    const int nb = getDB().getNbInputVars();
    const auto& v = getOutputRulesConditions();
    vector<bool> used(nb, false);
    for (const auto& cis : v)
        for (const auto& ci: cis)
            used[ci.var_idx] = true;
    vector<int> res;
    res.reserve(nb);
    for (int i = 0; i < nb; i++)
        if (used[i])
            res.push_back(i);
    return res;
}

void FuzzySystem::setVariablesSetPositions(const Matrix<double>& insets_pos_mat, const Matrix<double>& outsets_pos_mat)
{
    getDB().setPositions(insets_pos_mat, outsets_pos_mat);
}

NamedList FuzzySystem::describe() const{
    NamedList desc;

    NamedList params;
    params.add("nb_rules", getNbRules());
    params.add("nb_input_sets", getDB().getNbInputSets());
    params.add("nb_output_sets", getDB().getNbOutputSets());
    desc.add("parameters", params);

    auto db2 = getDB().subset(getUsedInputVariables(), getUsedOutputVariables());
    NamedList vars = db2.describe();
    desc.add("variables", vars);

    NamedList rules;
    const int nb_rules = getNbRules();
    for (int i = 0; i < nb_rules; i++) {
        FuzzyRule rule(getDB());
        rule.setConditions(getInputRulesConditions()[i], getOutputRulesConditions()[i]);
        rules.add("rule" + to_string(i + 1), rule.describe());
    }
    desc.add("rules", rules);

    NamedList default_rules;
    const auto& defs = getDefaultRulesOutputSets();
    const int nb = getDB().getNbOutputVars();
    for (int var_idx = 0; var_idx < nb; var_idx++) {
        NamedList defrule;
        const auto& var = getDB().getOutputVariable(var_idx);
        auto& set = *var.getSet(defs[var_idx]);
        NamedList set_desc = set.describe();
        defrule.add("var_name", var.getName());
        defrule.add("set_name", set_desc.get_string("name"));
        // set indexes are stable, and convenient for load()
        defrule.add("set_index", defs[var_idx]);
        defrule.add("set_position", set_desc.get_double("position"));
        defrule.add("set_pretty", set_desc.get_string("pretty"));
        default_rules.add("rule" + to_string(var_idx + 1), defrule);
    }
    desc.add("default_rules", default_rules);
    
    return desc;
}

void FuzzySystem::printDescription(ostream& out, const NamedList& desc)
{
   const char TAB = '\t';

    out << "FuzzySystem:" << endl;

    auto& params = desc.get_list("parameters");
    out << "## parameters:" << endl;

    out << TAB << "nb rules=" << params.get_int("nb_rules") << endl;
    out << TAB << "nb input sets=" << params.get_int("nb_input_sets")  << endl;
    out << TAB << "nb output sets=" << params.get_int("nb_output_sets") << endl;

    NamedList vars = desc.get_list("variables");
    out << "#" <<  "Used Fuzzy Variables:" << endl;
    NamedList ivars = vars.get_list("input");
    out << "## input variables:" << endl;
    for (const auto& var : ivars) {
        out << TAB;
        FuzzyVariable::printDescription(out, *var);
        out << endl;
    }
    NamedList ovars = vars.get_list("output");
    out << "## output variables:" << endl;
    for (const auto& var : ovars) {
        out << TAB;
        FuzzyVariable::printDescription(out, *var);
        out << endl;
    }

    auto& rules = desc.get_list("rules");
    out << "## rules:" << endl;
    for (auto& rule : rules) {
        out << TAB;
        FuzzyRule::printDescription(out, *rule);
        out << endl;
    }

    auto& defrules = desc.get_list("default_rules");
    out << "## default rules:" << endl;
    for (auto& defrule : defrules) {
        out << TAB << "ELSE " << defrule->get_string("var_name") 
        << " SHOULD BE " << defrule->get_string("set_pretty") << endl;
    }
}

ostream& operator<<(ostream& out, const FuzzySystem& fs) {
    auto desc = fs.describe();
    FuzzySystem::printDescription(out, fs.describe());
    return out;
}


void FuzzySystem::computeRulesImplications(const vector<double>& rule_fire_levels, Matrix<double>& results) const {
    const int nb_rules = getNbRules();
    const int nb_vars = getDB().getNbOutputVars();
    const int nb_sets = getDB().getNbOutputSets();

    assert(results.nbrows() == nb_vars && results.nbcols() == nb_sets);
    assert(rule_fire_levels.size() == nb_rules);

    results.reset(); // important since we add the levels

    for (int rule_idx = 0; rule_idx < nb_rules; rule_idx++) {
        const double fire_level = rule_fire_levels[rule_idx];
        // if the firelevel is missing the rule does not fire, thus it is ignored
        if (is_na(fire_level)) continue;

        const int nb = getOutputRulesConditions()[rule_idx].size();
        for (int i = 0; i <  nb; i++) {
            const auto ci = getOutputRulesConditions()[rule_idx][i];
            results[ci.var_idx][ci.set_idx] += fire_level;
        }
            
    }
}

void FuzzySystem::addDefaultRulesImplications(const vector<int>& default_rules_set_idx,  const vector<double>& outvars_max_fire_levels, Matrix<double>& results) const {
    const int nb_out_vars = getDB().getNbOutputVars();

    assert(default_rules_set_idx.size() == nb_out_vars);
    assert(outvars_max_fire_levels.size() == nb_out_vars);
    assert(results.nbrows() == nb_out_vars && results.nbcols() == getDB().getNbOutputSets());

    for (int var_idx = 0; var_idx < nb_out_vars; var_idx++) {
        const int set_idx = default_rules_set_idx[var_idx];
        const double max_fire_level = outvars_max_fire_levels[var_idx];
        double current_level = results[var_idx][set_idx];
        assert(!is_na(current_level));
        if (!is_na(max_fire_level)) {
            // N.B: only use the value if it is not missing.
            results[var_idx][set_idx] += 1 - max_fire_level;
        }
    }
}


// N.B: put results in fire_levels[]
void FuzzySystem::computeRulesFireLevels(int sample_idx, const DataFrame& df, vector<double>& fire_levels) const {
    assert(df.nbcols() == getDB().getNbInputVars());
    assert(sample_idx >= 0 && sample_idx < df.nbrows());

    const int nb_rules = getNbRules();
    fire_levels.clear();
    fire_levels.reserve(nb_rules);

    for (int rule_idx = 0; rule_idx < nb_rules; rule_idx++) {
        // double fire = getRule(rule_idx).evaluateFireLevel(df, sample_idx);
        double fire = FuzzyRule::evaluateFireLevel(getDB(), getInputRulesConditions()[rule_idx], df, sample_idx);
        fire_levels.push_back(fire);
    }
}
// N.B: mostly for testing and convenience purposes
vector<double> FuzzySystem::computeRulesFireLevels(const vector<double>& input_values) const {
    vector<double> fire_levels;
    fire_levels.reserve(getNbRules());
    for (int rule_idx = 0; rule_idx < getNbRules(); rule_idx++) {
        double fire = FuzzyRule::evaluateFireLevel(getDB(), getInputRulesConditions()[rule_idx], input_values);
        // double fire = getRule(rule_idx).evaluateFireLevel(input_values);
        fire_levels.push_back(fire);
    }
    return fire_levels;
}

// the fire levels for each rule are applied and MAXed to the corresponding consequent output vars
// N.B: those max fire levels are used for the default rules
void FuzzySystem::computeOutputVarsMaxFireLevels(const vector<double>& rules_fire_levels, vector<double>& outvars_max_fire_levels) const {
    const int nb_rules = getNbRules();
    const int nb_out_vars = getDB().getNbOutputVars();
    assert(getOutputRulesConditions().size() > 0);
    assert(rules_fire_levels.size() == nb_rules);
    
    outvars_max_fire_levels.resize(0);
    outvars_max_fire_levels.resize(nb_out_vars, MISSING_DATA_DOUBLE);

    for (int rule_idx = 0; rule_idx < nb_rules; rule_idx++) {
        const double fire_level = rules_fire_levels[rule_idx];
        const int nb = getOutputRulesConditions()[rule_idx].size();

        for (int i =0; i < nb; i++) {
            const int var_idx = getOutputRulesConditions()[rule_idx][i].var_idx;
            outvars_max_fire_levels[var_idx] = max(outvars_max_fire_levels[var_idx], fire_level);
        }
    }
}
// defuzzify the output variables
void FuzzySystem::defuzzify(const Matrix<double>& results, vector<double>& defuzz_values) const {
    const int nb_out_vars = getDB().getNbOutputVars();
    const int nb_sets = getDB().getNbOutputSets();
    assert(results.size() == nb_out_vars && results[0].size() == getDB().getNbOutputSets());
    
    vector<double> set_evals(nb_sets, 0);
    defuzz_values.clear();

    for (int var_idx = 0; var_idx < nb_out_vars; var_idx++) {
        const auto& results_for_var = results[var_idx];
        // gather values for this output variable sets
        set_evals.clear();
        for (int set_idx = 0; set_idx < nb_sets; set_idx++) {
            set_evals.push_back(results_for_var[set_idx]);
        }
        defuzz_values.push_back(getDB().getOutputVariable(var_idx).defuzz(set_evals));
    }
}


// double FuzzySystem::threshold_defuzzed_value(int out_var_idx, double defuzz) const {
//     const int nb_out = getDB().getNbOutputVars();
//     assert(out_var_idx >= 0 && out_var_idx < nb_out);
//     assert(areDefuzzThresholdsEnabled());
//     return apply_threshold(_defuzz_thresholds[out_var_idx], defuzz);
// }

// double FuzzySystem::apply_threshold(double threshold, double value) {
//     if (value >= threshold)
//         value = 1;
//     else if (value  >= 0)
//         value = 0;
//     else value = MISSING_DATA_DOUBLE;
//     return value;
// }

// evaluate the fuzzy system on data: and output the defuzzed values in predicted
DataFrame FuzzySystem::predict(const DataFrame& input)
{
    const int nb_samples = input.nbrows();
    const int nb_out_vars = getDB().getNbOutputVars();
    DataFrame res(nb_samples, nb_out_vars);
    vector<double> defuzzed(res.nbcols());
    
    vector<string> output_names;
    output_names.reserve(nb_out_vars);
    for (int i = 0; i < nb_out_vars; i++)
        output_names.push_back(getDB().getOutputVariable(i).getName());
    res.colnames(output_names);

    for (int i = 0; i < nb_samples; i++) {
        predictSample(i, input, defuzzed);
        res.fillRow(i, defuzzed);
    }

    return res;
}

// same as predict, but is smart on the input data frame:  will use the column names
// to match the values on the corresponding variables
DataFrame FuzzySystem::smartPredict(const DataFrame& df)
{
    const int nb = getDB().getNbInputVars();
    vector<string> dfin_colnames(nb);
    for (int i = 0; i < nb; i++) {
        dfin_colnames[i] = getDB().getInputVariable(i).getName();
    }

    return predict(df.subsetColumns(dfin_colnames));
}

// evaluate the fuzzy system on data for a sample: and output the defuzzed value by output var in defuzzed
void FuzzySystem::predictSample(int sample_idx, const DataFrame& dfin, vector<double>& defuzzed)
{
    assert(sample_idx >= 0);

    // rules processing
    vector<double> fire_levels; // TODO: put in state for reuse ?
    computeRulesFireLevels(sample_idx, dfin, fire_levels);
    computeRulesImplications(fire_levels, getState().output_sets_results);

    computeOutputVarsMaxFireLevels(fire_levels, getState().output_vars_max_fire_levels);
    addDefaultRulesImplications(getDefaultRulesOutputSets(), getState().output_vars_max_fire_levels, getState().output_sets_results);
    // karl TODO: Magali to check if that threshold is relevant, and if it should be a parameter or a constant
    // computeRulesFireLevelsStatistics(fire_levels, 0.2, getState().rules_fired, getState().rules_winners);

    defuzzify(getState().output_sets_results, defuzzed);
    // if (areDefuzzThresholdsEnabled()) {
    //     auto& v = getState().defuzz_thresholded_values;
    //     v.clear();
    //     const int nb_out_vars = getDB().getNbOutputVars();
    //     for (int var_idx = 0; var_idx < nb_out_vars; var_idx++) 
    //         v.push_back(threshold_defuzzed_value(var_idx, getState().defuzz_values[var_idx]));
    
    // }
}

