
#include <iostream>
#include <sstream>
#include <algorithm>
#include "fuzzy_rule.h"
#include "fuzzy_operator.h"
#include "types.h"


ConditionIndexes FuzzyRule::filterConditionIndexes(int nb_vars, int nb_sets, const ConditionIndexes& cis)
{
    ConditionIndexes goods;
    goods.reserve(cis.size());
    vector<bool> used(nb_vars, false);
    for (auto pair : cis) {
        const int var_idx = pair.var_idx;
        if (var_idx >= 0 && var_idx < nb_vars && !used[var_idx] &&
            pair.set_idx >= 0 && pair.set_idx < nb_sets) {
            used[var_idx] = true;
            goods.push_back(pair);
        }
    }

    return goods;   
}

ConditionIndexes FuzzyRule::filterConditionIndexesWhenFixedVars(int nb_sets, const ConditionIndexes& cis)
{
    // trick: we just copy and fix the var_idx, then reuse filterConditionIndexes()
    ConditionIndexes cis2(cis);
    const int nb_pairs = cis.size();
    // in this case of fixed vars, the var_idx is just the index of the pair
    for (int i = 0; i < nb_pairs; i++)
        cis2[i].var_idx = i;
    
    return filterConditionIndexes(1, nb_sets, cis2);
}

// // FIXME: Output values don't support -1 due to this implementation.
// #define DONT_CARE_EVAL_RULE -1.0

FuzzyRule::FuzzyRule(const FuzzyVariablesDB& db) : _db(db) {
    setConditions({{0,0}}, {{0,0}});
}

void FuzzyRule::setConditions(const ConditionIndexes& input_conds, const ConditionIndexes& output_conds)
{
    assert(!input_conds.empty());
    assert(!output_conds.empty());
    _input_cond = input_conds;
    _output_cond = output_conds;
}

NamedList FuzzyRule::describe() const{
    NamedList desc;
    NamedList antecedents;
    for (int i = 0; i < getNbInputConditions(); i++)  {
        NamedList antecedent;
        const auto& index = getInputConditionIndex(i);
        const auto& var = getDB().getInputVariable(index.var_idx);
        antecedent.add("var_name", var.getName());

        auto set_desc = var.getSet(index.set_idx)->describe();

        antecedent.add("set_name", set_desc.get_string("name"));
        // set indexes are stable, and convenient for load()
        antecedent.add("set_index", index.set_idx);
        antecedent.add("set_position", set_desc.get_double("position"));
        antecedent.add("set_pretty", set_desc.get_string("pretty"));

        antecedents.add("antecedent", antecedent);
    }
    desc.add("antecedents", antecedents);

    NamedList consequents;
    for (int i = 0; i < getNbOutputConditions(); i++) {
        NamedList consequent;

        const auto& index = getOutputConditionIndex(i);
        const auto& var = getDB().getOutputVariable(index.var_idx);
        consequent.add("var_name", var.getName());

        auto set_desc = var.getSet(index.set_idx)->describe();

        consequent.add("set_name", set_desc.get_string("name"));
        // set indexes are stable, and convenient for load()
        consequent.add("set_index", index.set_idx);
        consequent.add("set_position", set_desc.get_double("position"));
        consequent.add("set_pretty", set_desc.get_string("pretty"));

        consequents.add("consequent", consequent);
    }
    desc.add("consequents", consequents);

    return desc;
}

void FuzzyRule::printDescription(ostream& out, const NamedList& desc)
{
   out << "IF ";
    auto& antecedents = desc.get_list("antecedents");
    int nb = antecedents.size();
    for (int i = 0; i < nb; i++) {
        auto& antecedent = antecedents[i];
        out << quoted(antecedent.get_string("var_name"));
        out << " IS ";
        out << antecedent.get_string("set_pretty");
        if (i != nb-1)
            out << " AND ";
    }
    out << " THEN ";
    auto& consequents = desc.get_list("consequents");
    nb = consequents.size();
    for (int i = 0; i < nb; i++) {
        auto& consequent = consequents[i];
        out << quoted(consequent.get_string("var_name"));
        out << " SHOULD BE ";
        out << consequent.get_string("set_pretty");
        if (i != nb-1)
            out << " AND ";
    }
}

ostream& operator<<(ostream& out, const FuzzyRule& rule) {
    FuzzyRule::printDescription(out, rule.describe());
    return out;
}

// static
double FuzzyRule::evaluateInputConditionFireLevel(const FuzzyVariablesDB& db, const ConditionIndex& ci, double value)
{
    return is_na(value) ? MISSING_DATA_DOUBLE 
        : db.getInputVariable(ci.var_idx).fuzzify(ci.set_idx, value);     
}

double FuzzyRule::evaluateInputConditionFireLevel(int idx, double value) const { 
    return evaluateInputConditionFireLevel(getDB(), getInputConditionIndex(idx), value);
}
// static
double FuzzyRule::evaluateFireLevel(const FuzzyVariablesDB& db, const ConditionIndexes& cis, const vector<double>& input_vars_values)
{
    const int nb_input = cis.size();
    assert(nb_input > 0);
    assert(input_vars_values.size() == db.getNbInputVars());

    vector<double> evals(nb_input, 0);
    for (int i = 0; i < nb_input; i++) {
        const auto& ci = cis[i];
        evals[i] = evaluateInputConditionFireLevel(db, ci, input_vars_values[ci.var_idx]);
    }

    return combineFireLevels(evals);
}

double FuzzyRule::evaluateFireLevel(const vector<double>& input_vars_values) const {
    return evaluateFireLevel(getDB(), getInputConditionIndexes(), input_vars_values);
}

// evaluate the firing level of a rule where the input values are stored in a row of a dataframe
// the input variables are assumed to be in the same order as the df columns
double FuzzyRule::evaluateFireLevel(const DataFrame& df, const int row) const {
    return evaluateFireLevel(getDB(), getInputConditionIndexes(), df, row);
}

// static 
double FuzzyRule::evaluateFireLevel(const FuzzyVariablesDB& db, const ConditionIndexes& cis,const DataFrame& df, const int row)  {
    const int nb_input = cis.size();
    assert(nb_input > 0);
    assert(df.nbcols() == db.getNbInputVars());
    assert(row >= 0 && row < df.nbrows());

    vector<double> evals(nb_input, 0);
    for (int i = 0; i < nb_input; i++) {
        const auto& ci = cis[i];
        evals[i] = evaluateInputConditionFireLevel(db, ci, df.at(row, ci.var_idx));
    }
    
    return combineFireLevels(evals);
}

double FuzzyRule::combineFireLevels(const vector<double>& fire_levels) {
    const int nb_input = fire_levels.size();
    assert(nb_input > 0);

    double eval = fire_levels[0];
    if (nb_input > 1) {
        //TODO: The operator should be provided as a param
        FuzzyOperatorAND op;
        for (int i = 1; i < nb_input; i++) {
            eval = op.operate(eval, fire_levels[i]);
        }
    }
    return eval;
}