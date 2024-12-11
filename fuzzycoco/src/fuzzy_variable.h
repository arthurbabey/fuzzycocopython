#ifndef FUZZY_VARIABLE_H
#define FUZZY_VARIABLE_H

#include <stdexcept>
#include <cassert>
#include <iostream>
#include <vector>
using namespace std;

#include "fuzzy_set.h"
#include "named_list.h"

// base class for Variables, basically just a name and FuzzySets container
class FuzzyVariable
{
public:
    FuzzyVariable(string name, int nbsets);

    FuzzyVariable(string name, const vector<string>& set_names);
    FuzzyVariable(string name, vector<FuzzySet> sets) : _name(move(name)), _sets(move(sets)) {}
    FuzzyVariable(const FuzzyVariable& var) : _name(var._name), _sets(var._sets) {   }
      FuzzyVariable(FuzzyVariable&& var) : _name(move(var._name)), _sets(move(var._sets)) {}
    virtual ~FuzzyVariable() {}

    string getName() const {  return _name; }

    // bool isOutput(void) const { return _output; }
    // //  Define this variable as an output variable.
    // void setOutput(bool output) { _output = output; }

    const vector<FuzzySet>& getSets() const { return _sets; }
    void setSets(const vector<FuzzySet>& sets) { _sets = sets; }

    // TODO: change that to a FuzzySet& whenever possible
    FuzzySet* getSet(int idx) {
      assert(idx >= 0 && idx < getSetsCount());
      return &_sets[idx];
    }
    const FuzzySet* getSet(int idx) const {
      assert(idx >= 0 && idx < getSetsCount());
      return &_sets[idx];
    }
    // void addSet(FuzzySet& set) {  _sets.push_back(set); }
    int getSetsCount() const { return _sets.size(); }
    int getSetIndexByName(const string& name);

    // // setter
    // void setSetPositions(const vector<double>& set_positions);

    // for testing purposes
    bool operator==(const FuzzyVariable& var) const {
      return getName() == var.getName() && getSets() == var.getSets();
    }

    NamedList describe() const;
    static void printDescription(ostream& out, const NamedList& desc);
    friend ostream& operator<<(ostream& out, const FuzzyVariable& rule);

    static vector<string> build_default_set_names(int nbsets, const string& set_base_name);
private:
    string _name;
    vector<FuzzySet> _sets;

};

class FuzzyInputVariable : public FuzzyVariable {
public:
  using FuzzyVariable::FuzzyVariable;

  // fuzzify: conpute the membership function associated with the input value (from the variable universe)
  // N.B: the result should be in [0, 1]
  double fuzzify(int set_idx, double input) const;

  friend ostream& operator<<(ostream& out, const FuzzyInputVariable& rule);
};

class FuzzyOutputVariable : public FuzzyVariable {
public:
  using FuzzyVariable::FuzzyVariable;

  double defuzz(const vector<double>& set_eval) const;

  friend ostream& operator<<(ostream& out, const FuzzyOutputVariable& rule);
};


#endif // FUZZY_VARIABLE_H
