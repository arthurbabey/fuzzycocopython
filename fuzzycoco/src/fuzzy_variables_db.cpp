#include "fuzzy_variables_db.h"
#include <algorithm>

FuzzyVariablesDB::FuzzyVariablesDB(
  const vector<string>& input_names, int nb_in_sets, 
  const vector<string>& output_names, int nb_out_sets)
{
  const int nb_in_vars = input_names.size();
  const int nb_out_vars = output_names.size();
  
  assert(nb_in_vars > 0);
  assert(nb_in_sets > 0);
  assert(nb_out_vars > 0);
  assert(nb_out_sets > 0);

  _input_vars.reserve(nb_in_vars);
  for (auto name : input_names)
    _input_vars.push_back({name, nb_in_sets});

  _output_vars.reserve(nb_out_vars);
  for (auto name : output_names)
    _output_vars.push_back({name, nb_out_sets});
}

void FuzzyVariablesDB::setPositions(const Matrix<double>& insets_pos_mat, const Matrix<double>& outsets_pos_mat)
{
    assert(insets_pos_mat.nbrows() == getNbInputVars() && insets_pos_mat.nbcols() == getNbInputSets());
    assert(outsets_pos_mat.nbrows() == getNbOutputVars() && outsets_pos_mat.nbcols() == getNbOutputSets());

    auto fill = [](const Matrix<double>& m, auto& vars) {
        const int nb_vars = m.nbrows();
        const int nb_sets = m.nbcols();
        for (int var_idx = 0; var_idx < nb_vars; var_idx++) {
            auto row = m[var_idx];
            sort(row.begin(), row.end());
            auto& var = vars[var_idx];
            for (int set_idx = 0; set_idx < nb_sets; set_idx++) {
                var.getSet(set_idx)->setPosition(row[set_idx]);
            }
        }
    };

    fill(insets_pos_mat, _input_vars);
    fill(outsets_pos_mat, _output_vars);
}

NamedList FuzzyVariablesDB::describe() const {
    NamedList vars;
    NamedList input_vars;
    for (const auto& var : _input_vars) 
        input_vars.add(var.getName(), var.describe());
    vars.add("input", input_vars);
    
    NamedList output_vars;
    for (const auto& var : _output_vars) {
        output_vars.add(var.getName(), var.describe());
    }
    vars.add("output", output_vars);
 

  return vars;
}

FuzzyVariablesDB FuzzyVariablesDB::load(const NamedList& desc) {

  auto load_vars = [](const NamedList& vars, vector<string>& var_names, Matrix<string>& set_names, Matrix<double>& pos) {
      const int nb = vars.size();
      assert(nb > 0);
      var_names.reserve(nb);
      const int nb_sets = vars[0].get_list("Sets").size();
      assert(nb_sets > 0);
      pos.redim(nb, nb_sets);
      set_names.redim(nb, nb_sets);
      for (int i = 0; i < nb; i++) {
        const auto& var = vars[i];
        var_names.push_back(var.get_string("name"));
        const auto& sets = var.get_list("Sets");
        assert(sets.size() == nb_sets);
        for (int j = 0; j < nb_sets; j++) {
          pos[i][j] = sets[j].get_double("position");
          set_names[i][j] = sets[j].get_string("name");
      }
    }
  };

  vector<string> input_names, output_names;
  Matrix<double> posin, posout;
  Matrix<string> setin, setout;
  load_vars(desc.get_list("input"), input_names, setin, posin);
  load_vars(desc.get_list("output"), output_names, setout, posout);

  FuzzyVariablesDB db(input_names, posin.nbcols(), output_names, posout.nbcols());
  db.setPositions(posin, posout);
  
  // set names
  const int nb_input_sets = db.getNbInputSets();
  for (int i = 0; i < db.getNbInputVars(); i++)
    for (int j = 0; j < nb_input_sets; j++)
      db.getInputVariable(i).getSet(j)->setName(setin[i][j]);
  
  const int nb_output_sets = db.getNbOutputSets();
  for (int i = 0; i < db.getNbOutputVars(); i++)
    for (int j = 0; j < nb_output_sets; j++)
      db.getOutputVariable(i).getSet(j)->setName(setout[i][j]);  

  return db;
}

FuzzyVariablesDB FuzzyVariablesDB::subset(const vector<int>& input_var_idx, const vector<int>& output_var_idx) const {
  FuzzyVariablesDB db2;
  
  const int nbin = getNbInputVars();
  for (int idx : input_var_idx) {
    assert(idx >= 0 && idx < nbin);
    db2._input_vars.push_back(_input_vars[idx]);
  }

  const int nbout = getNbOutputVars();
  for (int idx : output_var_idx) {
    assert(idx >= 0 && idx < nbout);
    db2._output_vars.push_back(_output_vars[idx]);
  }

  return db2;
}

ostream& operator<<(ostream& out, const FuzzyVariablesDB& db) 
{
  const char TAB = '\t';
  out << "Fuzzy Variables Database:" << endl;
  out << "## input variables:" << endl;
  for (const auto& var : db._input_vars)
      out << TAB << var << endl;

  out << "## output variables:" << endl;
  for (const auto& var : db._output_vars)
      out << TAB << var << endl;
  return out;
}