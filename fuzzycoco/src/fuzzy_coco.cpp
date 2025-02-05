#include "fuzzy_coco.h"
#include "logging_logger.h"
using namespace logging;


FuzzyCocoSystem::FuzzyCocoSystem(const DataFrame& dfin, const DataFrame& dfout, const FuzzyCocoParams& params)
  :
    _rules_codec(params.global_params.nb_rules,
      {min(dfin.nbcols(), params.global_params.nb_max_var_per_rule), params.input_vars_params.nb_bits_vars, params.input_vars_params.nb_bits_sets},
      {dfout.nbcols(), params.output_vars_params.nb_bits_vars, params.output_vars_params.nb_bits_sets}),
    FuzzySystem(dfin.colnames(), dfout.colnames(), params.input_vars_params.nb_sets, params.output_vars_params.nb_sets)
{
  const int nb_input_vars = dfin.nbcols();
  const int nb_output_vars = dfout.nbcols();

    // params
  const int nb_input_sets = params.input_vars_params.nb_sets;
  const int nb_output_sets = params.output_vars_params.nb_sets;
  const int nb_rules = params.global_params.nb_rules;

  PosParams pos_input(nb_input_vars, nb_input_sets, params.input_vars_params.nb_bits_pos);
  PosParams pos_output(nb_output_vars, nb_output_sets, params.output_vars_params.nb_bits_pos);
  auto disc_in = FuzzyCoco::createDiscretizersForData(dfin, pos_input.nb_bits);
  auto disc_out = FuzzyCoco::createDiscretizersForData(dfout, pos_output.nb_bits);

  _vars_codec_ptr = make_unique<DiscretizedFuzzySystemSetPositionsCodec>(pos_input, pos_output, disc_in, disc_out);

  // pre-sizing
  _rules_in.reserve(nb_rules);
  _rules_out.reserve(nb_rules);

}

void FuzzyCocoSystem::setRulesGenome(const Genome& rules_genome) {
  auto it1 = rules_genome.cbegin();
  getRulesCodec().decode(it1, _rules_in, _rules_out, _default_rules);

  setRulesConditions(_rules_in, _rules_out);
  setDefaultRulesConditions(_default_rules);
}

void FuzzyCocoSystem::setMFsGenome(const Genome& mfs_genome) {
  auto it = mfs_genome.cbegin();
  getMFsCodec().decode(it, _pos_in, _pos_out);
  setVariablesSetPositions(_pos_in, _pos_out);
}


FuzzycocoFitnessMethod::FuzzycocoFitnessMethod(const DataFrame &dfin, const DataFrame &dfout, const FuzzyCocoParams& params)
    : FuzzycocoFitnessMethod(unique_ptr<FuzzySystemFitness>(new FuzzySystemWeightedFitness(params.metrics_weights)),
        dfin, dfout, params)
{}

FuzzycocoFitnessMethod::FuzzycocoFitnessMethod(unique_ptr<FuzzySystemFitness> fit_ptr,
                                               const DataFrame &dfin, const DataFrame &dfout, const FuzzyCocoParams& params)
  :   _fuzzy_system(dfin, dfout, params),
      _fsmc(),
      _actual_dfout(dfout), _actual_dfin(dfin),
      _fit_ptr(move(fit_ptr)), _thresholds(params.output_vars_defuzz_thresholds)
{
  assert(_thresholds.size() == dfout.nbcols());
}


double FuzzycocoFitnessMethod::fitnessImpl(const Genome& rules_genome, const Genome& mfs_genome)
{
    _fuzzy_system.setRulesGenome(rules_genome);
    _fuzzy_system.setMFsGenome(mfs_genome);
    if (!_fuzzy_system.ok()) return 0;

    return _fit_ptr->fitness(fitMetrics());
}

FuzzySystemMetrics FuzzycocoFitnessMethod::fitMetrics()
{
    auto predicted_output = _fuzzy_system.predict(_actual_dfin);
    // cerr << "predicted_output:" << predicted_output << endl;
    auto metrics = _fsmc.compute(predicted_output, _actual_dfout,_thresholds);

    // VERY IMPORTANT FOR NOW: need to add the number of variables used in the rules
    metrics.nb_vars = _fuzzy_system.computeTotalInputVarsUsedInRules();

    return metrics;
}

FuzzyCoco::FuzzyCoco(const DataFrame& dfin, const DataFrame& dfout, unique_ptr<FuzzycocoFitnessMethod> fit_method_ptr,
   const FuzzyCocoParams& params, RandomGenerator& rng)
  :  _actual_dfout(dfout), _actual_dfin(dfin), _params(params),
    _rng(rng),
    // _fs(dfin, dfout, params),
    _rules_evo(params.rules_params, rng),
    _mfs_evo(params.mfs_params, rng),
    _fitter_ptr(move(fit_method_ptr)),
    _coev_ptr(make_unique<CoevolutionEngine>(CoevolutionEngine(*_fitter_ptr,  _rules_evo, _mfs_evo, params.global_params.nb_cooperators)))
{
  if (params.has_missing()) throw runtime_error("ERROR: some parameters in params are missing!");
}

FuzzyCoco::FuzzyCoco(const DataFrame& dfin, const DataFrame& dfout, unique_ptr<FuzzySystemFitness> fit_ptr,
   const FuzzyCocoParams& params, RandomGenerator& rng)
  :  FuzzyCoco(dfin, dfout, make_unique<FuzzycocoFitnessMethod>(move(fit_ptr), dfin, dfout, params), params, rng)
{}

FuzzyCoco::FuzzyCoco(const DataFrame& dfin, const DataFrame& dfout, const FuzzyCocoParams& params, RandomGenerator& rng)
  : FuzzyCoco(dfin, dfout, unique_ptr<FuzzySystemFitness>(new FuzzySystemWeightedFitness(params.metrics_weights)), params, rng)
{}

ostream& operator<<(ostream& out, const FuzzyCoco& coco) {
  out << "FuzzyCoco:" << endl;
  out << "--------------------------------------------------------" << endl;
  out << "# PARAMS" << endl;
  out << coco._params << endl;
  out << "# Input Data" << endl << coco._actual_dfin << endl;
  out << "# Output Data" << endl << coco._actual_dfout << endl;
  out << "# Rules Codec " << coco.getFuzzySystem().getRulesCodec() << endl;
  out << "# MFs Codec " << coco.getFuzzySystem().getMFsCodec() << endl;
  return out;
}

NamedList FuzzyCoco::describeBestFuzzySystem()
{
  auto [best_rule, best_mf] = getBest();

  auto& fs = getFuzzySystem();
  fs.setRulesGenome(best_rule);
  fs.setMFsGenome(best_mf);

  double fitness = getFitnessMethod().fitnessImpl(best_rule, best_mf);
  NamedList desc;
  desc.add("fitness", fitness);
  auto mw = getParams().metrics_weights.describe();
  desc.add("fitness_metrics_weights", mw);

  const auto& thresholds = getParams().output_vars_defuzz_thresholds;
  NamedList thresh;
  const auto& db = fs.getDB();
  const int nb_out_vars = db.getNbOutputVars();
  assert(nb_out_vars == thresholds.size());
  for (int i = 0; i < nb_out_vars; i++)
    thresh.add(db.getOutputVariable(i).getName(), thresholds[i]);

  desc.add("fuzzy_system", fs.describe());
  desc.add("defuzz_thresholds", thresh);

  return desc;
}

CoevGeneration FuzzyCoco::start(int nb_pop_rules, int nb_pop_mfs) {
  Genomes rules;
  rules.reserve(nb_pop_rules);
  for (int i = 0; i < nb_pop_rules; i++) {
    Genome rules_geno(getFuzzySystem().getRulesCodec().size());
    randomize(rules_geno, _rng);
    rules.push_back(rules_geno);
  }

  Genomes mfs;
  mfs.reserve(nb_pop_mfs);
  for (int i = 0; i < nb_pop_mfs; i++) {
    Genome mf(getFuzzySystem().getMFsCodec().size());
    randomize(mf, _rng);
    mfs.push_back(mf);
  }

  return start(rules, mfs);
}

CoevGeneration FuzzyCoco::start(const Genomes& rules, const Genomes& mfs) {
  CoevGeneration cogen(rules, mfs);
  getCoevolutionEngine().initGeneration(cogen);
  return cogen;
}

CoevGeneration FuzzyCoco::next(CoevGeneration& cogen) {
  return getCoevolutionEngine().nextGeneration(cogen);
}

CoevGeneration FuzzyCoco::run() {
  auto gen = start();
  const int nb_generations = getParams().global_params.max_generations;
  const double max_fit = getParams().global_params.max_fitness;
  for (int i = 0; i < nb_generations; i++) {
    gen = next(gen);
    logger() << gen.fitness() << ", ";

    if (gen.fitness() >= max_fit) break;
  }
  logger() << endl;
  return gen;
}


// pair<Genome, Genome> FuzzyCoco::evolve(const Genomes& rules, const Genomes& mfs, int max_generations, double max_fit) {

//   auto [lastgen, generation_fitnesses] = getCoevolutionEngine().evolve(rules, mfs, max_generations, max_fit);

//   int nb_gen = generation_fitnesses.size();
//   cerr << "fitnesses: " << generation_fitnesses << endl;

//   return getCoevolutionEngine().getBest();
// }


vector<Discretizer> FuzzyCoco::createDiscretizersForData(const DataFrame& df, int nb_bits) {
  const int nb = df.nbcols();
  vector<Discretizer> res;
  res.reserve(nb);
  for (int i = 0; i < nb; i++) {
    res.push_back({nb_bits, df[i]});
  }

  return res;
}
