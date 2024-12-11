#include "fuzzy_coco.h"
#include "logging_logger.h"
using namespace logging;


FuzzycocoFitnessMethod::FuzzycocoFitnessMethod(FuzzySystem &fuzzy_system, FuzzySystemMetricsComputer &fsmc, FuzzySystemFitness &fit,
                                               const DataFrame &dfin, const DataFrame &dfout,
                                               RulesCodec &rules_codec, DiscretizedFuzzySystemSetPositionsCodec &vars_codec,
                                               const vector<double> &thresholds)
    : _fuzzy_system(fuzzy_system), _fsmc(fsmc), _rules_codec(rules_codec), _vars_codec(vars_codec),
      _actual_dfout(dfout), _actual_dfin(dfin),
      _fit(fit), _thresholds(thresholds)
{
  assert(_thresholds.size() == dfout.nbcols());
  const int nb_rules = _rules_codec.getNbRules();
  _rules_in.reserve(nb_rules);
  _rules_out.reserve(nb_rules);
}

void FuzzycocoFitnessMethod::setRulesGenome(const Genome& rules_genome) {
  auto it1 = rules_genome.cbegin();
  _rules_codec.decode(it1, _rules_in, _rules_out, _default_rules);

  _fuzzy_system.setRulesConditions(_rules_in, _rules_out);
  _fuzzy_system.setDefaultRulesConditions(_default_rules);
}

void FuzzycocoFitnessMethod::setMFsGenome(const Genome& mfs_genome) {
  auto it = mfs_genome.cbegin();
  _vars_codec.decode(it, _pos_in, _pos_out);
  _fuzzy_system.setVariablesSetPositions(_pos_in, _pos_out);
}

double FuzzycocoFitnessMethod::fitnessImpl(const Genome& rules_genome, const Genome& mfs_genome)
{
    setRulesGenome(rules_genome);
    setMFsGenome(mfs_genome);
    if (!_fuzzy_system.ok()) return 0;

    return _fit.fitness(fitMetrics());
}

FuzzySystemMetrics FuzzycocoFitnessMethod::fitMetrics()
{
    auto predicted_output = _fuzzy_system.predict(_actual_dfin);
    // cerr << "predicted_output:" << predicted_output << endl;
    auto metrics = _fsmc.compute(predicted_output, _actual_dfout,_thresholds);

    // VERY IMPORTANT FOR NOW: need to add the number of variables used in the rules
    metrics.nb_vars = _fuzzy_system.computeTotalInputVarsUsedInRules();
    // cerr << metrics << endl;
    // cerr << _fuzzy_system;

    return metrics;
}


FuzzyCoco::FuzzyCoco(const DataFrame& dfin, const DataFrame& dfout, const FuzzyCocoParams& params, RandomGenerator& rng)
  :  _actual_dfout(dfout), _actual_dfin(dfin), _params(params),
    _rng(rng),
    _rules_codec(params.global_params.nb_rules,
      {min(dfin.nbcols(), params.global_params.nb_max_var_per_rule), params.input_vars_params.nb_bits_vars, params.input_vars_params.nb_bits_sets},
      {dfout.nbcols(), params.output_vars_params.nb_bits_vars, params.output_vars_params.nb_bits_sets}),
    _fs(dfin.colnames(), dfout.colnames(), params.input_vars_params.nb_sets, params.output_vars_params.nb_sets),
    _fsmc(),
    _fsfit(params.metrics_weights),
    _rules_evo(params.rules_params, rng),
    _mfs_evo(params.mfs_params, rng)
{
  if (params.has_missing()) throw runtime_error("ERROR: some parameters in params are missing!");

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
  // DiscretizedFuzzySystemSetPositionsCodec vars_codec(pos_input, pos_output, disc_in, disc_out);
  _vars_codec_ptr = make_unique<DiscretizedFuzzySystemSetPositionsCodec>(pos_input, pos_output, disc_in, disc_out);

  _fitter_ptr = make_unique<FuzzycocoFitnessMethod>(FuzzycocoFitnessMethod(_fs, _fsmc, _fsfit, dfin, dfout, _rules_codec, *_vars_codec_ptr,
    params.output_vars_defuzz_thresholds));

  _coev_ptr = make_unique<CoevolutionEngine>(CoevolutionEngine(*_fitter_ptr,  _rules_evo, _mfs_evo, params.global_params.nb_cooperators));
}

ostream& operator<<(ostream& out, const FuzzyCoco& coco) {
  out << "FuzzyCoco:" << endl;
  out << "--------------------------------------------------------" << endl;
  out << "# PARAMS" << endl;
  out << coco._params << endl;
  out << "# Input Data" << endl << coco._actual_dfin << endl;
  out << "# Output Data" << endl << coco._actual_dfout << endl;
  out << "# Rules Codec " << coco.getRulesCodec() << endl;
  out << "# MFs Codec " << coco.getMFsCodec() << endl;
  return out;
}

const FuzzySystem& FuzzyCoco::buildFuzzySystem(const Genome& rule_geno, const Genome& mfs_geno)
{
  auto& fitter = getFitnessMethod();
  fitter.setRulesGenome(rule_geno);
  fitter.setMFsGenome(mfs_geno);

  return fitter.getFuzzySystem();
}

NamedList FuzzyCoco::describeBestFuzzySystem()
{
  auto [best_rule, best_mf] = getBest();
  FuzzySystem fs = buildFuzzySystem(best_rule, best_mf);
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
    Genome rules_geno(getRulesCodec().size());
    randomize(rules_geno, _rng);
    rules.push_back(rules_geno);
  }

  Genomes mfs;
  mfs.reserve(nb_pop_mfs);
  for (int i = 0; i < nb_pop_mfs; i++) {
    Genome mf(getMFsCodec().size());
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
