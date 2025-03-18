#ifndef FUZZY_COCO_H
#define FUZZY_COCO_H

#include <memory>
#include "fuzzy_coco_params.h"
#include "fuzzy_system.h"
#include "fuzzy_system_metrics_computer.h"
#include "fuzzy_system_fitness.h"
#include "coevolution_engine.h"
#include "genome_codec.h"
#include "discretizer.h"

// extension of FuzzySystem to deal with the Rules and Genomes encoding
class FuzzyCocoSystem : public FuzzySystem {
public:
  FuzzyCocoSystem(const DataFrame& dfin, const DataFrame& dfout, const FuzzyCocoParams& params);

  void setRulesGenome(const Genome& rules_genome);
  void setMFsGenome(const Genome& mfs_genome);

// accessors
RulesCodec& getRulesCodec() { return _rules_codec; }
const RulesCodec& getRulesCodec() const { return _rules_codec; }
DiscretizedFuzzySystemSetPositionsCodec& getMFsCodec() { return *_vars_codec_ptr; }
const DiscretizedFuzzySystemSetPositionsCodec& getMFsCodec() const { return *_vars_codec_ptr; }

private:
  // the genome codecs
  RulesCodec _rules_codec;
  unique_ptr<DiscretizedFuzzySystemSetPositionsCodec> _vars_codec_ptr;

  // internal state
  vector<ConditionIndexes> _rules_in;
  vector<ConditionIndexes> _rules_out;
  vector<int> _default_rules;
  Matrix<double> _pos_in, _pos_out;
};

// a CoopCoevolutionFitnessMethod that knows how to evaluate a co-population (rules, MFs)
class FuzzycocoFitnessMethod : public CoopCoevolutionFitnessMethod {
public:
    FuzzycocoFitnessMethod(const DataFrame& dfin, const DataFrame& dfout, const FuzzyCocoParams& params);
    FuzzycocoFitnessMethod(unique_ptr<FuzzySystemFitness> fit_ptr,
        const DataFrame& dfin, const DataFrame& dfout, const FuzzyCocoParams& params);

    double fitnessImpl(const Genome& rules_genome, const Genome& vars_genome) override;

    FuzzySystemMetrics fitMetrics();

public:
  FuzzyCocoSystem& getFuzzySystem() { return _fuzzy_system; }
  const FuzzyCocoSystem& getFuzzySystem() const { return _fuzzy_system; }
  FuzzySystemFitness& getFuzzySystemFitness() { return *_fit_ptr; }

private:
    FuzzyCocoSystem _fuzzy_system;
    FuzzySystemMetricsComputer _fsmc;
    unique_ptr<FuzzySystemFitness> _fit_ptr;
    const DataFrame& _actual_dfin;
    const DataFrame& _actual_dfout;
    const vector<double>& _thresholds;
};


class FuzzyCoco;

class FuzzyCocoGeneration {
public:
    FuzzyCocoGeneration(const EvolutionParams& params_rules, const Genomes& rules,
         const EvolutionParams& params_mfs, const Genomes& mfs,
         FuzzycocoFitnessMethod& fitter,
         RandomGenerator& rng);
private:
    CoevolutionEngine& _coev_engine;
};

class FuzzyCoco
{
public:
    FuzzyCoco(const DataFrame& dfin, const DataFrame& dfout,
        const FuzzyCocoParams& params, RandomGenerator& rng);

    FuzzyCoco(const DataFrame& dfin, const DataFrame& dfout, unique_ptr<FuzzySystemFitness> fit_ptr,
        const FuzzyCocoParams& params, RandomGenerator& rng);

    FuzzyCoco(const DataFrame& dfin, const DataFrame& dfout, unique_ptr<FuzzycocoFitnessMethod> fit_method_ptr,
        const FuzzyCocoParams& params, RandomGenerator& rng);
    // void setFitnessMethod(unique_ptr<FuzzycocoFitnessMethod> fit_method_ptr);
    virtual ~FuzzyCoco() {}


    // ================ main interface ======================
    // highest level function, Runs everythung using the params
    CoevGeneration run();

    CoevGeneration start() { return start(getParams().rules_params.pop_size, getParams().mfs_params.pop_size); }
    CoevGeneration start(int nb_pop_rules, int nb_pop_mfs);
    CoevGeneration start(const Genomes& rules, const Genomes& mfs);

    CoevGeneration next(CoevGeneration& cogen);
    pair<Genome, Genome> getBest() const { return getCoevolutionEngine().getBest(); }

    NamedList describeBestFuzzySystem();

    // accessors

    FuzzyCocoSystem& getFuzzySystem() { return getFitnessMethod().getFuzzySystem(); }
    const FuzzyCocoSystem& getFuzzySystem() const { return getFitnessMethod().getFuzzySystem(); }
    FuzzycocoFitnessMethod& getFitnessMethod() { return *_fitter_ptr; }
    const FuzzycocoFitnessMethod& getFitnessMethod() const { return *_fitter_ptr; }

    CoevolutionEngine& getCoevolutionEngine() { return *_coev_ptr; }
    const CoevolutionEngine& getCoevolutionEngine() const { return *_coev_ptr; }

    const FuzzyCocoParams& getParams() const { return _params; }
    static vector<Discretizer> createDiscretizersForData(const DataFrame& df, int nb_bits);
public:
    void createRulesPop();
    void createVarsPop();

    friend ostream& operator<<(ostream& out, const FuzzyCoco& ds);

    // ARTHUR: expose fitness history of fuzzy coco through fuzzy system fitness
    const std::vector<double>& getEvolutionFitnessHistory() const { return _fitness_history; }

protected:
    // void run();
    // FuzzySystem* loadFuzzySystem(QList<QStringList>* listFile, SystemParameters& p);
    // void delete_fuzzy_systems();
private:
    FuzzyCocoParams _params;
    const DataFrame& _actual_dfin;
    const DataFrame& _actual_dfout;
    RandomGenerator& _rng;

    unique_ptr<FuzzycocoFitnessMethod> _fitter_ptr;

    EvolutionEngine _rules_evo;
    EvolutionEngine _mfs_evo;
    unique_ptr<CoevolutionEngine> _coev_ptr;

    std::vector<double> _fitness_history;
};

#endif // FUZZY_COCO_H
