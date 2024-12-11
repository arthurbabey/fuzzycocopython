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

class FuzzycocoFitnessMethod : public CoopCoevolutionFitnessMethod {
public:
    FuzzycocoFitnessMethod(FuzzySystem& fuzzy_system, FuzzySystemMetricsComputer& fsmc, FuzzySystemFitness& fit,
        const DataFrame& dfin, const DataFrame& dfout,
        RulesCodec& rules_codec, DiscretizedFuzzySystemSetPositionsCodec& vars_codec,
        const vector<double>& thresholds);

    double fitnessImpl(const Genome& rules_genome, const Genome& vars_genome) override;

    FuzzySystemMetrics fitMetrics();

public:
    FuzzySystem& getFuzzySystem() { return _fuzzy_system; }
    void setRulesGenome(const Genome& rules_genome);
    void setMFsGenome(const Genome& mfs_genome);

private:
    FuzzySystem& _fuzzy_system;
    RulesCodec& _rules_codec;
    DiscretizedFuzzySystemSetPositionsCodec& _vars_codec;
    FuzzySystemMetricsComputer& _fsmc;
    FuzzySystemFitness& _fit;
    const DataFrame& _actual_dfin;
    const DataFrame& _actual_dfout;

    vector<ConditionIndexes> _rules_in;
    vector<ConditionIndexes> _rules_out;
    vector<int> _default_rules;
    Matrix<double> _pos_in, _pos_out;
    const vector<double>& _thresholds;
};

// basically an iterator
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

    // pair<Genome, Genome> evolve(const Genomes& rules, const Genomes& mfs_positions, int max_generations, double max_fit);
    const FuzzySystem& buildFuzzySystem(const Genome& rule_geno, const Genome& mfs_geno);

    // accessors
    FuzzycocoFitnessMethod& getFitnessMethod() { return *_fitter_ptr; }

    RulesCodec& getRulesCodec() { return _rules_codec; }
    const RulesCodec& getRulesCodec() const { return _rules_codec; }
    DiscretizedFuzzySystemSetPositionsCodec& getMFsCodec() { return *_vars_codec_ptr; }
    const DiscretizedFuzzySystemSetPositionsCodec& getMFsCodec() const { return *_vars_codec_ptr; }
    CoevolutionEngine& getCoevolutionEngine() { return *_coev_ptr; }
    const CoevolutionEngine& getCoevolutionEngine() const { return *_coev_ptr; }

    const FuzzyCocoParams& getParams() const { return _params; }
    static vector<Discretizer> createDiscretizersForData(const DataFrame& df, int nb_bits);
public:
    void createRulesPop();
    void createVarsPop();

    friend ostream& operator<<(ostream& out, const FuzzyCoco& ds);

protected:
    // void run();
    // FuzzySystem* loadFuzzySystem(QList<QStringList>* listFile, SystemParameters& p);
    // void delete_fuzzy_systems();
private:
    FuzzyCocoParams _params;
    const DataFrame& _actual_dfin;
    const DataFrame& _actual_dfout;
    RandomGenerator& _rng;
    RulesCodec _rules_codec;
    unique_ptr<DiscretizedFuzzySystemSetPositionsCodec> _vars_codec_ptr;
    unique_ptr<FuzzycocoFitnessMethod> _fitter_ptr;
    FuzzySystem _fs;
    FuzzySystemMetricsComputer _fsmc;
    FuzzySystemWeightedFitness _fsfit;
    EvolutionEngine _rules_evo;
    EvolutionEngine _mfs_evo;
    unique_ptr<CoevolutionEngine> _coev_ptr;

    public:
    // FuzzySystem* computeFuzzySystem(const FuzzyCocoParams& params, const FuzzyCocoData& data);
    // FuzzySystem* computeFuzzySystemFromScript(const string& script, const FuzzyCocoData& data);
    // FuzzySystem* computeFuzzySystemFromScriptFile(const string& script_filename, const FuzzyCocoData& data);

    // vector<float> predict(const FuzzyCocoData& data, FuzzySystem& fs);
    // FuzzySystemMetrics eval(const FuzzyCocoData& data, FuzzySystem& fs);

    // static FuzzySystem* bestFSystem;
    // static QString bestFuzzySystemDescription;
    // static qreal bestFitness;
    // static void savePredictionResults(QString filename, const QVector<float>& results);
    // static void saveFuzzyAndFitness(FuzzySystem *fSystem, qreal fitness);
    // static void saveSystemStats(QString name, qreal minFitness, qreal maxFitness, qreal meanFitness, qreal standardDeviation, int populationSize, int generation);
    // static SystemParameters *sysParams;
};

#endif // FUZZY_COCO_H
