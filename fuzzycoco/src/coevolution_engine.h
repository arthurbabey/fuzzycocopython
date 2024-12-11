#ifndef COEVOLUTIONENGINE_H
#define COEVOLUTIONENGINE_H

#include <memory>
#include "evolution_engine.h"

class CoevolutionFitnessMethod 
{
public:
  CoevolutionFitnessMethod() {}
  virtual ~CoevolutionFitnessMethod() {}

  double fitness(const Genome& left_genome, const Genome& right_genome) {
    double fit = fitnessImpl(left_genome, right_genome);

    if (fit > _best_fitness) {
      _best_fitness = fit;
      _best.first = left_genome;
      _best.second = right_genome;
    }
    return fit;
  }

  // best so far
  pair<Genome, Genome> getBest() const { return _best; }
  double getBestFitness() const { return _best_fitness; }
  // this is the main method to implement
  virtual double fitnessImpl(const Genome& left_genome, const Genome& right_genome) = 0;

  private:
    double _best_fitness = numeric_limits<double>::lowest();
    pair<Genome, Genome> _best;
};

class CoopCoevolutionFitnessMethod : public CoevolutionFitnessMethod {
public:
  CoopCoevolutionFitnessMethod() {}

  virtual double coopFitness(bool left, const Genome& genome, const Genomes& cooperators);
  // virtual vector<double> coopFitness(bool left, const Genomes& genomes, const Genomes& cooperators);
  // virtual vector<double> fitnesses(const Genome& genome1);
};
class CoopCoevolutionFitnessMethodAdaptator : public EvolutionFitnessMethod {
  public:
    CoopCoevolutionFitnessMethodAdaptator(bool left, CoopCoevolutionFitnessMethod& fit, const Genomes& cooperators) 
      : _left(left), _fit(fit), _cooperators(cooperators) {}
    double fitness(const Genome& genome) override { return _fit.coopFitness(_left, genome, _cooperators);}
  private:
    bool _left;
    CoopCoevolutionFitnessMethod& _fit;
    const Genomes& _cooperators;
};


struct CoevGeneration {
  CoevGeneration() {}
  CoevGeneration(const Genomes& left_genos, const Genomes& right_genos) 
  : left_gen(left_genos), right_gen(right_genos) {}
  // CoevGeneration(const CoevGeneration&& cogen) 
  //       : left_gen(move(cogen.left_gen)), right_gen(move(cogen.right_gen)), fitness(cogen.fitness) {}

  Generation left_gen;
  Generation right_gen;
  double _fitness = 0;
  int generation_number = 0;
  double fitness() const { return _fitness; }
};

class CoevolutionEngine 
{
public:
    // init the engine with some params, nothing else
    // need to check that nb_cooperators <= elite_size
    CoevolutionEngine(CoopCoevolutionFitnessMethod& fit, EvolutionEngine& left_engine, EvolutionEngine& right_engine, int nb_cooperators) 
        : _fit(fit), _left_engine(left_engine), _right_engine(right_engine), _nb_cooperators(nb_cooperators) {}
    ~CoevolutionEngine() {}
    

    // main function
    pair<CoevGeneration, vector<double>>  evolve(const Genomes& left_genos, const Genomes& right_genos, int nb_generations, double maxFit);

    CoevGeneration start(const Genomes& left_genos, const Genomes& right_genos);
    void initGeneration(CoevGeneration& cogen) { nextGeneration(cogen, true); }
    CoevGeneration nextGeneration(CoevGeneration& cogen, bool only_update = false);

    // Generation popNextGeneration(const Genomes& genos, EvolutionFitnessMethod& fit);

    static Genomes selectCooperators(const Genomes& elite, int nb_cooperators);

    // N.B: do not depend upon the Generation: return the best pair seen so far
    pair<Genome, Genome> getBest() const { return _fit.getBest(); }

    // static Genomes selectCooperators(const Genomes& elite, int nb_cooperators);
private:
    EvolutionEngine& _left_engine;
    EvolutionEngine& _right_engine;
    CoopCoevolutionFitnessMethod& _fit;
    int _nb_cooperators;
};

#endif // COEVOLUTIONENGINE_H
