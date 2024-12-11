// modified from evolutionengine.h to remove the multithreading and the coupling with higher level classes
//  *     such as ComputeThread/FuzzyCoco. Actually implement an Evolution Base class

#ifndef EVOLUTIONENGINE_H
#define EVOLUTIONENGINE_H


#include "crossover_method.h"
#include "mutation_method.h"
#include "selection_method.h"
#include "evolution_params.h"

// Evolve a next Generation of genomes:
// - a subgroup (Elite) is selected and passed over to the next generation
// - a subgroup is selected (evolvers) for mutation and crossover --> evolved
// the next generation is the group formed by those two subgroups: the elite + the evolved
// N.B: evolvers and elite can have an intersection

struct GenerationFitness {
    GenerationFitness(int nb) : fitnesses(nb) {}
    vector<double> fitnesses;
    double fitness = 0;
};

struct Generation {
    Generation() {}
    Generation(const Genomes &individuals, const Genomes &elite) : individuals(individuals), elite(elite), fitnesses(individuals.size()) {}
    Generation(const Genomes &individuals) : Generation(individuals, {}) {}
    // use Implicitly-declared move constructor instead
    // Generation(const Generation&& gen)
    //     : individuals(move(gen.individuals)), elite(move(gen.elite)), fitnesses(move(gen.fitnesses)), fitness(gen.fitness) {}

    Genomes individuals;
    Genomes elite;

    vector<double> fitnesses;
    double fitness = 0;
};

class EvolutionFitnessMethod
{
public:
  EvolutionFitnessMethod() {}
  virtual ~EvolutionFitnessMethod() {}

  virtual double fitness(const Genome& genome) = 0;
  virtual double globalFitness(const vector<double>& fitnesses);
};

class EvolutionEngine
{
public:
    // init the engine with some params, nothing else
    EvolutionEngine(const EvolutionParams& params, RandomGenerator& rng);
    ~EvolutionEngine() {}

    const EvolutionParams& params() const { return _params; }

    // main function
    pair<Generation, vector<double>> evolve(const Genomes& genomes, EvolutionFitnessMethod& fitness_method,
        int nb_generations, double maxFit);

    Generation nextGeneration(const Generation& generation, EvolutionFitnessMethod& fitness_method);

    // initial selection, when no fitnesses yet
    Genomes selectElite(const Genomes& genomes);
    Genomes selectElite(const Genomes& genomes, const vector<double>& fitnesses);
    Genomes selectEvolvers(int nb, const Genomes& genomes, const vector<double>& fitnesses);

    void updateGeneration(Generation& generation, EvolutionFitnessMethod& fitness_method);

    static Genomes selectBest(const Generation& generation);
private:
    EvolutionParams _params;
    RandomGenerator& _rng;
    OnePointCrossoverMethod _crossover_method;
    TogglingMutationMethod _mutation_method;
    ElitismWithRandomMethod _elite_selection_method;
    RankBasedSelectionMethod _individuals_selection_method;
};



#endif // EVOLUTIONENGINE_H
