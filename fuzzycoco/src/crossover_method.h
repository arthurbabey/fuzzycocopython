#ifndef CROSSOVER_METHOD_H
#define CROSSOVER_METHOD_H

#include "genome.h"
#include "random_generator.h"

class CrossoverMethod {
public:
    CrossoverMethod() {}
    virtual ~CrossoverMethod() {}
    virtual void reproducePairOf(Genome& gen1, Genome& gen2) = 0;
    virtual void reproduceAllPairsOf(vector<Genome>& genomes);
};

class OnePointCrossoverMethod : public CrossoverMethod {
public:
    OnePointCrossoverMethod(RandomGenerator& rng, double prob) : _rng(rng), _prob(prob) {}
    virtual void reproducePairOf(Genome& gen1, Genome& gen2) override;
private:
    RandomGenerator& _rng;
    double _prob;
};


#endif // CROSSOVER_METHOD_H

