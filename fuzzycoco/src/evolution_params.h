
#ifndef EVOLUTION_PARAMS_H
#define EVOLUTION_PARAMS_H

#include <string>
#include <iostream>
#include "dataframe.h"
struct EvolutionParams
{
    // the size (nb of genomes) of the population to evolve
    int pop_size = MISSING_DATA_INT;
    // the number of elite individuals
    int elite_size = 5;
    // the crossover probability
    double cx_prob = 0.5;
    // the probability that a genome is a target for a mutation
    double mut_flip_genome = 0.5;
    // the probability that a bit of a genome is mutated
    double mut_flip_bit = 0.025;

    bool has_missing() const {
        return is_na(pop_size) || is_na(elite_size) || is_na(cx_prob) || is_na(mut_flip_genome) ||  is_na(mut_flip_bit);
    }

    bool operator!=(const EvolutionParams& p) const { return ! (*this == p); }
    bool operator==(const EvolutionParams& p) const {
        return
            pop_size == p.pop_size &&
            // maxGen == p.maxGen &&
            // maxFit == p.maxFit &&
            elite_size == p.elite_size &&
            // nb_evolvers == p.nb_evolvers &&
            cx_prob == p.cx_prob &&
            mut_flip_genome == p.mut_flip_genome &&
            mut_flip_bit == p.mut_flip_bit;
    }

    inline friend ostream& operator<<(ostream& out, const EvolutionParams& p) {
        DataFrame df(1, 5);
        df.colnames({"pop_size", "elite_size", "cx_prob", "mut_flip_genome", "mut_flip_bit"});
        auto D = [](int i) { return is_na(i) ? MISSING_DATA_DOUBLE : double(i); };
        vector<double> row = {D(p.pop_size), D(p.elite_size), D(p.cx_prob), D(p.mut_flip_genome), D(p.mut_flip_bit)};
        df.fillRow(0, row);
        out << df;

        return out;
  }
};
#endif // EVOLUTION_PARAMS_H
