#include "coevolution_engine.h"

double CoopCoevolutionFitnessMethod::coopFitness(bool left, const Genome& genome, const Genomes& cooperators) {
    double max_fitness = numeric_limits<double>::lowest();
    double fit = -1;
    for (const auto& coop : cooperators)
        if (left)
            fit = fitness(genome, coop);
        else
            fit = fitness(coop, genome);
        max_fitness = max(max_fitness, fit);
      
    return max_fitness;
}


// vector<double> CoopCoevolutionFitnessMethod::coopFitness(bool left, const Genomes& genomes, const Genomes& cooperators)
// {
//     cerr << "CoopCoevolutionFitnessMethod::coopFitness():" << "left=" << left << " " << genomes[0].size() << "-" << cooperators[0].size() << endl;
//     const int nb = genomes.size();
//     vector<double> fitnesses(nb);
//     for (int i = 0; i < nb; i++)
//         fitnesses[i] = coopFitness(left, genomes[i], cooperators);
//     return fitnesses;
// }

CoevGeneration CoevolutionEngine::start(const Genomes& left_genos, const Genomes& right_genos)
{
    CoevGeneration cogen(left_genos, right_genos);
    Genomes left_coops = selectCooperators(left_genos, _nb_cooperators);
    Genomes right_coops = selectCooperators(right_genos, _nb_cooperators); 
    // auto left_fitnesses = _fit.coopFitness(true, left_genos, right_coops);
    // auto right_fitnesses = _fit.coopFitness(false, right_genos, left_coops);

    cogen.left_gen.elite = _left_engine.selectElite(left_genos);
    cogen.right_gen.elite = _right_engine.selectElite(right_genos);

    // only update cogen in-place: set genome fitnesses and elite
    initGeneration(cogen);

    return cogen;
}


pair<CoevGeneration, vector<double>> CoevolutionEngine::evolve(const Genomes& left_genos, const Genomes& right_genos, int nb_generations, double maxFit)
{
    // make initial generation: need to select cooperators, compute fitness etc...
    CoevGeneration cogen = start(left_genos, right_genos);
    vector<double> fitnesses;

    vector<double> generation_fitnesses;
    generation_fitnesses.reserve(nb_generations);
    for (int i = 0; i < nb_generations; i++) {
        cogen = nextGeneration(cogen, false);
        // cerr << cogen.fitness() << ", ";
        generation_fitnesses.push_back(cogen.fitness());
        if (cogen.fitness() >= maxFit) break; // early return if we reach the max fitness
    }
    // cerr << endl;

    return make_pair(cogen, generation_fitnesses);
}

// performs selection and reproduction
CoevGeneration CoevolutionEngine::nextGeneration(CoevGeneration& cogen, bool only_update)
{
    CoevGeneration newcogen;
    newcogen.generation_number = cogen.generation_number + 1;

    Genomes right_coops = selectCooperators(cogen.right_gen.elite, _nb_cooperators);
    CoopCoevolutionFitnessMethodAdaptator left_fit(true, _fit, right_coops);
    if (only_update)
        _left_engine.updateGeneration(cogen.left_gen, left_fit);
    else 
        newcogen.left_gen =_left_engine.nextGeneration(cogen.left_gen, left_fit);

    Genomes left_coops = selectCooperators(cogen.left_gen.elite, _nb_cooperators);
    CoopCoevolutionFitnessMethodAdaptator right_fit(false, _fit, left_coops);
    
    if (only_update)
        _right_engine.updateGeneration(cogen.right_gen, right_fit);
    else 
        newcogen.right_gen = _right_engine.nextGeneration(cogen.right_gen, right_fit);

    newcogen._fitness = max(newcogen.left_gen.fitness, newcogen.right_gen.fitness);

    return newcogen;
}   

//  pair<Genomes, Genomes> CoevolutionEngine::computeBest(const CoevGeneration& coevgen)
//  {
//     // start with the left ones
//     Genomes left_best = EvolutionEngine::selectBest(coevgen.left_gen);


//     Genomes right_best = EvolutionEngine::selectBest(coevgen.right_gen);
//     return make_pair(left_best, right_best);
//  }

Genomes CoevolutionEngine::selectCooperators(const Genomes& elite, int nb_cooperators)
{
    if (nb_cooperators >= elite.size()) return elite;
    return Genomes(elite.begin(), elite.begin() + nb_cooperators);
}