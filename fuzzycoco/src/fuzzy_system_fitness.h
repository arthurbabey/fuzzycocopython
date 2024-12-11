/**
  * @file   fuzzy_system_fitness.h
  * @author Karl Forner <karl.forner@gmail.com>
  * @author Lonza
  * @date   09.2024
  * @class FuzzySystemFitness
  *
  * @brief an asbtract base class for computing the fitness of a Fuzzy System
  */

#ifndef FUZZY_SYSTEM_FITNESS_H
#define FUZZY_SYSTEM_FITNESS_H

#include "fuzzy_system_metrics.h"

class FuzzySystemFitness
{
public:
  FuzzySystemFitness() {}
  virtual ~FuzzySystemFitness() {}

  virtual double fitness(const FuzzySystemMetrics& metrics);
};

class FuzzySystemWeightedFitness : public FuzzySystemFitness {
public:
  FuzzySystemWeightedFitness(const FuzzySystemMetrics& weights) : _weights(weights) {}
  double fitness(const FuzzySystemMetrics& metrics) override;

private:
  FuzzySystemMetrics _weights;
};

#endif // FUZZY_SYSTEM_FITNESS_H
