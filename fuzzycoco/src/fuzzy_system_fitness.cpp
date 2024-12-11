#include "fuzzy_system_fitness.h"
#include <cmath>

double FuzzySystemFitness::fitness(const FuzzySystemMetrics& metrics) {
  int nb_vars = max(1, metrics.nb_vars);
  double fit =
    metrics.sensitivity
    + metrics.specificity
    + metrics.accuracy
    + metrics.ppv
    + pow(2, -metrics.rmse )
    + pow(2, -metrics.rrse )
    + pow(2, -metrics.rae )
    + pow(2, -metrics.mse )
    + 1.0 / nb_vars;

  return fit;
}

double FuzzySystemWeightedFitness::fitness(const FuzzySystemMetrics& metrics) {
  double num =
    _weights.sensitivity * metrics.sensitivity
    + _weights.specificity * metrics.specificity
    + _weights.accuracy * + metrics.accuracy
    + _weights.ppv * metrics.ppv
    + _weights.rmse * pow(2, -metrics.rmse )
    + _weights.rrse * pow(2, -metrics.rrse )
    + _weights.rae * pow(2, -metrics.rae )
    + _weights.mse * pow(2, -metrics.mse )
    + _weights.nb_vars * (1.0  / max(1, metrics.nb_vars));

  double denum =
    _weights.sensitivity
    + _weights.specificity
    + _weights.accuracy
    + _weights.ppv
    + _weights.rmse
    + _weights.rrse
    + _weights.rae
    + _weights.mse
    + _weights.nb_vars;

  return num / denum;
}
