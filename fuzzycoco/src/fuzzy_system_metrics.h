/**
  * @file   fuzzy_system_metrics.h
  * @author Karl Forner <karl.forner@gmail.com>
  * @author Lonza
  * @date   09.2024
  * @struct FuzzySystemMetrics
  * @brief a structure that contains a FuzzySystem Performance metrics
  */

#ifndef FUZZY_SYSTEM_METRICS_H
#define FUZZY_SYSTEM_METRICS_H

#include <string>
#include <iostream>
#include <vector>
using namespace std;
#include "types.h"
#include "named_list.h"

struct FuzzySystemMetrics
{
  double sensitivity;
  double specificity;
  double accuracy;
  double ppv;
  double rmse;
  double rrse;
  double rae;
  double mse;
  double distanceThreshold;
  double distanceMinThreshold;
  int nb_vars; // the number of variables used in the system (can be used to penalize huge systems)
  double overLearn;
  int true_positives;
  int false_positives;
  int true_negatives;
  int false_negatives;

  FuzzySystemMetrics() { reset(); }

  void reset() {
    true_positives = false_positives = true_negatives = false_negatives = 0;
    sensitivity = 0;
    specificity = 0;
    accuracy = 0;
    ppv = 0;
    rmse = 0;
    rrse = 0;
    rae = 0;
    mse = 0;
    distanceThreshold = 0;
    distanceMinThreshold = 0;
    nb_vars = 0;
    overLearn = 0;
  }

  bool operator==(const FuzzySystemMetrics& p) const {
    return
        sensitivity == p.sensitivity &&
        specificity == p.specificity &&
        accuracy == p.accuracy &&
        ppv == p.ppv &&
        rmse == p.rmse &&
        rrse == p.rrse &&
        rae == p.rae &&
        mse == p.mse &&
        distanceThreshold == p.distanceThreshold &&
        distanceMinThreshold == p.distanceMinThreshold &&
        nb_vars == p.nb_vars &&
        overLearn == p.overLearn &&
        true_positives == p.true_positives &&
        false_positives == p.false_positives &&
        true_negatives == p.true_negatives &&
        false_negatives == p.false_negatives;
  }

  void operator+=(const FuzzySystemMetrics& m) {
    double v = 0;
    sensitivity += !is_na(v = m.sensitivity) ? v : 0;
    specificity += !is_na(v = m.specificity) ? v : 0;
    // sensitivity += m.sensitivity;
    // specificity += m.specificity;
    accuracy += !is_na(v = m.accuracy) ? v : 0;
    ppv += !is_na(v = m.ppv) ? v : 0;
    rmse += !is_na(v = m.rmse) ? v : 0;
    rrse += !is_na(v = m.rrse) ? v : 0;
    rae += !is_na(v = m.rae) ? v : 0;
    mse += !is_na(v = m.mse) ? v : 0;
    distanceThreshold += !is_na(v = m.distanceThreshold) ? v : 0;
    distanceMinThreshold += !is_na(v = m.distanceMinThreshold) ? v : 0;
    // nb_vars += !is_na(v = m.nb_vars) ? v : 0;
    overLearn += !is_na(v = m.overLearn) ? v : 0;
    true_positives += !is_na(v = m.true_positives) ? v : 0;
    true_negatives += !is_na(v = m.true_negatives) ? v : 0;
    false_positives += !is_na(v = m.false_positives) ? v : 0;
    false_negatives += !is_na(v = m.false_negatives) ? v : 0;
  }

  NamedList describe() const {
    NamedList desc;
    desc.add("sensitivity", sensitivity);
    desc.add("specificity", specificity);
    desc.add("accuracy", accuracy);
    desc.add("ppv", ppv);
    desc.add("rmse", rmse);
    desc.add("rrse", rrse);
    desc.add("rae", rae);
    desc.add("mse", mse);
    desc.add("distanceThreshold", distanceThreshold);
    desc.add("distanceMinThreshold", distanceMinThreshold);
    desc.add("nb_vars", nb_vars);
    desc.add("overLearn", overLearn);
    desc.add("true_positives", true_positives);
    desc.add("false_positives", false_positives);
    desc.add("true_negatives", true_negatives);
    desc.add("false_negatives", false_negatives);

    return desc;
  }

  static FuzzySystemMetrics load(const NamedList& desc) {
    FuzzySystemMetrics m;
    m.sensitivity = desc.get_double("sensitivity");
    m.specificity = desc.get_double("specificity");
    m.accuracy = desc.get_double("accuracy");
    m.ppv = desc.get_double("ppv");
    m.rmse = desc.get_double("rmse");
    m.rrse = desc.get_double("rrse");
    m.rae = desc.get_double("rae");
    m.mse = desc.get_double("mse");
    m.distanceThreshold = desc.get_double("distanceThreshold");
    m.distanceMinThreshold = desc.get_double("distanceMinThreshold");
    m.nb_vars = desc.get_int("nb_vars");
    m.overLearn = desc.get_double("overLearn");
    m.true_positives = desc.get_int("true_positives");
    m.false_positives = desc.get_int("false_positives");
    m.true_negatives = desc.get_int("true_negatives");
    m.false_negatives = desc.get_int("false_negatives");

    return m;
  }

  inline friend ostream& operator<<(ostream& out, const FuzzySystemMetrics& p) {
    out << "FuzzySystemMetrics: ("
    << "sensitivity=" << p.sensitivity << ", "
    << "specificity=" << p.specificity << ", "
    << "accuracy=" << p.accuracy << ", "
    << "ppv=" << p.ppv << ", "
    << "rmse=" << p.rmse << ", "
    << "rrse=" << p.rrse << ", "
    << "rae=" << p.rae << ", "
    << "mse=" << p.mse << ", "
    << "distanceThreshold=" << p.distanceThreshold << ", "
    << "distanceMinThreshold=" << p.distanceMinThreshold << ", "
    << "nb_vars=" << p.nb_vars << ", "
    << "overLearn=" << p.overLearn  << ", "
    << "TP=" << p.true_positives << ", "
    << "FP=" << p.false_positives << ", "
    << "TN=" << p.true_negatives << ", "
    << "FN=" << p.false_negatives << ", "
    << ")";

    return out;
  }
};

#endif // FUZZY_SYSTEM_METRICS_H
