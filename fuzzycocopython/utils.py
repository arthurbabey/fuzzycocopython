import json
import os
import uuid
import math 

import numpy as np
from lfa_toolbox.core.lv.linguistic_variable import LinguisticVariable
from lfa_toolbox.core.mf.lin_piece_wise_mf import (
    LeftShoulderMF,
    RightShoulderMF,
    TriangularMF,
)
from lfa_toolbox.core.mf.singleton_mf import SingletonMF
from lfa_toolbox.core.rules.default_fuzzy_rule import DefaultFuzzyRule
from lfa_toolbox.core.rules.fuzzy_rule import FuzzyRule
from lfa_toolbox.core.rules.fuzzy_rule_element import Antecedent, Consequent

from ._fuzzycoco_core import (
    GlobalParams,
    VarsParams,
    EvolutionParams,
    FitnessParams,
    FuzzyCocoParams,
    DataFrame,

)
def create_input_linguistic_variable(var_name, set_items):
    set_items = sorted(set_items, key=lambda s: s.get("position", 0))
    n = len(set_items)
    ling_values_dict = {}
    for i, s in enumerate(set_items):
        pos = s.get("position")
        # We ignore the original "name" and use the provided generic label.
        set_name = s.get("name")
        if n == 1:
            mf = LeftShoulderMF(pos, pos + 1)
        elif n == 2:
            if i == 0:
                mf = LeftShoulderMF(pos, set_items[i + 1]["position"])
            else:
                mf = RightShoulderMF(set_items[i - 1]["position"], pos)
        else:
            if i == 0:
                mf = LeftShoulderMF(pos, set_items[i + 1]["position"])
            elif i == n - 1:
                mf = RightShoulderMF(set_items[i - 1]["position"], pos)
            else:
                mf = TriangularMF(
                    set_items[i - 1]["position"], pos, set_items[i + 1]["position"]
                )
        ling_values_dict[set_name] = mf
    return LinguisticVariable(var_name, ling_values_dict)


def create_output_linguistic_variable(var_name, set_items):
    set_items = sorted(set_items, key=lambda s: s.get("position", 0))
    ling_values_dict = {}
    for s in set_items:
        pos = s.get("position")
        set_name = s.get("name")
        mf = SingletonMF(pos)
        ling_values_dict[set_name] = mf
    return LinguisticVariable(var_name, ling_values_dict)


def generate_generic_labels(n):
    match n:
        case 1:
            return ["Medium"]
        case 2:
            return ["Low", "High"]
        case 3:
            return ["Low", "Medium", "High"]
        case 4:
            return ["Very Low", "Low", "High", "Very High"]
        case 5:
            return ["Very Low", "Low", "Medium", "High", "Very High"]
        case 6:
            return [
                "Very Low",
                "Low",
                "Slightly Low",
                "Slightly High",
                "High",
                "Very High",
            ]
        case 7:
            return [
                "Very Low",
                "Low",
                "Slightly Low",
                "Medium",
                "Slightly High",
                "High",
                "Very High",
            ]
        case _:
            return [f"Set {i+1}" for i in range(n)]


def parse_fuzzy_system_from_coco(coco):
    """
    Parse the internal FuzzySystem of a FuzzyCoco object and return:
      • list  linguistic_variables
      • list  fuzzy_rules
      • list  default_rules

    The helper symbols assumed to exist in your code-base:
      generate_generic_labels
      create_input_linguistic_variable
      create_output_linguistic_variable
      Antecedent, Consequent, FuzzyRule, DefaultFuzzyRule
    """
    fs = coco.get_fuzzy_system()          # ← accessor you just bound

    # ------------------------------------------------------------------ variables
    linguistic_variables = []
    lv_dict       = {}      # var_name -> LinguisticVariable
    label_mapping = {}      # var_name -> {orig_set : generic_label}

    # -------- input vars
    for var in fs.get_input_variables():                  # [{'name': …, 'sets':[...]}, ...]
        name  = var["name"]
        sets  = var["sets"]
        glabs = generate_generic_labels(len(sets))
        mapping, new_sets = {}, []
        for g, s in zip(glabs, sets):
            mapping[s["name"]] = g
            ns = dict(s); ns["name"] = g
            new_sets.append(ns)
        label_mapping[name] = mapping
        lv = create_input_linguistic_variable(name, new_sets)
        linguistic_variables.append(lv)
        lv_dict[name] = lv

    # -------- output vars
    for var in fs.get_output_variables():
        name, sets = var["name"], var["sets"]
        glabs = generate_generic_labels(len(sets))
        mapping, new_sets = {}, []
        for g, s in zip(glabs, sets):
            mapping[s["name"]] = g
            ns = dict(s); ns["name"] = g
            new_sets.append(ns)
        label_mapping[name] = mapping
        lv = create_output_linguistic_variable(name, new_sets)
        linguistic_variables.append(lv)
        lv_dict[name] = lv

    # ------------------------------------------------------------------ rules
    fuzzy_rules   = []
    default_rules = []

    fixed_act_func  = (np.min, "AND_min")
    fixed_impl_func = (np.min, "AND_min")

    for rule in fs.get_rules():           # list[dict] with antecedents / consequents
        ants = []
        cons = []

        for ant in rule.get("antecedents", []):
            v  = ant["var_name"]
            lab= label_mapping[v].get(ant["set_name"], ant["set_name"])
            ants.append(Antecedent(lv_dict[v], lab, is_not=False))

        for con in rule.get("consequents", []):
            v  = con["var_name"]
            lab= label_mapping[v].get(con["set_name"], con["set_name"])
            cons.append(Consequent(lv_dict[v], lab))

        if ants:
            fuzzy_rules.append(FuzzyRule(ants, fixed_act_func, cons, fixed_impl_func))
        else:
            default_rules.append(DefaultFuzzyRule(cons, fixed_impl_func))

    return linguistic_variables, fuzzy_rules, default_rules



# ───────────────────────── internal helpers ────────────────────────────
def _bits_for_nb(nb):
    return 0 if nb < 2 else int(math.ceil(math.log2(nb)))

def _make_dataframe(mat, header):
    rows = [header] + [[str(v) for v in row] for row in mat]
    return DataFrame(rows, False)

def _build_cpp_params(
    *,
    n_features,
    # global
    nb_rules,
    nb_max_var_per_rule,
    max_generations,
    max_fitness,
    nb_cooperators,
    influence_rules_initial_population,
    influence_evolving_ratio,
    # vars
    nb_sets_in,
    nb_bits_vars_in,
    nb_bits_sets_in,
    nb_bits_pos_in,
    nb_sets_out,
    nb_bits_vars_out,
    nb_bits_sets_out,
    nb_bits_pos_out,
    # GA
    rules_pop_size,
    mfs_pop_size,
    elite_size,
    cx_prob,
    mut_flip_genome,
    mut_flip_bit,
    # fitness
    threshold,
    metrics_weights,
    features_weights,
):
    gp = GlobalParams()
    gp.nb_rules = nb_rules
    gp.nb_max_var_per_rule = nb_max_var_per_rule
    gp.max_generations = max_generations
    gp.max_fitness = max_fitness
    gp.nb_cooperators = nb_cooperators
    gp.influence_rules_initial_population = influence_rules_initial_population
    gp.influence_evolving_ratio = influence_evolving_ratio

    iv = VarsParams()
    iv.nb_sets       = nb_sets_in
    iv.nb_bits_vars  = nb_bits_vars_in or (_bits_for_nb(n_features) + 1)
    iv.nb_bits_sets  = nb_bits_sets_in or _bits_for_nb(nb_sets_in)
    iv.nb_bits_pos   = nb_bits_pos_in     # default 8 provided in __init__

    ov = VarsParams()
    ov.nb_sets       = nb_sets_out
    ov.nb_bits_vars  = nb_bits_vars_out or 1
    ov.nb_bits_sets  = nb_bits_sets_out or _bits_for_nb(nb_sets_out)
    ov.nb_bits_pos   = nb_bits_pos_out    # default 8 provided in __init__

    rp = EvolutionParams()
    rp.pop_size        = rules_pop_size
    rp.elite_size      = elite_size
    rp.cx_prob         = cx_prob
    rp.mut_flip_genome = mut_flip_genome
    rp.mut_flip_bit    = mut_flip_bit

    mp = EvolutionParams()
    mp.pop_size        = mfs_pop_size
    mp.elite_size      = elite_size
    mp.cx_prob         = cx_prob
    mp.mut_flip_genome = mut_flip_genome
    mp.mut_flip_bit    = mut_flip_bit

    fp = FitnessParams()
    fp.output_vars_defuzz_thresholds = [threshold]
    if metrics_weights is not None:
        fp.metrics_weights = metrics_weights
    if features_weights is not None:
        fp.features_weights = features_weights

    params = FuzzyCocoParams()
    params.global_params      = gp
    params.input_vars_params  = iv
    params.output_vars_params = ov
    params.rules_params       = rp
    params.mfs_params         = mp
    params.fitness_params     = fp
    return params