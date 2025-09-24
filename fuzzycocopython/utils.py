import numpy as np
import pandas as pd

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
from typing import Dict, List, Tuple

from ._fuzzycoco_core import (
    GlobalParams,
    VarsParams,
    EvolutionParams,
    FitnessParams,
    FuzzyCocoParams,
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


def _auto_bits(n):
    if n is None or n < 2:
        return 0
    return int(np.ceil(np.log2(n)))


def make_fuzzy_params(params=None, **flat_kwargs):
    params = params.copy() if params else {}

    def get(sub, key, default):
        return params.get(sub, {}).get(key, flat_kwargs.get(key, default))

    # Global
    g = GlobalParams()
    g.nb_rules = get("global_params", "nb_rules", 5)
    g.nb_max_var_per_rule = get("global_params", "nb_max_var_per_rule", None)
    g.max_generations = get("global_params", "max_generations", 100)
    g.max_fitness = get("global_params", "max_fitness", 1.0)
    g.nb_cooperators = get("global_params", "nb_cooperators", 2)
    g.influence_rules_initial_population = get("global_params", "influence_rules_initial_population", False)
    g.influence_evolving_ratio = get("global_params", "influence_evolving_ratio", 0.8)

    # Input vars
    in_sets = get("input_vars_params", "nb_sets", 2)
    in_vars = VarsParams()
    in_vars.nb_sets = in_sets
    in_vars.nb_bits_vars = get("input_vars_params", "nb_bits_vars",
                            _auto_bits(in_sets) + 1)     # NOMBRE DE VARIABLE  # ← default if None
    in_vars.nb_bits_sets = get("input_vars_params", "nb_bits_sets",
                            _auto_bits(in_sets))
    in_vars.nb_bits_pos = get("input_vars_params", "nb_bits_pos", 8) # 8 surement trop grand

    # Output vars
    out_sets = get("output_vars_params", "nb_sets", 2)
    out_vars = VarsParams()
    out_vars.nb_sets = out_sets
    out_vars.nb_bits_vars = get("output_vars_params", "nb_bits_vars",
                                _auto_bits(out_sets) + 1)     # NOMBRE DE VARIABLE  # ← default if None
    out_vars.nb_bits_sets = get("output_vars_params", "nb_bits_sets",
                                _auto_bits(out_sets))
    out_vars.nb_bits_pos = get("output_vars_params", "nb_bits_pos", 8) # 8 surement trop grand

    # Evolution params
    rules = EvolutionParams()
    rules.pop_size        = get("rules_params", "pop_size", 50)      # default 50
    rules.elite_size      = get("rules_params", "elite_size", 5)
    rules.cx_prob         = get("rules_params", "cx_prob", 0.5)
    rules.mut_flip_genome = get("rules_params", "mut_flip_genome", 0.5)
    rules.mut_flip_bit    = get("rules_params", "mut_flip_bit", 0.025)

    mfs = EvolutionParams()
    mfs.pop_size        = get("mfs_params", "pop_size", 50)          # default 50
    mfs.elite_size      = get("mfs_params", "elite_size", 5)
    mfs.cx_prob         = get("mfs_params", "cx_prob", 0.5)
    mfs.mut_flip_genome = get("mfs_params", "mut_flip_genome", 0.5)
    mfs.mut_flip_bit    = get("mfs_params", "mut_flip_bit", 0.025)

    # Fitness
    fit = FitnessParams()
    fit.output_vars_defuzz_thresholds = [
        get("fitness_params", "threshold", flat_kwargs.get("threshold", 0.5))
    ]
    metrics_weights = get(
        "fitness_params", "metrics_weights", flat_kwargs.get("metrics_weights", {})
    )
    for k, v in metrics_weights.items():
        setattr(fit.metrics_weights, k, v)
    fit.features_weights = get(
        "fitness_params", "features_weights", flat_kwargs.get("features_weights", {})
    )

    p = FuzzyCocoParams()
    p.global_params = g
    p.input_vars_params = in_vars
    p.output_vars_params = out_vars
    p.rules_params = rules
    p.mfs_params = mfs
    p.fitness_params = fit
    return p


def _parse_variables(desc):
    fs = desc.get("fuzzy_system", {})
    return fs.get("variables", {})

def _parse_rules(desc):
    fs = desc.get("fuzzy_system", {})
    return fs.get("rules", {})

def _parse_default_rules(desc):
    fs = desc.get("fuzzy_system", {})
    return fs.get("default_rules", {})



def parse_fuzzy_system_from_description(desc):
    """High‑level aggregator used by fit() to expose .variables_, .rules_, etc."""
    return (
        _parse_variables(desc),
        _parse_rules(desc),
        _parse_default_rules(desc),
    )


def to_linguistic_components(variables_dict, rules_dict, default_rules_dict):
    """
    Takes variables_/rules_/default_rules_ (parsed from description)
    and returns:
      linguistic_variables, fuzzy_rules, default_rules
    using the user-provided helper factories:
      - create_input_linguistic_variable
      - create_output_linguistic_variable
    """
    # ---- build LinguisticVariable objects ------------------------
    linguistic_variables = []
    lv_by_name           = {}
    label_map            = {}          # var -> {orig_set: generic_label}

    # input vars
    for var_name, sets in variables_dict.get("input", {}).items():
        # sets: {'sepal_length.1': pos, 'sepal_length.2': pos, ...}
        set_items = [{"name": k, "position": v} for k, v in sets.items()]
        generic = generate_generic_labels(len(set_items))
        for s, lbl in zip(set_items, generic):
            label_map.setdefault(var_name, {})[s["name"]] = lbl
            s["name"] = lbl
        lv = create_input_linguistic_variable(var_name, set_items)
        linguistic_variables.append(lv)
        lv_by_name[var_name] = lv

    # output vars
    for var_name, sets in variables_dict.get("output", {}).items():
        set_items = [{"name": k, "position": v} for k, v in sets.items()]
        generic = generate_generic_labels(len(set_items))
        for s, lbl in zip(set_items, generic):
            label_map.setdefault(var_name, {})[s["name"]] = lbl
            s["name"] = lbl
        lv = create_output_linguistic_variable(var_name, set_items)
        linguistic_variables.append(lv)
        lv_by_name[var_name] = lv

    # ---- build FuzzyRule / DefaultFuzzyRule ----------------------
    fixed_act = (np.min, "AND_min")
    fixed_imp = (np.min, "AND_min")
    fuzzy_rules     = []
    default_rules   = []

    for _, rule in rules_dict.items():
        ants = []
        for var, mf_dict in rule["antecedents"].items():
            orig_set = next(iter(mf_dict.keys()))
            label    = label_map[var][orig_set]
            ants.append(Antecedent(lv_by_name[var], label, is_not=False))

        cons = []
        for var, mf_dict in rule["consequents"].items():
            orig_set = next(iter(mf_dict.keys()))
            label    = label_map[var][orig_set]
            cons.append(Consequent(lv_by_name[var], label))

        fuzzy_rules.append(FuzzyRule(ants, fixed_act, cons, fixed_imp))

    # default consequents
    for var, orig_set in default_rules_dict.items():
        label = label_map[var][orig_set]
        cons  = [Consequent(lv_by_name[var], label)]
        default_rules.append(DefaultFuzzyRule(cons, fixed_imp))

    return linguistic_variables, fuzzy_rules, default_rules


# ---- same helpers as before ----------------------------------------------

def _build_pos_and_label_maps(variables_dict: Dict) -> Tuple[Dict, Dict]:
    pos_index = {"input": {}, "output": {}}
    label_map = {"input": {}, "output": {}}
    for io in ("input", "output"):
        for var, sets in variables_dict.get(io, {}).items():
            items = sorted(sets.items(), key=lambda kv: kv[1])  # by position
            generic = generate_generic_labels(len(items))
            pos_index[io][var] = {}
            label_map[io][var] = {}
            for (orig_set, pos), gen in zip(items, generic):
                pos_index[io][var][gen] = {"position": float(pos), "orig_set": orig_set}
                label_map[io][var][orig_set] = gen
    return pos_index, label_map

def _format_var_lines_for_io(pos_index: Dict, io: str, digits: int = 6) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for var, label_info in pos_index.get(io, {}).items():
        label_items = sorted(label_info.items(), key=lambda kv: kv[1]["position"])
        out[var] = [
            f"{lbl} = {info['position']:.{digits}g} (from {info['orig_set']})"
            for lbl, info in label_items
        ]
    return out

def _lookup_label_and_pos(pos_index: Dict, label_map: Dict, var: str, orig_set: str) -> Tuple[str, float, str]:
    io = "input" if var in pos_index["input"] else "output"
    lbl = label_map[io][var][orig_set]
    pos = pos_index[io][var][lbl]["position"]
    return lbl, pos, io

def _rule_key(name: str) -> Tuple[int, str]:
    num = "".join(ch for ch in name if ch.isdigit())
    return (int(num) if num else 10**9, name)

# ---- VIEWS (human-readable text) -----------------------------------------

def to_views_components(
    variables_dict: Dict, rules_dict: Dict, default_rules_dict: Dict, *, digits: int = 6
) -> Tuple[Dict[str, List[str]], List[str], List[str]]:
    pos_index, label_map = _build_pos_and_label_maps(variables_dict)

    # flat variable map
    vars_in  = _format_var_lines_for_io(pos_index, "input", digits)
    vars_out = _format_var_lines_for_io(pos_index, "output", digits)
    variables_view: Dict[str, List[str]] = dict(vars_in)
    for var, lines in vars_out.items():
        if var not in variables_view:
            variables_view[var] = lines
        else:
            variables_view[f"{var} (output)"] = lines  # collision guard

    def ants_to_text(ant_dict: Dict) -> str:
        parts = []
        for var, mf_dict in ant_dict.items():
            orig_set = next(iter(mf_dict.keys()))
            lbl, pos, _ = _lookup_label_and_pos(pos_index, label_map, var, orig_set)
            parts.append(f"{var} is {lbl} ({pos:.{digits}g})")
        return " AND ".join(parts) if parts else "TRUE"

    def cons_to_text(cons_dict: Dict) -> str:
        parts = []
        for var, mf_dict in cons_dict.items():
            orig_set = next(iter(mf_dict.keys()))
            lbl, pos, _ = _lookup_label_and_pos(pos_index, label_map, var, orig_set)
            parts.append(f"{var} is {lbl} ({pos:.{digits}g})")
        return " AND ".join(parts) if parts else "(no consequent)"

    rules_view: List[str] = []
    for rname, rdef in sorted(rules_dict.items(), key=lambda kv: _rule_key(kv[0])):
        ants = ants_to_text(rdef.get("antecedents", {}))
        cons = cons_to_text(rdef.get("consequents", {}))
        rules_view.append(f"{rname.upper()}: IF {ants} THEN {cons}")

    default_rules_view: List[str] = []
    for var, orig_set in default_rules_dict.items():
        lbl, pos, _ = _lookup_label_and_pos(pos_index, label_map, var, orig_set)
        default_rules_view.append(f"DEFAULT: {var} is {lbl} ({pos:.{digits}g})")

    return variables_view, rules_view, default_rules_view

# ---- TABLES (machine-friendly; great in notebooks) -----------------------

def to_tables_components(
    variables_dict: Dict, rules_dict: Dict, default_rules_dict: Dict
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    pos_index, label_map = _build_pos_and_label_maps(variables_dict)

    # variables_df
    rows_vars = []
    for io in ("input", "output"):
        for var, label_info in pos_index[io].items():
            for lbl, info in label_info.items():
                rows_vars.append({
                    "io": io,
                    "var": var,
                    "label": lbl,
                    "position": info["position"],
                    "orig_set": info["orig_set"],
                })
    variables_df = pd.DataFrame(rows_vars).sort_values(["io","var","position"]).reset_index(drop=True)

    # rules_df
    rows_rules = []
    for rname, rdef in sorted(rules_dict.items(), key=lambda kv: _rule_key(kv[0])):
        for role, part in (("antecedent", rdef.get("antecedents", {})),
                           ("consequent", rdef.get("consequents", {}))):
            for var, mf_dict in part.items():
                orig_set = next(iter(mf_dict.keys()))
                lbl, pos, io = _lookup_label_and_pos(pos_index, label_map, var, orig_set)
                rows_rules.append({
                    "rule": rname,
                    "role": role,
                    "io": io,
                    "var": var,
                    "label": lbl,
                    "position": pos,
                    "orig_set": orig_set,
                })
    # defaults as rule rows too (role='default')
    for var, orig_set in default_rules_dict.items():
        lbl, pos, io = _lookup_label_and_pos(pos_index, label_map, var, orig_set)
        rows_rules.append({
            "rule": "default",
            "role": "default",
            "io": io,
            "var": var,
            "label": lbl,
            "position": pos,
            "orig_set": orig_set,
        })

    rules_df = pd.DataFrame(rows_rules).sort_values(["rule","role","var"]).reset_index(drop=True)
    return variables_df, rules_df
