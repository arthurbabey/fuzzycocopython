from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

target_names = iris.target_names
feature_names = iris.feature_names
feature_names = [name.replace(' (cm)', '') for name in feature_names]
feature_names = [name.replace(' ', '_') for name in feature_names]

from fuzzycocopython.fuzzycoco_base import FuzzyCocoClassifier

model = FuzzyCocoClassifier(nb_rules=5, nb_max_var_per_rule=5, max_generations=100, rules_pop_size=50)
model.fit(X_train, y_train, feature_names=feature_names)


print(model.description_)
print()
print()
print(model.rules_)
print(model.variables_)
print(model.default_rules_)
print(model.feature_names_in_)

print(model.score(X_test, y_test))