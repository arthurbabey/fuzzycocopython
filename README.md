# FuzzyCocoPython

## Installation

Clone the repository and install locally:
```bash
git clone https://github.com/arthurbabey/fuzzycocopython.git
cd fuzzycocopython
pip install .
```

Or install directly from GitHub:
```bash
pip install git+https://github.com/arthurbabey/fuzzycocopython.git
```


# ---------- save ----------
with open("model.pkl", "wb") as f:
    pickle.dump(clf, f)

# ---------- load ----------
with open("model.pkl", "rb") as f:
    clf2 = pickle.load(f)