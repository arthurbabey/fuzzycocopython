from __future__ import annotations

import importlib
import importlib.machinery
import importlib.util
import sys
from pathlib import Path


_PKG_DIR = Path(__file__).resolve().parent
_CORE_MODULE_NAME = f"{__name__}._fuzzycoco_core"
_loaded_core = sys.modules.get(_CORE_MODULE_NAME)
_needs_reload = True
if _loaded_core is not None:
    try:
        _needs_reload = Path(_loaded_core.__file__).resolve().parent != _PKG_DIR
    except (OSError, AttributeError):
        _needs_reload = True

if _needs_reload:
    for _suffix in importlib.machinery.EXTENSION_SUFFIXES:
        candidate = _PKG_DIR / f"_fuzzycoco_core{_suffix}"
        if candidate.exists():
            spec = importlib.util.spec_from_file_location(_CORE_MODULE_NAME, candidate)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            sys.modules[_CORE_MODULE_NAME] = module
            break
    else:  # pragma: no cover - fallback to whatever is already importable
        importlib.import_module(f".{'_fuzzycoco_core'}", __name__)


from .fuzzycoco_base import (
    FuzzyCocoClassifier,
    FuzzyCocoRegressor,
    load_model,
    save_model,
)

#__all__ = ["FuzzyCocoClassifier", "FuzzyCocoRegressor", "FuzzyCocoBase"]
