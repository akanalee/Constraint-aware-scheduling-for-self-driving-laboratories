#!/usr/bin/env python
"""
_olympus_bnn_loader.py  -  Shared Olympus BayesNeuralNet emulator loader.

Sidesteps three issues with stock Olympus on this Windows/TF 2.20 env:
  1) TF2's keras.Sequential rejects tfp DenseLocalReparameterization layers
     unless TF_USE_LEGACY_KERAS=1 is set before `import tensorflow`.
  2) `olympus.models.__init__` parses emulator directory names with '/' which
     breaks on Windows — we call `load_emulator(folder)` directly to bypass
     the `_validate_model_kind` path.
  3) olympus.models.model_bayes_neural_net imports `silence_tensorflow`;
     we stub it out (we already silence TF via env vars) to avoid its
     re-import of TF that can fail with DLL errors after other imports.

Exposes `get_emulator(name)` which loads once and caches.
"""

import os
import sys
import types

# Must be set before TF is imported anywhere in this process.
os.environ.setdefault('TF_USE_LEGACY_KERAS', '1')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

# Stub silence_tensorflow so olympus.model_bayes_neural_net doesn't re-load TF.
if 'silence_tensorflow' not in sys.modules:
    _stub = types.ModuleType('silence_tensorflow')
    _stub.silence_tensorflow = lambda: None
    sys.modules['silence_tensorflow'] = _stub

import tensorflow as tf  # noqa: E402  triggers the legacy-keras-aware load
_ = tf.__version__

# Use the TOP-LEVEL olympus-main (not the nested DRL_scheduler_for_SDL copy),
# which has a working datasets/__init__.py without the circular import.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_OLYMPUS_SRC = os.path.normpath(os.path.join(
    _THIS_DIR, '..', '..', 'olympus-main', 'olympus-main', 'src'
))
if _OLYMPUS_SRC not in sys.path:
    sys.path.insert(0, _OLYMPUS_SRC)

from olympus.emulators.emulator import load_emulator  # noqa: E402

_EMULATOR_DIR = os.path.join(
    _OLYMPUS_SRC, 'olympus', 'emulators'
)

_cache: dict = {}

def get_emulator(name: str):
    if not name.startswith('emulator_'):
        name = f'emulator_{name}'
    if name in _cache:
        return _cache[name]
    folder = os.path.join(_EMULATOR_DIR, name)
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Olympus emulator not found: {folder}")
    e = load_emulator(folder)
    _cache[name] = e
    return e
