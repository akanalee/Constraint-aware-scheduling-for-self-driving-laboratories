import os

from case_study.olympus_env import configure

configure()

import tensorflow
import olympus
from olympus.emulators.emulator import load_emulator

TF_VERSION = tensorflow.__version__

_EMULATOR_DIR = os.path.join(os.path.dirname(os.path.abspath(olympus.__file__)), 'emulators')

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
