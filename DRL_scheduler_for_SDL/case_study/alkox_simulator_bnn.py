#!/usr/bin/env python
"""
alkox_simulator_bnn.py  -  Olympus BayesNeuralNet oracle for AlkOx biocatalytic
benzyl-alcohol oxidation.

Backed by olympus.emulators.emulator_alkox_BayesNeuralNet, which is pre-trained
on the 208-row alkox dataset. Returns the enzyme cascade conversion (regression
target, continuous, default goal=maximize).

Param space (from dataset config.json):
    catalase         continuous [0.05, 1.0]   mg/mL
    peroxidase       continuous [0.5,  10.0]  U/mL
    alcohol_oxidase  continuous [2.0,  8.0]   U/mL
    ph               continuous [6.0,  8.0]
Target: conversion (continuous)

API mirrors SuzukiReactionSimulator:
    sim = AlkOxReactionSimulator(seed=0)
    result = sim.query(bo_params)
    # -> {"conversion_pct": float}
"""

import os
import sys
import numpy as np
from typing import Dict, Optional, Tuple

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_OLYMPUS_SRC = os.path.normpath(os.path.join(
    _THIS_DIR, '..', 'olympus-main', 'olympus-main', 'src'
))
if _OLYMPUS_SRC not in sys.path:
    sys.path.insert(0, _OLYMPUS_SRC)

from case_study._olympus_bnn_loader import get_emulator


_PARAM_ORDER = ['catalase', 'peroxidase', 'alcohol_oxidase', 'ph']

_BOUNDS = {
    'catalase':        (0.05, 1.0),
    'peroxidase':      (0.5,  10.0),
    'alcohol_oxidase': (2.0,  8.0),
    'ph':              (6.0,  8.0),
}


class AlkOxReactionSimulator:
    """BNN-backed AlkOx enzymatic-cascade oracle.

    API-compatible with SuzukiReactionSimulator. The single regression output
    is exposed as `conversion_pct` (Olympus default unit; values fall in the
    0-100+ range as observed in dataset_alkox/data.csv).
    """

    NUM_BNN_SAMPLES = 20

    # Process-level memoization for query_noiseless. The underlying BNN is a
    # TensorFlow + tfp.layers.DenseLocalReparameterization model — its
    # Monte-Carlo predictions are stochastic (tf.random.normal in TF1 graph
    # mode, not seedable per-call without rebuilding the graph). Without
    # caching, repeating the same input gives different ensemble means each
    # call. That breaks two things:
    #   1) Warmup pre-scoring (in AlkOxBOOptimizer.__init__) selects
    #      "low-yield" points based on values that the run loop later
    #      disagrees with — high-yield points leak into the warmup set.
    #   2) DRL and Native episodes that query the same param vector get
    #      different conversions, so they end up training BO on different
    #      data and diverge.
    # Cache key: (base_seed, rounded-param-tuple). Rounding to 9 decimals
    # collapses float-rep noise without losing meaningful precision.
    _NOISELESS_CACHE: Dict[Tuple, float] = {}
    _CACHE_PARAM_DECIMALS = 9

    def __init__(self, data_path: Optional[str] = None, seed: Optional[int] = None):
        _ = data_path
        self.rng = np.random.RandomState(seed)
        self._base_seed = int(seed) if seed is not None else 0
        self._emu = get_emulator('alkox_BayesNeuralNet')

    @classmethod
    def _cache_key(cls, base: int, params: Dict) -> Tuple:
        return (int(base),) + tuple(
            round(float(params[k]), cls._CACHE_PARAM_DECIMALS) for k in _PARAM_ORDER
        )

    @classmethod
    def clear_noiseless_cache(cls) -> None:
        """Wipe the process-level memoization (e.g. between independent
        experimental runs that intentionally want fresh BNN draws)."""
        cls._NOISELESS_CACHE.clear()

    def query(self, params: Dict) -> Dict:
        """Noisy query — adds aleatoric+epistemic Gaussian noise."""
        clipped = self._clip(params)
        mean, alea, epi = self._predict(clipped)
        total_std = float(np.sqrt(float(alea) ** 2 + float(epi) ** 2))
        y_raw = float(mean) + float(self.rng.normal(0.0, total_std))
        y_clamped = float(max(0.0, y_raw))
        return {'conversion_pct': y_clamped}

    def query_noiseless(self, params: Dict) -> Dict:
        """Noiseless BNN ensemble mean. Made input-deterministic via a
        process-level (base_seed, params) -> conversion cache: the
        underlying TF/tfp BNN is stochastic per call, so we memoize the
        first draw and return it for every subsequent matching input.
        This guarantees warmup pre-scoring agrees with run-loop tells AND
        DRL vs Native episodes see identical conversions for shared
        param vectors."""
        clipped = self._clip(params)
        key = self._cache_key(self._base_seed, clipped)
        cached = self._NOISELESS_CACHE.get(key)
        if cached is not None:
            return {'conversion_pct': cached}
        mean, _, _ = self._predict(clipped)
        val = float(max(0.0, float(mean)))
        self._NOISELESS_CACHE[key] = val
        return {'conversion_pct': val}

    def get_param_space(self) -> Dict:
        return {k: {'type': 'continuous', 'low': lo, 'high': hi}
                for k, (lo, hi) in _BOUNDS.items()}

    def _clip(self, params: Dict) -> Dict:
        p = dict(params)
        for key, (lo, hi) in _BOUNDS.items():
            if key in p:
                p[key] = float(np.clip(float(p[key]), lo, hi))
        return p

    def _predict(self, params: Dict) -> Tuple[float, float, float]:
        feat = [[float(params[k]) for k in _PARAM_ORDER]]
        mean, alea, epi = self._emu.run(feat, num_samples=self.NUM_BNN_SAMPLES)
        m = mean[0]
        return float(m[0]), float(alea[0, 0]), float(epi[0, 0])


if __name__ == '__main__':
    sim = AlkOxReactionSimulator(seed=0)
    cases = [
        {'catalase': 0.5, 'peroxidase': 5.0, 'alcohol_oxidase': 5.0, 'ph': 7.0},
        {'catalase': 1.0, 'peroxidase': 10.0, 'alcohol_oxidase': 8.0, 'ph': 6.5},
        {'catalase': 0.05, 'peroxidase': 0.5, 'alcohol_oxidase': 2.0, 'ph': 8.0},
    ]
    for c in cases:
        r = sim.query_noiseless(c)
        print(f"cat={c['catalase']:.2f} per={c['peroxidase']:.1f} "
              f"AOx={c['alcohol_oxidase']:.1f} pH={c['ph']:.1f} "
              f"-> conversion={r['conversion_pct']:.3f}")
