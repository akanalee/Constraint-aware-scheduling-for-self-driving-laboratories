from typing import Dict, Optional, Tuple

import numpy as np

from case_study._olympus_bnn_loader import get_emulator


_PARAM_ORDER = ['catalase', 'peroxidase', 'alcohol_oxidase', 'ph']

_BOUNDS = {
    'catalase':        (0.05, 1.0),
    'peroxidase':      (0.5,  10.0),
    'alcohol_oxidase': (2.0,  8.0),
    'ph':              (6.0,  8.0),
}


class AlkOxReactionSimulator:
    NUM_BNN_SAMPLES = 20

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
        cls._NOISELESS_CACHE.clear()

    def query_noiseless(self, params: Dict) -> Dict:
        clipped = self._clip(params)
        key = self._cache_key(self._base_seed, clipped)
        cached = self._NOISELESS_CACHE.get(key)
        if cached is not None:
            return {'conversion_pct': cached}
        mean, _, _ = self._predict(clipped)
        val = float(max(0.0, float(mean)))
        self._NOISELESS_CACHE[key] = val
        return {'conversion_pct': val}

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
