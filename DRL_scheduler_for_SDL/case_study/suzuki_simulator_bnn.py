#!/usr/bin/env python
"""
suzuki_simulator_bnn.py  -  Olympus BayesNeuralNet oracle for Suzuki-Miyaura.

Drop-in replacement for SuzukiReactionSimulator backed by an ENSEMBLE
of Olympus's four suzuki_{i,ii,iii,iv}_BayesNeuralNet emulators. Each
Olympus variant is a BNN trained on one of the four replicate datasets
(data_1..4), same chemistry, independent noise; the ensemble mean is
our proxy for a BNN trained on the concatenated 356-row corpus (which
Olympus does not ship). Validation R² vs RBF+kNN LOO R² = 0.977 vs 0.362
— the RBF can't represent the 82 chemical-discontinuity near-neighbour
pairs, BNN can.

External API matches the RBF simulator exactly:
    input  : {ligand (L0..L6), res_time (min), temperature, catalyst_loading}
    output : {yield_pct, turnover, reaction_time_min}

Internal conversion to Olympus BNN's native param space:
    * res_time: minutes -> seconds (BNN trained on 60-600 s)
    * ligand : accepted as string; L0..L6 are a subset of the BNN's
               L0..L7 one-hot (we never emit L7, which is fine).
"""

import numpy as np
from typing import Dict, Optional, Tuple

from case_study._olympus_bnn_loader import get_emulator


LIGAND_NAMES = [f"L{i}" for i in range(7)]
LIGAND_TO_IDX = {name: i for i, name in enumerate(LIGAND_NAMES)}

PARAM_SPACE = {
    "ligand":           {"type": "categorical", "options": LIGAND_NAMES, "n": 7},
    "res_time":         {"type": "continuous", "low": 1.0,   "high": 10.0,  "unit": "min"},
    "temperature":      {"type": "continuous", "low": 30.0,  "high": 110.0, "unit": "C"},
    "catalyst_loading": {"type": "continuous", "low": 0.489, "high": 2.516, "unit": "mol%"},
}

TRUE_OPTIMUM = {
    "ligand": "L0", "res_time": 5.00, "temperature": 73.5,
    "catalyst_loading": 2.501, "yield_pct": 99.8, "turnover": 39.9,
}


class SuzukiReactionSimulator:
    """BNN-backed Suzuki yield/turnover oracle."""

    NUM_BNN_SAMPLES = 20
    _VARIANTS = ('suzuki_i_BayesNeuralNet',
                 'suzuki_ii_BayesNeuralNet',
                 'suzuki_iii_BayesNeuralNet',
                 'suzuki_iv_BayesNeuralNet')

    def __init__(self, data_path: Optional[str] = None, seed: Optional[int] = None):
        _ = data_path  # accepted for API compat, unused (BNN is pre-trained)
        self.rng = np.random.RandomState(seed)
        self._emus = [get_emulator(name) for name in self._VARIANTS]

    def query(self, params: Dict) -> Dict:
        lig, rt_min, T, cat = self._clip(params)
        mean, alea, epi = self._predict(lig, rt_min, T, cat)
        # mean/alea/epi are length-2 vectors: [yield, turnover].
        std = np.sqrt(alea ** 2 + epi ** 2)
        y = float(mean[0] + self.rng.normal(0.0, float(std[0])))
        ton = float(mean[1] + self.rng.normal(0.0, float(std[1])))
        return {
            "yield_pct":         float(np.clip(y, 0.0, 100.0)),
            "turnover":          float(np.clip(ton, 0.0, None)),
            "reaction_time_min": rt_min,
        }

    def query_noiseless(self, params: Dict) -> Dict:
        lig, rt_min, T, cat = self._clip(params)
        mean, _, _ = self._predict(lig, rt_min, T, cat)
        return {
            "yield_pct":         float(np.clip(float(mean[0]), 0.0, 100.0)),
            "turnover":          float(np.clip(float(mean[1]), 0.0, None)),
            "reaction_time_min": rt_min,
        }

    def get_param_space(self) -> Dict:
        return PARAM_SPACE

    def get_true_optimum(self) -> Dict:
        return {
            "global_optimum":   {"ligand": "L0", "res_time": 5.00, "temperature": 73.5,
                                 "catalyst_loading": 2.501, "yield_pct": 99.8, "turnover": 39.9},
        }

    # ------------------------------------------------------------------
    def _parse_ligand(self, raw) -> str:
        if isinstance(raw, str):
            if raw in LIGAND_TO_IDX:
                return raw
            # tolerate "0", "1", ... or int-like
            try:
                idx = int(raw)
                return LIGAND_NAMES[idx]
            except (ValueError, IndexError):
                pass
        if isinstance(raw, (int, np.integer)):
            return LIGAND_NAMES[int(raw) % len(LIGAND_NAMES)]
        raise ValueError(f"Unrecognized ligand: {raw!r}")

    def _clip(self, params: Dict) -> Tuple[str, float, float, float]:
        # Tight clip = intersection of all 4 Olympus variants' bounds,
        # so no variant ever triggers the "not within bounds" warning.
        # Variants: i=(0.498,2.515) ii=(0.492,2.516) iii=(0.499,2.513)
        #           iv=(0.489,2.510). Intersection = (0.499, 2.510).
        lig = self._parse_ligand(params["ligand"])
        rt = float(np.clip(params["res_time"], 1.0, 10.0))           # min
        T = float(np.clip(params["temperature"], 30.0, 110.0))       # C
        cat = float(np.clip(params["catalyst_loading"], 0.499, 2.510))  # mol%
        return lig, rt, T, cat

    def _predict(self, ligand: str, rt_min: float, T: float, cat: float
                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Ensemble mean across the 4 BNN replicates. Total variance =
        # within-model (alea+epi) averaged + between-model (variance of
        # the 4 means), giving a principled total std.
        feat = [[ligand, rt_min * 60.0, T, cat]]
        means, aleas, epis = [], [], []
        for emu in self._emus:
            m, a, e = emu.run(feat, num_samples=self.NUM_BNN_SAMPLES)
            means.append(m[0])
            aleas.append(a[0])
            epis.append(e[0])
        means = np.stack(means, axis=0)           # (4, 2)
        aleas = np.stack(aleas, axis=0)
        epis = np.stack(epis, axis=0)
        mean = means.mean(axis=0)
        within_var = (aleas ** 2 + epis ** 2).mean(axis=0)
        between_var = means.var(axis=0)
        alea = np.sqrt(within_var)                 # treat as aleatoric proxy
        epi = np.sqrt(between_var)                 # epistemic via disagreement
        return mean, alea, epi


if __name__ == "__main__":
    sim = SuzukiReactionSimulator(seed=0)
    for c in [
        {"ligand": "L0", "res_time": 5.00, "temperature": 73.5, "catalyst_loading": 2.501},
        {"ligand": "L3", "res_time": 3.15, "temperature": 110.0, "catalyst_loading": 2.508},
        {"ligand": "L1", "res_time": 4.61, "temperature": 69.3, "catalyst_loading": 2.507},
    ]:
        print(c, "->", sim.query_noiseless(c))
