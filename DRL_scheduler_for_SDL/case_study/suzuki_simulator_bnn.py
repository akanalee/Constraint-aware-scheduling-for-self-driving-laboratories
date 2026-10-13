import numpy as np
from typing import Dict, Optional, Tuple

from case_study._olympus_bnn_loader import get_emulator


LIGAND_NAMES = [f"L{i}" for i in range(7)]
LIGAND_TO_IDX = {name: i for i, name in enumerate(LIGAND_NAMES)}


class SuzukiReactionSimulator:
    NUM_BNN_SAMPLES = 20
    _VARIANTS = ('suzuki_i_BayesNeuralNet',
                 'suzuki_ii_BayesNeuralNet',
                 'suzuki_iii_BayesNeuralNet',
                 'suzuki_iv_BayesNeuralNet')

    def __init__(self, data_path: Optional[str] = None, seed: Optional[int] = None):
        _ = data_path
        self.rng = np.random.RandomState(seed)
        self._emus = [get_emulator(name) for name in self._VARIANTS]

    def query_noiseless(self, params: Dict) -> Dict:
        lig, rt_min, T, cat = self._clip(params)
        mean, _, _ = self._predict(lig, rt_min, T, cat)
        return {
            "yield_pct":         float(np.clip(float(mean[0]), 0.0, 100.0)),
            "turnover":          float(np.clip(float(mean[1]), 0.0, None)),
            "reaction_time_min": rt_min,
        }

    def _parse_ligand(self, raw) -> str:
        if isinstance(raw, str):
            if raw in LIGAND_TO_IDX:
                return raw
            try:
                idx = int(raw)
                return LIGAND_NAMES[idx]
            except (ValueError, IndexError):
                pass
        if isinstance(raw, (int, np.integer)):
            return LIGAND_NAMES[int(raw) % len(LIGAND_NAMES)]
        raise ValueError(f"Unrecognized ligand: {raw!r}")

    def _clip(self, params: Dict) -> Tuple[str, float, float, float]:
        lig = self._parse_ligand(params["ligand"])
        rt = float(np.clip(params["res_time"], 1.0, 10.0))
        T = float(np.clip(params["temperature"], 30.0, 110.0))
        cat = float(np.clip(params["catalyst_loading"], 0.499, 2.510))
        return lig, rt, T, cat

    def _predict(self, ligand: str, rt_min: float, T: float, cat: float
                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        feat = [[ligand, rt_min * 60.0, T, cat]]
        means, aleas, epis = [], [], []
        for emu in self._emus:
            m, a, e = emu.run(feat, num_samples=self.NUM_BNN_SAMPLES)
            means.append(m[0])
            aleas.append(a[0])
            epis.append(e[0])
        means = np.stack(means, axis=0)
        aleas = np.stack(aleas, axis=0)
        epis = np.stack(epis, axis=0)
        mean = means.mean(axis=0)
        within_var = (aleas ** 2 + epis ** 2).mean(axis=0)
        between_var = means.var(axis=0)
        alea = np.sqrt(within_var)
        epi = np.sqrt(between_var)
        return mean, alea, epi
