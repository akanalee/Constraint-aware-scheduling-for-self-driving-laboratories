import os
import sys
import time
from typing import Dict, List, Optional

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                'DRL_scheduler_for_SDL'))

from case_study.olympus_env import configure

configure()

from olympus.objects import ParameterContinuous, ParameterVector
from olympus.campaigns import ParameterSpace, Campaign

from botorch.models import SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import qUpperConfidenceBound, LogExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood


_PARAM_NAMES = ('catalase', 'peroxidase', 'alcohol_oxidase', 'ph')


_WARMUP_PLAN: List[Dict] = [
    {'catalase': 0.309086, 'peroxidase': 1.294896, 'alcohol_oxidase': 7.441774, 'ph': 7.988166, '_warmup_conv':  1.109304, 'source': 'warmup_fixed'},
    {'catalase': 0.315738, 'peroxidase': 1.282272, 'alcohol_oxidase': 5.801763, 'ph': 7.916562, '_warmup_conv':  2.253969, 'source': 'warmup_fixed'},
    {'catalase': 0.222121, 'peroxidase': 6.282852, 'alcohol_oxidase': 2.210838, 'ph': 7.616428, '_warmup_conv':  4.005845, 'source': 'warmup_fixed'},
    {'catalase': 0.528231, 'peroxidase': 3.439331, 'alcohol_oxidase': 2.173025, 'ph': 7.686391, '_warmup_conv':  5.502713, 'source': 'warmup_fixed'},
    {'catalase': 0.781643, 'peroxidase': 1.539171, 'alcohol_oxidase': 2.168880, 'ph': 7.693606, '_warmup_conv':  7.500474, 'source': 'warmup_fixed'},
    {'catalase': 0.933207, 'peroxidase': 3.702107, 'alcohol_oxidase': 3.645370, 'ph': 6.802004, '_warmup_conv': 10.011752, 'source': 'warmup_fixed'},
    {'catalase': 0.114516, 'peroxidase': 5.450945, 'alcohol_oxidase': 6.891239, 'ph': 6.725906, '_warmup_conv': 12.502816, 'source': 'warmup_fixed'},
    {'catalase': 0.829130, 'peroxidase': 2.752314, 'alcohol_oxidase': 6.289579, 'ph': 6.827091, '_warmup_conv': 15.009007, 'source': 'warmup_fixed'},
    {'catalase': 0.248530, 'peroxidase': 9.887650, 'alcohol_oxidase': 5.679520, 'ph': 7.167598, '_warmup_conv': 17.509522, 'source': 'warmup_fixed'},
    {'catalase': 0.569749, 'peroxidase': 6.669305, 'alcohol_oxidase': 6.580115, 'ph': 7.342116, '_warmup_conv': 20.031734, 'source': 'warmup_fixed'},
    {'catalase': 0.836561, 'peroxidase': 4.533966, 'alcohol_oxidase': 6.338459, 'ph': 6.106140, '_warmup_conv': 22.524108, 'source': 'warmup_fixed'},
    {'catalase': 0.567368, 'peroxidase': 4.167343, 'alcohol_oxidase': 7.698980, 'ph': 6.358375, '_warmup_conv': 25.058498, 'source': 'warmup_fixed'},
]
assert len(_WARMUP_PLAN) == 12
assert all(_WARMUP_PLAN[i]['_warmup_conv']
           < _WARMUP_PLAN[i + 1]['_warmup_conv']
           for i in range(len(_WARMUP_PLAN) - 1))
assert all(0.05 <= e['catalase']        <= 1.0
       and 0.5  <= e['peroxidase']      <= 10.0
       and 2.0  <= e['alcohol_oxidase'] <= 8.0
       and 6.0  <= e['ph']              <= 8.0
           for e in _WARMUP_PLAN)


def _lhs_continuous(n: int, low: float, high: float,
                    rng: np.random.RandomState) -> np.ndarray:
    edges = np.linspace(low, high, n + 1)
    vals = np.array([rng.uniform(edges[i], edges[i + 1]) for i in range(n)])
    rng.shuffle(vals)
    return vals


class AlkOxBOOptimizer:
    def __init__(self, bo_config: Dict, seed: int = 0, sim_factory=None,
                 warmup_plan: Optional[List[Dict]] = None):
        self.cfg = bo_config
        self.seed = seed
        self._rng = np.random.RandomState(seed)
        self._sim_factory = sim_factory
        self._provided_warmup_plan = warmup_plan

        pspace = self.cfg['parameter_space']
        self.bounds: Dict[str, tuple] = {
            n: (float(pspace[n]['low']), float(pspace[n]['high']))
            for n in _PARAM_NAMES
        }
        self.n_warmup = int(self.cfg.get('n_warmup', 12))
        self.goal = self.cfg.get('goal', 'maximize')

        self.warmup_conv_ceiling = float(self.cfg.get('warmup_conv_ceiling', 10.0))
        self.warmup_pool_size    = int(self.cfg.get('warmup_pool_size', 400))
        self.warmup_max_tries    = int(self.cfg.get('warmup_max_tries', 20))

        self.ucb_steps      = int(self.cfg.get('ucb_steps', 8))
        self.ucb_beta_start = float(self.cfg.get('ucb_beta_start', 4.0))
        self.ucb_beta_end   = float(self.cfg.get('ucb_beta_end', 1.0))
        self.acqf_restarts  = int(self.cfg.get('acqf_num_restarts', 20))
        self.acqf_raw_samples = int(self.cfg.get('acqf_raw_samples', 512))
        self.gp_noise_std   = self.cfg.get('gp_noise_std', 3.0)

        self.param_space = ParameterSpace()
        for n in _PARAM_NAMES:
            lo, hi = self.bounds[n]
            self.param_space.add(ParameterContinuous(name=n, low=lo, high=hi))

        self.campaign = Campaign()
        self.campaign.set_param_space(self.param_space)

        self._torch_seed = seed
        torch.manual_seed(seed)

        self._bounds_t = torch.tensor(
            [[self.bounds[n][0] for n in _PARAM_NAMES],
             [self.bounds[n][1] for n in _PARAM_NAMES]],
            dtype=torch.double,
        )
        self._X_obs: List[List[float]] = []
        self._Y_obs: List[float] = []
        self._bo_step = 0

        planner_kind = self.cfg.get('planner_kind', 'Botorch')

        if self._provided_warmup_plan is not None:
            self._warmup_plan: List[Dict] = [
                dict(p) for p in self._provided_warmup_plan
            ]
            convs = [p.get('_warmup_conv', float('nan'))
                     for p in self._warmup_plan]
            print(f"[AlkOxBO] using injected warmup plan: n={len(self._warmup_plan)} "
                  f"conv_range=[{min(convs):.2f}, {max(convs):.2f}]")
        else:
            self._warmup_plan = self._build_lhs_warmup_plan()
        self._warmup_idx: int = 0

        self._warmup_replacement_pool: List[Dict] = self._lhs_samples(
            self.n_warmup, source_tag='warmup_replacement_lhs'
        )
        self._warmup_replacement_idx: int = 0

        self.updates: List[Dict] = []
        self.last_ask_wall_time_sec: float = 0.0
        self._ask_count = 0
        self._tell_count = 0

        print(f"[AlkOxBO] planner={planner_kind}, "
              f"params={list(_PARAM_NAMES)}, "
              f"n_warmup={self.n_warmup}, seed={seed}")

    def _build_lhs_warmup_plan(self) -> List[Dict]:
        if bool(self.cfg.get('use_fixed_warmup', True)):
            plan: List[Dict] = [dict(p) for p in _WARMUP_PLAN]
            convs = [p['_warmup_conv'] for p in plan]
            print(f"[AlkOxBO] fixed warmstart: n={len(plan)}  "
                  f"conv ramp=[{convs[0]:.2f} .. {convs[-1]:.2f}]")
            return plan

        WARMUP_CONV_BANDS: List[tuple] = [
            (0.0,  1.0),
            (2.5,  3.0),
            (5.0,  6.0),
            (7.1,  8.0),
            (10.1,  11.0),
            (12.0,  13.0),
            (15.1,  16.0),
            (17.1,  18.0),
            (20.0, 21.0),
            (22.1, 23.0),
            (25.2, 26.0),
            (27.1, 28.0),
        ]

        if len(WARMUP_CONV_BANDS) != self.n_warmup:
            print(f"[AlkOxBO] WARN: WARMUP_CONV_BANDS has "
                  f"{len(WARMUP_CONV_BANDS)} segments but n_warmup="
                  f"{self.n_warmup}; using min(len, n_warmup) segments.")

        if self._sim_factory is None:
            return self._lhs_samples(self.n_warmup, source_tag='warmup_lhs')

        sim = self._sim_factory(self.seed)
        scored: List[tuple] = []

        for attempt in range(self.warmup_max_tries):
            pool = self._lhs_samples(self.warmup_pool_size,
                                     source_tag=f'warmup_pool_try{attempt}')
            scored = []
            for p in pool:
                kw = {k: float(p[k]) for k in _PARAM_NAMES}
                conv = float(sim.query_noiseless(kw)['conversion_pct'])
                scored.append((conv, p))
            scored.sort(key=lambda t: t[0])

            picked: List[tuple] = []
            missing: List[int] = []
            for bi, (lo, hi) in enumerate(WARMUP_CONV_BANDS):
                hit = next(((c, p) for c, p in scored if lo <= c <= hi), None)
                if hit is None:
                    missing.append(bi)
                else:
                    picked.append((bi, hit[0], hit[1]))

            if not missing and len(picked) >= self.n_warmup:
                picked = picked[:self.n_warmup]
                chosen = [p for _, _, p in picked]
                chosen_convs = [c for _, c, _ in picked]
                bands_str = ", ".join(
                    f"[{WARMUP_CONV_BANDS[bi][0]},{WARMUP_CONV_BANDS[bi][1]}]:{c:.2f}"
                    for bi, c, _ in picked
                )
                print(f"[AlkOxBO] warmup selected: n={len(chosen)} "
                      f"conv_range=[{min(chosen_convs):.2f}, "
                      f"{max(chosen_convs):.2f}] "
                      f"per-band: {bands_str} "
                      f"(try {attempt + 1}/{self.warmup_max_tries})")
                for p, c in zip(chosen, chosen_convs):
                    p['source'] = 'warmup_low_yield_lhs'
                    p['_warmup_conv'] = c
                return chosen

        used_ids = {id(p) for _, _, p in picked}
        leftover = [(c, p) for c, p in scored if id(p) not in used_ids]
        need = self.n_warmup - len(picked)
        fill_pairs = leftover[:need]
        chosen = [p for _, _, p in picked] + [p for _, p in fill_pairs]
        chosen_convs = [c for _, c, _ in picked] + [c for c, _ in fill_pairs]
        miss_str = ", ".join(
            f"[{WARMUP_CONV_BANDS[bi][0]},{WARMUP_CONV_BANDS[bi][1]}]"
            for bi in missing
        )
        print(f"[AlkOxBO] WARN: empty bands {miss_str} after "
              f"{self.warmup_max_tries} tries; filled {need} slot(s) with "
              f"lowest leftover points, "
              f"conv_range=[{min(chosen_convs):.2f}, {max(chosen_convs):.2f}]")
        for p, c in zip(chosen, chosen_convs):
            p.setdefault('source', 'warmup_low_yield_fallback')
            p['_warmup_conv'] = c
        return chosen

    def _lhs_samples(self, n: int, source_tag: str) -> List[Dict]:
        cols = {name: _lhs_continuous(n, lo, hi, self._rng)
                for name, (lo, hi) in self.bounds.items()}
        plan: List[Dict] = []
        for i in range(n):
            p = {name: float(cols[name][i]) for name in _PARAM_NAMES}
            p['source'] = source_tag
            plan.append(p)
        return plan

    def ask(self, n: int = 1) -> List[Dict]:
        t0 = time.time()
        out: List[Dict] = []
        for _ in range(n):
            if self._warmup_idx < len(self._warmup_plan):
                p = dict(self._warmup_plan[self._warmup_idx])
                self._warmup_idx += 1
            else:
                p = self._botorch_ask()
                p['source'] = 'botorch'
            out.append(p)
            self._ask_count += 1
        self.last_ask_wall_time_sec = time.time() - t0
        return out

    def _botorch_ask(self) -> Dict:
        if not self._X_obs:
            return {n: float(self._rng.uniform(*self.bounds[n]))
                    for n in _PARAM_NAMES}

        torch.manual_seed(self._torch_seed * 100003 + (self._bo_step + 1))
        train_X = torch.tensor(self._X_obs, dtype=torch.double)
        train_Y = torch.tensor(self._Y_obs, dtype=torch.double).unsqueeze(-1)
        if self.goal != 'maximize':
            train_Y = -train_Y

        model = SingleTaskGP(
            train_X, train_Y,
            input_transform=Normalize(d=train_X.shape[-1], bounds=self._bounds_t),
            outcome_transform=Standardize(m=1),
        )
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)

        if self.gp_noise_std is not None:
            Y_std = float(train_Y.std()) if train_Y.numel() > 1 else 1.0
            target_var = (float(self.gp_noise_std) / (Y_std + 1e-9)) ** 2
            model.likelihood.noise = torch.tensor([target_var], dtype=torch.double)

        self._bo_step += 1
        if self._bo_step <= self.ucb_steps and self.ucb_steps > 0:
            if self.ucb_steps == 1:
                beta = self.ucb_beta_end
            else:
                frac = (self._bo_step - 1) / (self.ucb_steps - 1)
                beta = self.ucb_beta_start + frac * (
                    self.ucb_beta_end - self.ucb_beta_start
                )
            acqf = qUpperConfidenceBound(model, beta=float(beta))
            acqf_tag = f"qUCB(beta={beta:.2f})"
        else:
            best_f = float(train_Y.max().item())
            acqf = LogExpectedImprovement(model, best_f=best_f, maximize=True)
            acqf_tag = "LogEI"

        cand, _ = optimize_acqf(
            acq_function=acqf,
            bounds=self._bounds_t,
            q=1,
            num_restarts=self.acqf_restarts,
            raw_samples=self.acqf_raw_samples,
        )
        cand_np = cand.detach().cpu().numpy().reshape(-1)
        out = {n: float(cand_np[i]) for i, n in enumerate(_PARAM_NAMES)}
        print(f"[AlkOxBO] BO ask #{self._bo_step}  acqf={acqf_tag}  "
              f"cand=cat={out['catalase']:.3f} per={out['peroxidase']:.2f} "
              f"AOx={out['alcohol_oxidase']:.2f} pH={out['ph']:.2f}")
        return out

    def tell(self, params: Dict, conversion_pct: float):
        pv = self._dict_to_paramvector(params)
        self.campaign.add_observation(pv, float(conversion_pct))
        self._tell_count += 1

        self._X_obs.append([float(params[n]) for n in _PARAM_NAMES])
        self._Y_obs.append(float(conversion_pct))

        values = np.array(self.campaign.observations.get_values(),
                          dtype=float).reshape(-1)
        best = float(values.max() if self.goal == 'maximize' else values.min())

        rec = {
            'step':           len(self.updates) + 1,
            'job_id':         params.get('job_id'),
            'source':         params.get('source', 'unknown'),
            'conversion_pct': float(conversion_pct),
            'best_so_far':    best,
            'n_observations': int(values.size),
            'last_ask_wall_time_sec': float(self.last_ask_wall_time_sec),
            'wall_timestamp': time.time(),
        }
        for name in _PARAM_NAMES:
            rec[name] = float(params[name])
        self.updates.append(rec)

    def _dict_to_paramvector(self, params: Dict) -> ParameterVector:
        clean = {n: float(params[n]) for n in _PARAM_NAMES}
        return ParameterVector().from_dict(clean, self.param_space)

    def summary(self) -> Dict:
        if not self.updates:
            return {'n_updates': 0}
        ys = np.array([u['conversion_pct'] for u in self.updates])
        bsf = np.array([u['best_so_far'] for u in self.updates])
        ask_times = np.array([u['last_ask_wall_time_sec'] for u in self.updates])
        return {
            'n_updates':         int(len(self.updates)),
            'n_ask':             int(self._ask_count),
            'n_tell':            int(self._tell_count),
            'best_yield':        float(bsf.max() if self.goal == 'maximize' else ys.min()),
            'mean_yield':        float(ys.mean()),
            'final_best_so_far': float(bsf[-1]),
            'total_ask_time_s':  float(ask_times.sum()),
            'mean_ask_time_s':   float(ask_times.mean()),
        }
