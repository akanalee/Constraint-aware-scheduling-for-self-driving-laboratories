import os
import sys
import time
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                'DRL_scheduler_for_SDL'))

from case_study.olympus_env import configure

configure()

_WARMUP_PLAN: List[Dict] = [
    {'ligand': 'L3', 'res_time': 7.428878, 'temperature': 53.057318, 'catalyst_loading': 0.751002, '_warmup_yield':  0.500794, 'source': 'warmup_fixed'},
    {'ligand': 'L5', 'res_time': 2.488402, 'temperature': 70.591932, 'catalyst_loading': 2.202233, '_warmup_yield':  2.023259, 'source': 'warmup_fixed'},
    {'ligand': 'L3', 'res_time': 6.541705, 'temperature': 40.843019, 'catalyst_loading': 2.387630, '_warmup_yield':  2.978848, 'source': 'warmup_fixed'},
    {'ligand': 'L6', 'res_time': 5.450308, 'temperature': 79.263441, 'catalyst_loading': 1.195621, '_warmup_yield':  4.427042, 'source': 'warmup_fixed'},
    {'ligand': 'L4', 'res_time': 5.553404, 'temperature': 74.734775, 'catalyst_loading': 2.292321, '_warmup_yield':  5.912057, 'source': 'warmup_fixed'},
    {'ligand': 'L4', 'res_time': 3.256727, 'temperature': 97.071392, 'catalyst_loading': 0.766324, '_warmup_yield':  7.625892, 'source': 'warmup_fixed'},
    {'ligand': 'L1', 'res_time': 4.306843, 'temperature': 47.764780, 'catalyst_loading': 2.140489, '_warmup_yield':  8.817390, 'source': 'warmup_fixed'},
    {'ligand': 'L1', 'res_time': 8.724521, 'temperature': 48.661878, 'catalyst_loading': 0.993188, '_warmup_yield': 10.232245, 'source': 'warmup_fixed'},
    {'ligand': 'L6', 'res_time': 5.110445, 'temperature': 75.857664, 'catalyst_loading': 2.049860, '_warmup_yield': 11.667093, 'source': 'warmup_fixed'},
    {'ligand': 'L0', 'res_time': 5.193331, 'temperature': 54.231325, 'catalyst_loading': 1.325810, '_warmup_yield': 13.137818, 'source': 'warmup_fixed'},
    {'ligand': 'L0', 'res_time': 2.695901, 'temperature': 63.718761, 'catalyst_loading': 0.930723, '_warmup_yield': 14.018102, 'source': 'warmup_fixed'},
    {'ligand': 'L2', 'res_time': 7.956551, 'temperature': 48.542361, 'catalyst_loading': 2.360484, '_warmup_yield': 14.626442, 'source': 'warmup_fixed'},
]
assert len(_WARMUP_PLAN) == 12
assert sorted({e['ligand'] for e in _WARMUP_PLAN}) == ['L0','L1','L2','L3','L4','L5','L6']
assert all(_WARMUP_PLAN[i]['_warmup_yield'] < _WARMUP_PLAN[i+1]['_warmup_yield']
           for i in range(len(_WARMUP_PLAN) - 1))

from olympus.objects import (
    ParameterCategorical, ParameterContinuous, ParameterVector
)
from olympus.campaigns import ParameterSpace, Campaign

import torch
from botorch.models import MixedSingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import qUpperConfidenceBound, LogExpectedImprovement
from botorch.optim import optimize_acqf_mixed
from botorch.models.transforms import Normalize, Standardize


def _lhs_continuous(n: int, low: float, high: float, rng: np.random.RandomState) -> np.ndarray:
    edges = np.linspace(low, high, n + 1)
    vals = np.array([rng.uniform(edges[i], edges[i + 1]) for i in range(n)])
    rng.shuffle(vals)
    return vals


class SuzukiBOOptimizer:
    def __init__(self, bo_config: Dict, seed: int = 0, simulator=None,
                 warmup_plan: Optional[List[Dict]] = None):
        self.cfg = bo_config
        self.seed = seed
        self._rng = np.random.RandomState(seed)
        self._sim = simulator
        self._provided_warmup_plan = warmup_plan

        pspace = self.cfg['parameter_space']
        self.ligands = list(pspace['ligand']['options'])
        self.rt_lo, self.rt_hi = pspace['res_time']['low'], pspace['res_time']['high']
        self.T_lo, self.T_hi = pspace['temperature']['low'], pspace['temperature']['high']
        self.cat_lo, self.cat_hi = pspace['catalyst_loading']['low'], pspace['catalyst_loading']['high']
        self.n_warmup = int(self.cfg.get('n_warmup', 16))
        self.goal = self.cfg.get('goal', 'maximize')

        self.ucb_steps        = int(self.cfg.get('ucb_steps', 8))
        self.ucb_beta_start   = float(self.cfg.get('ucb_beta_start', 2.0))
        self.ucb_beta_end     = float(self.cfg.get('ucb_beta_end', 1.0))
        self.acqf_restarts    = int(self.cfg.get('acqf_num_restarts', 20))
        self.acqf_raw_samples = int(self.cfg.get('acqf_raw_samples', 256))
        self.gp_noise_std     = self.cfg.get('gp_noise_std', 2.25)

        self._torch_seed = int(seed)
        torch.manual_seed(self._torch_seed)
        self._bo_step = 0

        self._X_obs: List[List[float]] = []
        self._Y_obs: List[float]       = []

        self.param_space = ParameterSpace()
        self.param_space.add(ParameterCategorical(name='ligand', options=self.ligands))
        self.param_space.add(ParameterContinuous(name='res_time',
                                                 low=float(self.rt_lo), high=float(self.rt_hi)))
        self.param_space.add(ParameterContinuous(name='temperature',
                                                 low=float(self.T_lo), high=float(self.T_hi)))
        self.param_space.add(ParameterContinuous(name='catalyst_loading',
                                                 low=float(self.cat_lo), high=float(self.cat_hi)))

        self.campaign = Campaign()
        self.campaign.set_param_space(self.param_space)

        planner_kind = self.cfg.get('planner_kind', 'Botorch_custom_qUCB_LogEI')

        if self._provided_warmup_plan is not None:
            self._warmup_plan: List[Dict] = [
                dict(p) for p in self._provided_warmup_plan
            ]
            print(f"[SuzukiBO] using injected warmup plan: n={len(self._warmup_plan)}")
        else:
            self._warmup_plan = self._build_warmup_plan()
        self._warmup_idx: int = 0

        self.updates: List[Dict] = []
        self.last_ask_wall_time_sec: float = 0.0
        self._ask_count = 0
        self._tell_count = 0

        print(f"[SuzukiBO] planner={planner_kind}, ligands={self.ligands}, "
              f"n_warmup={self.n_warmup}, seed={seed}")

    def _build_warmup_plan(self) -> List[Dict]:
        if bool(self.cfg.get('use_fixed_warmup', True)):
            plan: List[Dict] = [dict(p) for p in _WARMUP_PLAN]
            ys = [p['_warmup_yield'] for p in plan]
            lh = {l: sum(1 for p in plan if p['ligand']==l) for l in self.ligands}
            print(f"[SuzukiBO] fixed warmstart: n={len(plan)}  "
                  f"yield ramp=[{ys[0]:.2f} .. {ys[-1]:.2f}]  "
                  f"ligand histogram={lh}")
            return plan

        return self._build_lhs_warmup_plan()

    def _build_lhs_warmup_plan(self) -> List[Dict]:
        return self._lhs_samples(self.n_warmup, source_tag='warmup_lhs')

    def _lhs_samples(self, n: int, source_tag: str) -> List[Dict]:
        n_lig = len(self.ligands)
        rts = _lhs_continuous(n, self.rt_lo, self.rt_hi, self._rng)
        Ts = _lhs_continuous(n, self.T_lo, self.T_hi, self._rng)
        cats = _lhs_continuous(n, self.cat_lo, self.cat_hi, self._rng)
        lig_slots = [self.ligands[i % n_lig] for i in range(n)]
        self._rng.shuffle(lig_slots)
        plan = []
        for i in range(n):
            plan.append({
                'ligand':           lig_slots[i],
                'res_time':         float(rts[i]),
                'temperature':      float(Ts[i]),
                'catalyst_loading': float(cats[i]),
                'source':           source_tag,
            })
        return plan

    def ask(self, n: int = 1) -> List[Dict]:
        t0 = time.time()
        out: List[Dict] = []
        for _ in range(n):
            if self._warmup_idx < len(self._warmup_plan):
                p = dict(self._warmup_plan[self._warmup_idx])
                self._warmup_idx += 1
            else:
                p = self._torch_ask()
                p['source'] = 'botorch'
            out.append(p)
            self._ask_count += 1
        self.last_ask_wall_time_sec = time.time() - t0
        return out

    def _torch_ask(self) -> Dict:
        n_lig = len(self.ligands)
        n_dim = n_lig + 3

        torch.manual_seed(self._torch_seed * 100003 + (self._bo_step + 1))

        train_X = torch.tensor(self._X_obs, dtype=torch.double)
        train_Y = torch.tensor(self._Y_obs, dtype=torch.double).unsqueeze(-1)
        if self.goal != 'maximize':
            train_Y = -train_Y

        bounds = torch.tensor(
            [[0.0] * n_lig + [self.rt_lo, self.T_lo, self.cat_lo],
             [1.0] * n_lig + [self.rt_hi, self.T_hi, self.cat_hi]],
            dtype=torch.double,
        )
        cat_dims = list(range(n_lig))

        model = MixedSingleTaskGP(
            train_X, train_Y, cat_dims=cat_dims,
            input_transform=Normalize(d=n_dim, bounds=bounds),
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

        fixed_features_list = [
            {int(d): (1.0 if d == k else 0.0) for d in range(n_lig)}
            for k in range(n_lig)
        ]
        results, _ = optimize_acqf_mixed(
            acq_function=acqf,
            bounds=bounds,
            q=1,
            num_restarts=self.acqf_restarts,
            raw_samples=self.acqf_raw_samples,
            fixed_features_list=fixed_features_list,
        )
        r = results.detach().cpu().numpy().reshape(-1)
        lig_idx = int(np.argmax(r[:n_lig]))
        out = {
            'ligand':           self.ligands[lig_idx],
            'res_time':         float(r[n_lig]),
            'temperature':      float(r[n_lig + 1]),
            'catalyst_loading': float(r[n_lig + 2]),
        }
        print(f"[SuzukiBO] BO ask #{self._bo_step}  acqf={acqf_tag}  "
              f"L={out['ligand']}  rt={out['res_time']:.2f}  "
              f"T={out['temperature']:.2f}  cat={out['catalyst_loading']:.3f}")
        return out

    def tell(self, params: Dict, yield_pct: float):
        pv = self._dict_to_paramvector(params)
        self.campaign.add_observation(pv, float(yield_pct))
        self._tell_count += 1

        n_lig = len(self.ligands)
        x = [0.0] * n_lig
        x[self.ligands.index(str(params['ligand']))] = 1.0
        x += [float(params['res_time']),
              float(params['temperature']),
              float(params['catalyst_loading'])]
        self._X_obs.append(x)
        self._Y_obs.append(float(yield_pct))

        values = np.array(self.campaign.observations.get_values(), dtype=float).reshape(-1)
        best = float(values.max() if self.goal == 'maximize' else values.min())

        self.updates.append({
            'step':          len(self.updates) + 1,
            'job_id':        params.get('job_id'),
            'source':        params.get('source', 'unknown'),
            'ligand':        params['ligand'],
            'res_time':      float(params['res_time']),
            'temperature':   float(params['temperature']),
            'catalyst_loading': float(params['catalyst_loading']),
            'yield_pct':     float(yield_pct),
            'best_so_far':   best,
            'n_observations': int(values.size),
            'last_ask_wall_time_sec': float(self.last_ask_wall_time_sec),
            'wall_timestamp': time.time(),
        })

    def _dict_to_paramvector(self, params: Dict) -> ParameterVector:
        clean = {
            'ligand':           str(params['ligand']),
            'res_time':         float(params['res_time']),
            'temperature':      float(params['temperature']),
            'catalyst_loading': float(params['catalyst_loading']),
        }
        return ParameterVector().from_dict(clean, self.param_space)

    def summary(self) -> Dict:
        if not self.updates:
            return {'n_updates': 0}
        ys = np.array([u['yield_pct'] for u in self.updates])
        bsf = np.array([u['best_so_far'] for u in self.updates])
        ask_times = np.array([u['last_ask_wall_time_sec'] for u in self.updates])
        return {
            'n_updates':        int(len(self.updates)),
            'n_ask':            int(self._ask_count),
            'n_tell':           int(self._tell_count),
            'best_yield':       float(bsf.max() if self.goal == 'maximize' else ys.min()),
            'mean_yield':       float(ys.mean()),
            'final_best_so_far': float(bsf[-1]),
            'total_ask_time_s': float(ask_times.sum()),
            'mean_ask_time_s':  float(ask_times.mean()),
        }
