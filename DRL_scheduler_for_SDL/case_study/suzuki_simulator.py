"""=================================================================== -------- v1 (TPS, smoothing=1.0)          : in-sample MAE=8.38%, LOO MAE=18.02% v2 (TPS sm=0 + kNN-3)            : in-sample MAE=0.6"""

import os
import glob
import csv as _csv
import numpy as np
from scipy.interpolate import RBFInterpolator
from scipy.spatial.distance import cdist
from typing import Dict, List, Optional, Tuple
from collections import defaultdict


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

LIGAND_NAMES  = [f"L{i}" for i in range(7)]
LIGAND_TO_IDX = {name: i for i, name in enumerate(LIGAND_NAMES)}

PARAM_SPACE = {
    "ligand":           {"type": "categorical", "options": LIGAND_NAMES, "n": 7},
    "res_time":         {"type": "continuous", "low": 1.0,   "high": 10.0,  "unit": "min"},
    "temperature":      {"type": "continuous", "low": 30.0,  "high": 110.0, "unit": "°C"},
    "catalyst_loading": {"type": "continuous", "low": 0.489, "high": 2.516, "unit": "mol%"},
}

TRUE_OPTIMUM = {
    "ligand": "L0", "res_time": 5.00, "temperature": 73.5,
    "catalyst_loading": 2.501, "yield_pct": 99.8, "turnover": 39.9,
}

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SEARCH_DIRS = [
    _THIS_DIR,
    os.path.join(_THIS_DIR, "data"),
    os.path.join(_THIS_DIR, "..", "data"),
    os.path.join(_THIS_DIR, "..", "..", "data"),
]

_RT_ROUND   = 3
_TEMP_ROUND = 1
_CAT_ROUND  = 3

_RT_NORM  = 9.0    # range: 1–10 min
_T_NORM   = 80.0   # range: 30–110 °C
_CAT_NORM = 2.027  # range: 0.489–2.516 mol%

_FALLBACK_THRESHOLD = 0.10
_KNN_K              = 5

_RBF_KERNEL   = "linear"
_RBF_SMOOTHING = 0.0   # exact interpolation on aggregated unique points


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

def _find_csvs(dirs):
    for d in dirs:
        hits = sorted(glob.glob(os.path.join(os.path.normpath(d), "data_*.csv")))
        if hits:
            return hits
    return []


def _parse_csvs(files):
    rows_p, rows_y, rows_t = [], [], []
    for fpath in files:
        with open(fpath, "r", newline="", encoding="utf-8") as fh:
            for raw in _csv.reader(fh):
                row = [c.strip() for c in raw if c.strip()]
                if not row or row[0].startswith("#") or len(row) < 6:
                    continue
                try:
                    raw_lig = row[0]
                    lig_idx = int(raw_lig[1:]) if raw_lig.upper().startswith("L") \
                              else int(float(raw_lig))
                    if not (0 <= lig_idx <= 6):
                        continue
                    res_min = float(row[1]) / 60.0   # seconds  minutes
                    temp    = float(row[2])
                    cat     = float(row[3])
                    yld     = float(row[4])           # yield % — col 4
                    ton     = float(row[5])           # turnover — col 5
                except (ValueError, IndexError):
                    continue
                rows_p.append([lig_idx, res_min, temp, cat])
                rows_y.append(yld)
                rows_t.append(ton)
    if not rows_p:
        raise RuntimeError(f"No valid rows found in: {files}")
    return (np.array(rows_p, dtype=float),
            np.array(rows_y, dtype=float),
            np.array(rows_t, dtype=float))


def _aggregate(X_raw, y_raw, t_raw):
    """Average replicates: same (rt, T, cat)  mean yield/turnover."""
    gy, gt = defaultdict(list), defaultdict(list)
    for i in range(len(X_raw)):
        key = (round(float(X_raw[i, 0]), _RT_ROUND),
               round(float(X_raw[i, 1]), _TEMP_ROUND),
               round(float(X_raw[i, 2]), _CAT_ROUND))
        gy[key].append(y_raw[i])
        gt[key].append(t_raw[i])
    keys = list(gy.keys())
    X = np.array([[k[0], k[1], k[2]] for k in keys], dtype=float)
    y = np.array([np.mean(gy[k]) for k in keys], dtype=float)
    t = np.array([np.mean(gt[k]) for k in keys], dtype=float)
    return X, y, t


def _normalise(X):
    """Normalise (rt, T, cat) to [0,1] for distance computation."""
    Xn = X.copy().astype(float)
    Xn[:, 0] = (Xn[:, 0] - 1.0)   / _RT_NORM
    Xn[:, 1] = (Xn[:, 1] - 30.0)  / _T_NORM
    Xn[:, 2] = (Xn[:, 2] - 0.489) / _CAT_NORM
    return Xn


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

class SuzukiReactionSimulator:
    """Suzuki yield + turnover simulator (v3).

    Kernel: linear RBF, smoothing=0 (exact on aggregated unique points).
    Fallback: distance-weighted kNN (k=5, threshold=0.10).
    Accuracy on full 356-point set: in-sample MAE ~0; LOO MAE 17.66%
    (physical floor: dataset has d<0.01 yet Delta-yield>40% neighbors).

    Usage::

        sim = SuzukiReactionSimulator()
        result = sim.query({
            "ligand": "L0", "res_time": 5.0,
            "temperature": 73.5, "catalyst_loading": 2.501,
        })
        # -> {"yield_pct": float, "turnover": float, "reaction_time_min": float}
    """

    # Realistic HPLC measurement noise. Was 3.0 — over-pessimistic and
    # double-counted with the RBF interpolator's ~17% error in sparse regions,
    # causing BO's GP to learn an inflated likelihood noise that suppressed EI.
    NOISE_STD_YIELD    = 0.5
    NOISE_STD_TURNOVER = 1.0

    def __init__(self, data_path=None, seed=None):
        self.rng = np.random.RandomState(seed)

        self._yield_interps:    Dict[int, Optional[RBFInterpolator]] = {}
        self._turnover_interps: Dict[int, Optional[RBFInterpolator]] = {}
        self._unique_X_norm:    Dict[int, np.ndarray] = {}
        self._unique_X_raw:     Dict[int, np.ndarray] = {}
        self._unique_y:         Dict[int, np.ndarray] = {}
        self._unique_t:         Dict[int, np.ndarray] = {}

        self._params    = None
        self._yields    = None
        self._turnovers = None

        self._load_and_fit(data_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def query(self, params: Dict) -> Dict:
        """Noisy query — simulates experimental measurement uncertainty."""
        lig, rt, temp, cat = self._clip(params)
        y   = self._predict(lig, rt, temp, cat, yield_=True)
        ton = self._predict(lig, rt, temp, cat, yield_=False)
        y   += self.rng.normal(0.0, self.NOISE_STD_YIELD)
        ton += self.rng.normal(0.0, self.NOISE_STD_TURNOVER)
        return {"yield_pct":          float(np.clip(y, 0., 100.)),
                "turnover":           float(np.clip(ton, 0., None)),
                "reaction_time_min":  rt}

    def query_noiseless(self, params: Dict) -> Dict:
        """Noiseless query — for analysis and BO oracle benchmarking."""
        lig, rt, temp, cat = self._clip(params)
        y   = self._predict(lig, rt, temp, cat, yield_=True)
        ton = self._predict(lig, rt, temp, cat, yield_=False)
        return {"yield_pct":          float(np.clip(y, 0., 100.)),
                "turnover":           float(np.clip(ton, 0., None)),
                "reaction_time_min":  rt}

    def get_true_optimum(self) -> Dict:
        return {
            "global_optimum":   {"ligand": "L0", "res_time": 5.00, "temperature": 73.5,
                                 "catalyst_loading": 2.501, "yield_pct": 99.8, "turnover": 39.9},
            "l3_local_optimum": {"ligand": "L3", "res_time": 3.15, "temperature": 110.0,
                                 "catalyst_loading": 2.508, "yield_pct": 98.7, "turnover": 39.4},
            "l1_local_optimum": {"ligand": "L1", "res_time": 4.61, "temperature": 69.3,
                                 "catalyst_loading": 2.507, "yield_pct": 99.5, "turnover": 39.7},
        }

    def get_param_space(self) -> Dict:
        return PARAM_SPACE

    def get_data_stats(self) -> Dict:
        if self._yields is None:
            return {}
        lig_stats = {}
        for li in range(7):
            mask = self._params[:, 0] == li
            if not mask.any():
                continue
            ys = self._yields[mask]
            lig_stats[f"L{li}"] = {
                "n_obs":        int(mask.sum()),
                "n_unique_pts": int(len(self._unique_X_raw.get(li, []))),
                "max_yield":    float(ys.max()),
                "mean_yield":   float(ys.mean()),
                "min_yield":    float(ys.min()),
            }
        return {
            "total_obs":      len(self._yields),
            "yield_range":    [float(self._yields.min()), float(self._yields.max())],
            "turnover_range": [float(self._turnovers.min()), float(self._turnovers.max())],
            "rt_range_min":   [float(self._params[:, 1].min()), float(self._params[:, 1].max())],
            "temp_range":     [float(self._params[:, 2].min()), float(self._params[:, 2].max())],
            "cat_range":      [float(self._params[:, 3].min()), float(self._params[:, 3].max())],
            "per_ligand":     lig_stats,
        }

    # ------------------------------------------------------------------
    # Internal prediction — linear RBF + kNN hybrid (v3)
    # ------------------------------------------------------------------

    def _predict(self, lig: int, rt: float, temp: float, cat: float,
                 yield_: bool) -> float:
        """
        Distance-aware hybrid prediction (v3: linear kernel).

        1. Linear RBF exact interpolation on aggregated unique points.
        2. If normalised distance to nearest training point > threshold,
           linearly blend in distance-weighted kNN to suppress extrapolation
           artifacts from the linear kernel at the domain boundary.

        Linear kernel (f(r)=r) degrades gracefully to linear extrapolation
        outside the convex hull of training data — safer than TPS (f(r)=r²logr)
        which has unbounded oscillation at boundaries. Combined with kNN
        fallback this achieves the best LOO accuracy on this dataset.
        """
        xq_raw   = np.array([[rt, temp, cat]])
        interp   = self._yield_interps[lig] if yield_ else self._turnover_interps[lig]
        rbf_pred = float(interp(xq_raw)[0]) if interp is not None else 50.0

        Xn    = self._unique_X_norm.get(lig)
        y_arr = self._unique_y.get(lig) if yield_ else self._unique_t.get(lig)

        if Xn is None or len(Xn) == 0:
            return float(np.clip(rbf_pred, 0., 100. if yield_ else None))

        xqn   = np.array([[(rt   - 1.0)   / _RT_NORM,
                            (temp - 30.0)  / _T_NORM,
                            (cat  - 0.489) / _CAT_NORM]])
        dists = cdist(xqn, Xn)[0]
        min_d = float(dists.min())

        alpha = float(np.clip(min_d / _FALLBACK_THRESHOLD, 0., 1.))
        if alpha == 0.0:
            pred = rbf_pred
        else:
            k       = min(_KNN_K, len(dists))
            top_k   = np.argsort(dists)[:k]
            weights = 1.0 / (dists[top_k] + 1e-10)
            knn     = float(np.average(y_arr[top_k], weights=weights))
            pred    = (1. - alpha) * rbf_pred + alpha * knn

        return float(np.clip(pred, 0., 100. if yield_ else None))

    # ------------------------------------------------------------------
    # Load & fit
    # ------------------------------------------------------------------

    def _clip(self, params: Dict) -> Tuple[int, float, float, float]:
        lig = self._parse_ligand(params["ligand"])
        rt  = float(np.clip(params["res_time"],         1.0,   10.0))
        T   = float(np.clip(params["temperature"],      30.0,  110.0))
        cat = float(np.clip(params["catalyst_loading"], 0.489, 2.516))
        return lig, rt, T, cat

    def _load_and_fit(self, data_path):
        csv_files = self._resolve(data_path)
        if not csv_files:
            if self._yield_interps:
                return
            raise FileNotFoundError(
                f"Cannot find data_*.csv files.\n"
                f"Search dirs: {_SEARCH_DIRS}\n"
                f"Specified path: {data_path}"
            )
        params, yields, turnovers = _parse_csvs(csv_files)
        self._params    = params
        self._yields    = yields
        self._turnovers = turnovers

        n_ligs = len(set(int(params[i, 0]) for i in range(len(params))))
        print(f"[SuzukiSimulator v3] Loaded: {len(yields)} obs from {len(csv_files)} files | "
              f"{n_ligs} ligands | yield ∈ [{yields.min():.1f}, {yields.max():.1f}]%")
        self._fit(params, yields, turnovers)

    def _fit(self, params, yields, turnovers):
        total_unique = 0
        for li in range(7):
            mask = params[:, 0] == li
            if not mask.any():
                continue

            X_raw = params[mask, 1:4]
            y_raw = yields[mask]
            t_raw = turnovers[mask]

            # Step 1: aggregate replicates  stable interpolation targets
            X_uniq, y_uniq, t_uniq = _aggregate(X_raw, y_raw, t_raw)
            if len(X_uniq) < 3:
                X_uniq, y_uniq, t_uniq = X_raw.copy(), y_raw.copy(), t_raw.copy()

            self._unique_X_raw[li]  = X_uniq
            self._unique_X_norm[li] = _normalise(X_uniq)
            self._unique_y[li]      = y_uniq
            self._unique_t[li]      = t_uniq
            total_unique += len(X_uniq)

            # Step 2: exact linear RBF fit (v3 core change vs v2)
            smooth = _RBF_SMOOTHING if len(X_uniq) >= 4 else 0.01
            for arr, store in [(y_uniq, self._yield_interps),
                               (t_uniq, self._turnover_interps)]:
                try:
                    store[li] = RBFInterpolator(X_uniq, arr,
                                                kernel=_RBF_KERNEL,
                                                smoothing=smooth)
                except np.linalg.LinAlgError:
                    try:
                        store[li] = RBFInterpolator(X_uniq, arr,
                                                    kernel=_RBF_KERNEL,
                                                    smoothing=0.1)
                    except Exception:
                        store[li] = None

        # In-sample validation on all original observations
        errs = []
        for i in range(len(params)):
            li = int(params[i, 0])
            rt, T, cat = params[i, 1], params[i, 2], params[i, 3]
            errs.append(abs(self._predict(li, rt, T, cat, yield_=True) - yields[i]))
        errs = np.array(errs)
        print(f"  Fit done | kernel={_RBF_KERNEL} | "
              f"{len(params)} obs  {total_unique} unique pts | "
              f"In-sample MAE={errs.mean():.3f}%  "
              f"|e|≤3%={(errs <= 3).mean() * 100:.1f}%  "
              f"|e|≤5%={(errs <= 5).mean() * 100:.1f}%")

        ref  = TRUE_OPTIMUM
        y_opt = self._predict(LIGAND_TO_IDX[ref["ligand"]],
                              ref["res_time"], ref["temperature"],
                              ref["catalyst_loading"], yield_=True)
        print(f"  Global optimum check: true={ref['yield_pct']:.1f}%  "
              f"pred={y_opt:.1f}%  err={abs(y_opt - ref['yield_pct']):.2f}%")

    def _resolve(self, data_path):
        if data_path is None:
            return _find_csvs(_SEARCH_DIRS)
        p = os.path.normpath(data_path)
        if os.path.isdir(p):
            hits = sorted(glob.glob(os.path.join(p, "data_*.csv")))
            if hits:
                return hits
        if p.endswith(".csv") and os.path.isfile(p):
            d = os.path.dirname(p)
            hits = sorted(glob.glob(os.path.join(d, "data_*.csv")))
            return hits if hits else [p]
        if p.endswith(".pkl"):
            d = os.path.dirname(p)
            hits = sorted(glob.glob(os.path.join(d, "data_*.csv")))
            if hits:
                return hits
            hits = _find_csvs(_SEARCH_DIRS)
            if hits:
                return hits
            try:
                import pickle, warnings
                with open(p, "rb") as fh:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        d_pkl = pickle.load(fh)
                raw_p = np.array(d_pkl["params"], dtype=float)
                raw_v = np.array(d_pkl["values"], dtype=float)
                lig   = np.argmax(raw_p[:, :7], axis=1).astype(float)
                pa    = np.column_stack([lig, raw_p[:, 7] / 60., raw_p[:, 8], raw_p[:, 9]])
                self._params    = pa
                self._yields    = raw_v[:, 0]
                self._turnovers = raw_v[:, 1]
                self._fit(pa, raw_v[:, 0], raw_v[:, 1])
                return []
            except Exception as exc:
                raise FileNotFoundError(f"Cannot read '{p}': {exc}")
        return _find_csvs(_SEARCH_DIRS)

    @staticmethod
    def _parse_ligand(ligand) -> int:
        if isinstance(ligand, str):
            if ligand not in LIGAND_TO_IDX:
                raise ValueError(f"Unknown ligand '{ligand}'. Valid: {LIGAND_NAMES}")
            return LIGAND_TO_IDX[ligand]
        idx = int(ligand)
        if not (0 <= idx < 7):
            raise ValueError(f"Ligand index {idx} out of range [0,6].")
        return idx


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    data_arg = sys.argv[1] if len(sys.argv) > 1 else None
    try:
        sim = SuzukiReactionSimulator(data_path=data_arg, seed=42)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    print()
    print("=== Data stats ===")
    stats = sim.get_data_stats()
    print(f"Total obs: {stats['total_obs']}, yield range: {stats['yield_range']}")
    for lig, ls in stats["per_ligand"].items():
        print(f"  {lig}: n_obs={ls['n_obs']}, n_unique={ls['n_unique_pts']}, "
              f"max_yield={ls['max_yield']:.1f}%")

    print()
    print("=== Noiseless queries ===")
    for cond in [
        {"ligand": "L0", "res_time": 5.00, "temperature": 73.5,  "catalyst_loading": 2.501},
        {"ligand": "L3", "res_time": 3.15, "temperature": 110.0, "catalyst_loading": 2.508},
        {"ligand": "L3", "res_time": 5.00, "temperature": 110.0, "catalyst_loading": 2.508},
        {"ligand": "L4", "res_time": 10.0, "temperature": 110.0, "catalyst_loading": 2.499},
    ]:
        r = sim.query_noiseless(cond)
        print(f"  L={cond['ligand']} rt={cond['res_time']:.2f} T={cond['temperature']:.0f} "
              f"cat={cond['catalyst_loading']:.3f}  yield={r['yield_pct']:.1f}%")

    print()
    print("=== BO tension: best yield ≠ shortest rt (L0, T=73.5, cat=2.501) ===")
    for rt in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]:
        r = sim.query_noiseless({"ligand": "L0", "res_time": rt,
                                 "temperature": 73.5, "catalyst_loading": 2.501})
        bar = "█" * int(r["yield_pct"] / 5)
        print(f"    rt={rt:4.1f}min  yield={r['yield_pct']:5.1f}%  {bar}")