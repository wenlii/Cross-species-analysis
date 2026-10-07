"""Inputs: CROSS_SPECIES_PREDICTION_DIR/. Outputs: CROSS_SPECIES_OUTPUT_DIR/hsdi/."""

from __future__ import annotations

import os
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = Path(os.environ.get("CROSS_SPECIES_OUTPUT_DIR", str(PACKAGE_ROOT / "outputs")))
PREDICTION_DIR = Path(os.environ.get("CROSS_SPECIES_PREDICTION_DIR", str(OUTPUT_ROOT / "prediction_outputs")))
HSDI_DIR = OUTPUT_ROOT / "hsdi"
HSDI_DIR.mkdir(parents=True, exist_ok=True)

import ast

import json

import warnings

from dataclasses import dataclass, asdict


from typing import Any, Dict, List, Optional, Sequence, Tuple

import joblib

import numpy as np

import pandas as pd

from patsy import dmatrix

from sklearn.linear_model import Ridge

from sklearn.metrics import r2_score

from sklearn.model_selection import GroupKFold

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler

from scipy.stats import pearsonr, spearmanr

warnings.filterwarnings(
    "ignore",
    message=r"Found unknown categories in columns .* encoded as all zeros",
    category=UserWarning,
)

FINALFIG_DIR = PREDICTION_DIR

PREDICT_DIR = PREDICTION_DIR

HS_DIR = HSDI_DIR

HUMAN_TRIPLES_CSV = FINALFIG_DIR / "human_triples_rate12_to_rate23.csv"

MONKEY_TRIPLES_CSV = FINALFIG_DIR / "monkey_triples_rate12_to_rate23.csv"

SPLINE_SPEC_CSV = FINALFIG_DIR / "spline_spec.csv"

METHOD_NAME = "fig5_new_baselineT2_noCratio"

REGIONS: Tuple[str, ...] = ("low_expansion", "high_expansion")

METRICS: Tuple[str, ...] = ("CT", "SA")

BOUND_EPS = 1e-6

SEX_CANONICAL_MALE = "male"

SEX_CANONICAL_FEMALE = "female"

@dataclass(frozen=True)
class PredictMethodConfig:
    name: str
    file_tag: str
    variant_label: str
    spline_degree: int
    n_spline_basis: int
    use_zero_intercept_spline: bool
    clip_features_to_human_quantiles: bool
    clip_lo_q: float
    clip_hi_q: float
    filter_extreme_rates: bool
    rate_abs_max: float
    dt_qc_kind: str
    dt_qc_min: float
    include_baseline_t2: bool
    residualize_baseline_t2: bool
    include_c_ratio: bool
    ratio_eps: float
    age_ratio: float
    monkey_rate_dt_scale: float
    conf_cols: Tuple[str, ...]
    missing_token: str
    monkey_adjust_sex: bool
    monkey_use_human_ref_for_scanner_site: bool
    ridge_alpha_rate: float
    ridge_alpha_baseline: float
    enable_hat_clip: bool
    hat_clip_q_lo: float
    hat_clip_q_hi: float
    hat_clip_fallback_abs: float
    enable_monkey_calibration: bool
    calibration_n_splits: int
    calibration_space: str
    allow_fit_all_calibration: bool
    n_splits: int
    alphas: Tuple[float, ...]
    l1_ratios: Tuple[float, ...]
    var_std_min: float
    const_std_min: float

@dataclass(frozen=True)
class ReverseConfig:
    ridge_alphas: Tuple[float, ...] = (1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0)
    n_splits: int = 5
    age_match_use_human_q: Tuple[float, float] = (0.01, 0.99)
    age_match_buffer_months_init: float = 6.0
    age_match_buffer_months_max: float = 84.0
    age_match_buffer_growth: float = 12.0
    min_monkey_n_per_cell: int = 30
    filter_extreme_rates: bool = True
    rate_abs_max: float = 8.0
    dt_used_min: float = 2.0
    var_std_min_inter: float = 1e-3
    const_std_min: float = 1e-6
    clip_features_in_monkey_train: bool = True
    clip_features_in_human_infer: bool = False
    clip_lo_q: float = 0.002
    clip_hi_q: float = 0.998
    ood_use_q_range: Tuple[float, float] = (0.005, 0.995)
    ood_z_abs_thresh: float = 6.0
    use_robust_scaler: bool = True
    add_human_rule_deviation: bool = True
    z_scale_floor: float = 1e-3
    z_scale_floor_frac_y: float = 0.05

METHODS: Dict[str, PredictMethodConfig] = {
    "fig5_new_baselineT2_noCratio": PredictMethodConfig(
        name="fig5_new_baselineT2_noCratio",
        file_tag="noCratio",
        variant_label="baselineT2_noCratio",
        spline_degree=3,
        n_spline_basis=5,
        use_zero_intercept_spline=False,
        clip_features_to_human_quantiles=True,
        clip_lo_q=0.001,
        clip_hi_q=0.999,
        filter_extreme_rates=True,
        rate_abs_max=5.0,
        dt_qc_kind="used",
        dt_qc_min=1.0,
        include_baseline_t2=True,
        residualize_baseline_t2=True,
        include_c_ratio=False,
        ratio_eps=1e-6,
        age_ratio=2.83,
        monkey_rate_dt_scale=2.83,
        conf_cols=("sex", "site", "scanner_manufacturer", "scanner_model"),
        missing_token="__MISSING__",
        monkey_adjust_sex=True,
        monkey_use_human_ref_for_scanner_site=True,
        ridge_alpha_rate=0.2,
        ridge_alpha_baseline=1.0,
        enable_hat_clip=True,
        hat_clip_q_lo=0.001,
        hat_clip_q_hi=0.999,
        hat_clip_fallback_abs=100.0,
        enable_monkey_calibration=True,
        calibration_n_splits=3,
        calibration_space="raw",
        allow_fit_all_calibration=True,
        n_splits=5,
        alphas=(1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1),
        l1_ratios=(0.0, 0.1, 0.3, 0.5),
        var_std_min=1e-3,
        const_std_min=1e-12,
    ),
}

REVERSE_CFG = ReverseConfig()

ACTIVE_METHOD_CFG = METHODS[METHOD_NAME]

MISSING_TOKEN = str(ACTIVE_METHOD_CFG.missing_token)

CONF_COLS: Tuple[str, ...] = tuple(ACTIVE_METHOD_CFG.conf_cols)

class RateResidualizer:


    def __init__(self, conf_cols: Sequence[str], missing_token: str = MISSING_TOKEN, ridge_alpha: float = 1.0):
        self.conf_cols = list(conf_cols)
        self.missing_token = str(missing_token)
        self.ridge_alpha = float(ridge_alpha)
        self.encoder: Optional[OneHotEncoder] = None
        self.lr_prev: Optional[Ridge] = None
        self.lr_next: Optional[Ridge] = None

    def _get_missing_token(self) -> str:
        return str(getattr(self, "missing_token", MISSING_TOKEN))

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)
        if not hasattr(self, "missing_token"):
            self.missing_token = MISSING_TOKEN

    def _prep_conf(self, conf_df: pd.DataFrame) -> pd.DataFrame:
        out = conf_df.copy()
        missing_token = self._get_missing_token()
        conf_cols = list(getattr(self, "conf_cols", CONF_COLS))
        for c in conf_cols:
            if c not in out.columns:
                out[c] = missing_token
            out[c] = out[c].fillna(missing_token).astype(str).str.strip()
            out.loc[out[c] == "", c] = missing_token
        return out[conf_cols]

    def hat_prev(self, conf_df: pd.DataFrame) -> np.ndarray:
        Z = self.encoder.transform(self._prep_conf(conf_df))
        return self.lr_prev.predict(Z).ravel()

    def hat_next(self, conf_df: pd.DataFrame) -> np.ndarray:
        Z = self.encoder.transform(self._prep_conf(conf_df))
        return self.lr_next.predict(Z).ravel()

    def transform_prev(self, conf_df: pd.DataFrame, x: np.ndarray) -> np.ndarray:
        return np.asarray(x, float) - self.hat_prev(conf_df)

    def transform_next(self, conf_df: pd.DataFrame, x: np.ndarray) -> np.ndarray:
        return np.asarray(x, float) - self.hat_next(conf_df)

class SingleVarResidualizer:
    def __init__(self, conf_cols: Sequence[str], missing_token: str = MISSING_TOKEN, ridge_alpha: float = 1.0):
        self.conf_cols = list(conf_cols)
        self.missing_token = str(missing_token)
        self.ridge_alpha = float(ridge_alpha)
        self.encoder: Optional[OneHotEncoder] = None
        self.lr: Optional[Ridge] = None

    def _get_missing_token(self) -> str:
        return str(getattr(self, "missing_token", MISSING_TOKEN))

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)
        if not hasattr(self, "missing_token"):
            self.missing_token = MISSING_TOKEN

    def _prep_conf(self, conf_df: pd.DataFrame) -> pd.DataFrame:
        out = conf_df.copy()
        missing_token = self._get_missing_token()
        conf_cols = list(getattr(self, "conf_cols", CONF_COLS))
        for c in conf_cols:
            if c not in out.columns:
                out[c] = missing_token
            out[c] = out[c].fillna(missing_token).astype(str).str.strip()
            out.loc[out[c] == "", c] = missing_token
        return out[conf_cols]

    def hat(self, conf_df: pd.DataFrame) -> np.ndarray:
        Z = self.encoder.transform(self._prep_conf(conf_df))
        return self.lr.predict(Z).ravel()

    def transform(self, conf_df: pd.DataFrame, x: np.ndarray) -> np.ndarray:
        return np.asarray(x, float) - self.hat(conf_df)

def _read_series_csv(path: Path) -> Dict[str, str]:
    s = pd.read_csv(path, header=None, index_col=0)
    if s.shape[1] == 0:
        return {}
    return s.iloc[:, 0].astype(str).to_dict()

def _parse_float_list(x: Any) -> List[float]:
    if x is None:
        return []
    if isinstance(x, np.ndarray):
        return [float(v) for v in x.ravel()]
    if isinstance(x, (list, tuple)):
        return [float(v) for v in x]

    s = str(x).strip()
    if s == "" or s.lower() == "nan":
        return []

    try:
        obj = ast.literal_eval(s)
        if isinstance(obj, (list, tuple, np.ndarray)):
            return [float(v) for v in np.asarray(obj, float).ravel()]
        if isinstance(obj, (int, float)):
            return [float(obj)]
    except Exception:
        pass

    s = s.strip("[]()")
    toks = [tok for tok in s.replace(";", ",").replace(" ", ",").split(",") if tok]
    out: List[float] = []
    for tok in toks:
        try:
            out.append(float(tok))
        except Exception:
            pass
    return out

def save_json(obj: Dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def detect_and_convert_age_to_months(series: pd.Series) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce").astype(float)
    med = vals.dropna().median()
    if np.isnan(med):
        return vals
    if med < 30:
        return vals * 12.0
    return vals

def safe_log_ratio(num: np.ndarray, den: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    num = np.asarray(num, float)
    den = np.asarray(den, float)
    return np.log((num + eps) / (den + eps))

def normalize_sex_codes(series: pd.Series, dataset: str) -> pd.Series:
    def _map_single(x: Any) -> str:
        if pd.isna(x):
            return str(x)
        s = str(x).strip()
        if s == "":
            return s
        sl = s.lower()
        if dataset == "human":
            if sl in {"1", "m", "male", "man"}:
                return SEX_CANONICAL_MALE
            if sl in {"2", "f", "female", "woman"}:
                return SEX_CANONICAL_FEMALE
        if dataset == "macaque":
            if sl in {"1", "m", "male", "man"}:
                return SEX_CANONICAL_MALE
            if sl in {"0", "f", "female", "woman"}:
                return SEX_CANONICAL_FEMALE
        return s

    return series.map(_map_single)

def _ensure_conf_cols(df: pd.DataFrame, conf_cols: Sequence[str], missing_token: str) -> pd.DataFrame:
    out = df.copy()
    for c in conf_cols:
        if c not in out.columns:
            out[c] = missing_token
        out[c] = out[c].astype(str).fillna(missing_token).str.strip()
        out.loc[out[c] == "", c] = missing_token
    return out

def _read_spline_spec(path: Path) -> Dict[str, Any]:
    d = _read_series_csv(path)
    knots = _parse_float_list(d.get("knots", ""))
    return {
        "lb": float(d["lb"]),
        "ub": float(d["ub"]),
        "knots": np.asarray(knots, float),
        "degree": int(float(d.get("degree", 3))),
        "age_ratio": float(d.get("AGE_RATIO", 2.83)),
        "include_baseline_t2": int(float(d.get("include_baseline_T2", 1))),
        "include_c_ratio": int(float(d.get("include_c_ratio", 1))),
        "residualize_baseline_t2": int(float(d.get("residualize_baseline_T2", 0))),

        "use_zero_intercept_spline": int(float(d.get("use_zero_intercept_spline", 0))),
    }

def add_age_spline_with_spec(df: pd.DataFrame, age_col: str, spec: Dict[str, Any]) -> Tuple[pd.DataFrame, List[str]]:
    ages = pd.to_numeric(df[age_col], errors="coerce").astype(float).values
    lb = float(spec["lb"])
    ub = float(spec["ub"])
    degree = int(spec["degree"])
    use_zero_intercept = bool(int(spec.get("use_zero_intercept_spline", 0)))
    ages = np.clip(ages, lb + BOUND_EPS, ub - BOUND_EPS)

    formula = (
        "0 + bs(age, knots=knots, degree=deg, include_intercept=False, lower_bound=lb, upper_bound=ub)"
        if use_zero_intercept
        else "bs(age, knots=knots, degree=deg, include_intercept=False, lower_bound=lb, upper_bound=ub)"
    )
    dm = dmatrix(
        formula,
        {"age": ages, "knots": spec["knots"], "deg": degree, "lb": lb, "ub": ub},
        return_type="dataframe",
    )
    spline_cols = [f"age_spline_{i:d}" for i in range(dm.shape[1])]
    dm.columns = spline_cols
    out = pd.concat([df.reset_index(drop=True), dm.reset_index(drop=True)], axis=1)
    return out, spline_cols

def filter_extreme_rows_for_metric(
    df: pd.DataFrame,
    r12_col: str,
    r23_col: str,
    rate_abs_max: float,
    dt_used_min: float,
) -> pd.DataFrame:
    out = df.copy()
    r12 = pd.to_numeric(out[r12_col], errors="coerce").astype(float)
    r23 = pd.to_numeric(out[r23_col], errors="coerce").astype(float)
    dt12 = pd.to_numeric(out["dt12_months_used"], errors="coerce").astype(float)
    dt23 = pd.to_numeric(out["dt23_months_used"], errors="coerce").astype(float)

    ok = np.isfinite(r12) & np.isfinite(r23) & np.isfinite(dt12) & np.isfinite(dt23)
    ok &= (np.abs(r12) <= float(rate_abs_max)) & (np.abs(r23) <= float(rate_abs_max))
    ok &= (dt12 >= float(dt_used_min)) & (dt23 >= float(dt_used_min))
    return out.loc[ok].copy()

def _filter_monkeys_by_human_age_window(
    monkey_df: pd.DataFrame,
    human_age_months: np.ndarray,
    age_ratio: float,
    reverse_cfg: ReverseConfig,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    hum = np.asarray(human_age_months, float)
    hum = hum[np.isfinite(hum)]
    hum_lo = float(np.quantile(hum, reverse_cfg.age_match_use_human_q[0]))
    hum_hi = float(np.quantile(hum, reverse_cfg.age_match_use_human_q[1]))

    target_lo_mon = hum_lo / float(age_ratio)
    target_hi_mon = hum_hi / float(age_ratio)

    mon = monkey_df.copy()
    mon["age_T2"] = pd.to_numeric(mon["age_T2"], errors="coerce").astype(float)

    buf = float(reverse_cfg.age_match_buffer_months_init)
    best = mon.iloc[0:0].copy()
    while buf <= float(reverse_cfg.age_match_buffer_months_max):
        sub = mon.loc[
            mon["age_T2"].between(target_lo_mon - buf, target_hi_mon + buf, inclusive="both")
        ].copy()
        best = sub
        if len(sub) >= int(reverse_cfg.min_monkey_n_per_cell):
            break
        buf += float(reverse_cfg.age_match_buffer_growth)

    return best, {
        "human_age_q_lo_months": hum_lo,
        "human_age_q_hi_months": hum_hi,
        "target_monkey_lo_months": target_lo_mon,
        "target_monkey_hi_months": target_hi_mon,
        "buffer_used_months": float(min(buf, reverse_cfg.age_match_buffer_months_max)),
        "n_monkey_after_age_match": int(len(best)),
    }

def build_design_matrix_rate12_to_rate23(
    df: pd.DataFrame,
    r12_col: str,
    spline_cols: Sequence[str],
    include_baseline_t2: bool,
    baseline_t2_col: Optional[str],
    include_c_ratio: bool,
    ratio_col: Optional[str],
    use_interactions: bool = True,
) -> Tuple[np.ndarray, List[str]]:

    parts: List[np.ndarray] = []
    names: List[str] = []

    xr = pd.to_numeric(df[r12_col], errors="coerce").values.astype(float).reshape(-1, 1)
    xdt = pd.to_numeric(df["dt23_months_used"], errors="coerce").values.astype(float).reshape(-1, 1)
    xs = df[list(spline_cols)].apply(pd.to_numeric, errors="coerce").values.astype(float)

    parts.extend([xr, xdt, xs])
    names.extend([r12_col, "dt23_months_used"])
    names.extend(list(spline_cols))

    xb = None
    if include_baseline_t2 and baseline_t2_col is not None:
        xb = pd.to_numeric(df[baseline_t2_col], errors="coerce").values.astype(float).reshape(-1, 1)
        parts.append(xb)
        names.append(baseline_t2_col)

    xratio = None
    if include_c_ratio and ratio_col is not None:
        xratio = pd.to_numeric(df[ratio_col], errors="coerce").values.astype(float).reshape(-1, 1)
        parts.append(xratio)
        names.append(ratio_col)

    if use_interactions:
        parts.append(xs * xr)
        names.extend([f"{r12_col}*{sc}" for sc in spline_cols])

        parts.append(xs * xdt)
        names.extend([f"dt23_months_used*{sc}" for sc in spline_cols])

        if xb is not None:
            parts.append(xs * xb)
            names.extend([f"{baseline_t2_col}*{sc}" for sc in spline_cols])

        if xratio is not None:
            parts.append(xs * xratio)
            names.extend([f"{ratio_col}*{sc}" for sc in spline_cols])

    return np.concatenate(parts, axis=1), names

def compute_keep_mask_low_variance(X: np.ndarray, names: Sequence[str], reverse_cfg: ReverseConfig) -> np.ndarray:
    std = np.nanstd(np.asarray(X, float), axis=0)
    keep = np.isfinite(std) & (std >= float(reverse_cfg.const_std_min))
    for j, name in enumerate(names):
        if "*" in str(name):
            keep[j] = bool(keep[j] and (std[j] >= float(reverse_cfg.var_std_min_inter)))
    return keep.astype(bool)

def apply_keep_mask(X: np.ndarray, names: Sequence[str], keep: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    keep = np.asarray(keep, bool)
    return np.asarray(X, float)[:, keep], [str(n) for n, k in zip(names, keep) if k]

def load_feature_mask_csv(mask_path: Path, expected_names: Sequence[str]) -> np.ndarray:
    df = pd.read_csv(mask_path)
    mp = {str(n): int(k) for n, k in zip(df["name"].astype(str), df["keep"].astype(int))}
    return np.array([bool(mp.get(str(n), 0)) for n in expected_names], dtype=bool)

def clip_bounds_from_train(Xtr: np.ndarray, qlo: float, qhi: float) -> Tuple[np.ndarray, np.ndarray]:
    Xtr = np.asarray(Xtr, float)
    return np.quantile(Xtr, qlo, axis=0), np.quantile(Xtr, qhi, axis=0)

def clip_X(X: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(X, float), np.asarray(lo, float), np.asarray(hi, float))

def robust_center_scale(
    err: np.ndarray,
    reverse_cfg: ReverseConfig,
    y_ref: Optional[np.ndarray] = None,
) -> Tuple[float, float, str]:
    err = np.asarray(err, float)
    err = err[np.isfinite(err)]
    if err.size == 0:
        return 0.0, 1.0, "empty"

    center = float(np.median(err))
    mad = float(np.median(np.abs(err - center)))
    scale = 1.4826 * mad

    floor_val = float(reverse_cfg.z_scale_floor)
    if y_ref is not None:
        y_ref_arr = np.asarray(y_ref, float)
        y_ref_arr = y_ref_arr[np.isfinite(y_ref_arr)]
        if y_ref_arr.size > 0:
            floor_val = max(
                floor_val,
                float(reverse_cfg.z_scale_floor_frac_y) * float(np.nanstd(y_ref_arr)),
            )

    if (not np.isfinite(scale)) or (scale < floor_val):
        return center, floor_val, "mad_floored"
    return center, scale, "mad"

def make_scaler(reverse_cfg: ReverseConfig):
    if reverse_cfg.use_robust_scaler:
        return RobustScaler(with_centering=True, with_scaling=True, quantile_range=(25.0, 75.0))
    return StandardScaler()

def compute_ood_metrics_from_monkey_train(
    X_h: np.ndarray,
    X_mon_train: np.ndarray,
    reverse_cfg: ReverseConfig,
) -> pd.DataFrame:
    X_h = np.asarray(X_h, float)
    X_mon_train = np.asarray(X_mon_train, float)

    qlo, qhi = reverse_cfg.ood_use_q_range
    lo = np.quantile(X_mon_train, qlo, axis=0)
    hi = np.quantile(X_mon_train, qhi, axis=0)

    med = np.nanmedian(X_mon_train, axis=0)
    mad = np.nanmedian(np.abs(X_mon_train - med.reshape(1, -1)), axis=0)
    scale = 1.4826 * mad

    ok = np.isfinite(scale) & (scale > 1e-12)
    outside = (X_h < lo.reshape(1, -1)) | (X_h > hi.reshape(1, -1))
    ood_frac = outside[:, ok].mean(axis=1) if ok.sum() > 0 else np.zeros(X_h.shape[0], dtype=float)

    z = np.zeros_like(X_h, dtype=float)
    z[:, ok] = (X_h[:, ok] - med[ok].reshape(1, -1)) / scale[ok].reshape(1, -1)

    ood_max_abs_z = np.max(np.abs(z[:, ok]), axis=1) if ok.sum() > 0 else np.zeros(X_h.shape[0], dtype=float)
    ood_n_absz_gt = (
        np.sum(np.abs(z[:, ok]) >= float(reverse_cfg.ood_z_abs_thresh), axis=1)
        if ok.sum() > 0
        else np.zeros(X_h.shape[0], dtype=int)
    )
    return pd.DataFrame(
        {
            "OOD_frac_q": ood_frac.astype(float),
            "OOD_max_abs_z": ood_max_abs_z.astype(float),
            "OOD_n_feat_absz_ge_thresh": ood_n_absz_gt.astype(int),
        }
    )

def _rename_oof_cols_for_merge(oof: pd.DataFrame) -> pd.DataFrame:
    o = oof.copy()
    if "id" not in o.columns and "subject_id" in o.columns:
        o = o.rename(columns={"subject_id": "id"})

    need_keys = ["id", "scan_1", "scan_2", "scan_3"]
    for k in need_keys:
        if k not in o.columns:
            raise KeyError(f"OOF file missing key column '{k}'.")

    if "r23_resid" not in o.columns or "pred_r23_resid" not in o.columns:
        raise KeyError("OOF file missing required columns: r23_resid, pred_r23_resid.")

    o = o.rename(
        columns={
            "r23_resid": "r23_resid_humanOOF",
            "pred_r23_resid": "pred_r23_resid_humanOOF",
        }
    )
    return o[need_keys + ["r23_resid_humanOOF", "pred_r23_resid_humanOOF"]].copy()

def _flatten_pivot_columns(wide: pd.DataFrame, key_cols: List[str]) -> pd.DataFrame:
    cols = wide.columns
    if isinstance(cols, pd.MultiIndex):
        new_cols = []
        for a, b in cols.to_list():
            a_str = str(a).strip()
            b_str = "" if b is None else str(b).strip()
            if a_str in key_cols and (b_str == "" or b_str.lower() == "nan"):
                new_cols.append(a_str)
            else:
                new_cols.append(f"HSdevZ_{a_str}_{b_str}")
        wide.columns = [c.strip() for c in new_cols]
        return wide

    new_cols = []
    for c in cols:
        if isinstance(c, tuple) and len(c) == 2:
            a, b = c
            a_str = str(a).strip()
            b_str = "" if b is None else str(b).strip()
            if a_str in key_cols and (b_str == "" or b_str.lower() == "nan"):
                new_cols.append(a_str)
            else:
                new_cols.append(f"HSdevZ_{a_str}_{b_str}")
        else:
            new_cols.append(str(c).strip())
    wide.columns = new_cols
    return wide

def _make_monkey_conf_for_transform(
    df: pd.DataFrame,
    human_ref_conf: Dict[str, str],
    method_cfg: PredictMethodConfig,
) -> pd.DataFrame:
    conf = df[list(method_cfg.conf_cols)].copy()
    for c in method_cfg.conf_cols:
        if c not in conf.columns:
            conf[c] = method_cfg.missing_token
        conf[c] = conf[c].astype(str).fillna(method_cfg.missing_token)

    if not method_cfg.monkey_adjust_sex:
        conf["sex"] = human_ref_conf.get("sex", method_cfg.missing_token)

    if method_cfg.monkey_use_human_ref_for_scanner_site:
        conf["site"] = human_ref_conf.get("site", method_cfg.missing_token)
        conf["scanner_manufacturer"] = human_ref_conf.get(
            "scanner_manufacturer", method_cfg.missing_token
        )
        conf["scanner_model"] = human_ref_conf.get("scanner_model", method_cfg.missing_token)

    return conf

def fit_macaque_rule_and_score_humans(
    human_triples: pd.DataFrame,
    monkey_triples: pd.DataFrame,
    region: str,
    metric: str,
    predict_out_dir: Path,
    hs_dir: Path,
    method_cfg: PredictMethodConfig,
    reverse_cfg: ReverseConfig,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    out_prefix = f"R12toR23_{region}_{metric}_{method_cfg.file_tag}"

    human_rr: RateResidualizer = joblib.load(
        predict_out_dir / f"{out_prefix}_human_rate_residualizer.joblib"
    )
    human_ref_conf = _read_series_csv(predict_out_dir / f"{out_prefix}_human_ref_confounds.csv")
    spline_spec = _read_spline_spec(predict_out_dir / "spline_spec.csv")

    human_age_months = pd.to_numeric(
        human_triples["age_T2"],
        errors="coerce",
    ).astype(float)

    if human_age_months.dropna().median() < 30.0:
        human_age_months = human_age_months * 12.0

    monkey_age_human_equivalent = (
        pd.to_numeric(
            monkey_triples["age_T2"],
            errors="coerce",
        ).astype(float)
        * float(spline_spec["age_ratio"])
    )

    spline_spec["lb"] = float(
        min(
            human_age_months.min(),
            monkey_age_human_equivalent.min(),
        )
        - 0.001
    )

    spline_spec["ub"] = float(
        max(
            human_age_months.max(),
            monkey_age_human_equivalent.max(),
        )
        + 0.001
    )


    include_baseline_t2 = bool(int(spline_spec["include_baseline_t2"]))
    include_c_ratio = bool(int(spline_spec["include_c_ratio"]))
    residualize_baseline_t2 = bool(int(spline_spec["residualize_baseline_t2"]))
    age_ratio = float(spline_spec["age_ratio"])

    human_br = None
    br_path = predict_out_dir / f"{out_prefix}_human_baseline_residualizer.joblib"
    if include_baseline_t2 and residualize_baseline_t2 and br_path.exists():
        human_br = joblib.load(br_path)

    r12_raw_col = f"{metric}_{region}_r12"
    r23_raw_col = f"{metric}_{region}_r23"
    base_t1_col = f"{metric}_{region}_base_T1"
    base_t2_col = f"{metric}_{region}_base_T2"
    ratio_col = f"{metric}_{region}_logratio_T2T1"


    hum_age_months = detect_and_convert_age_to_months(human_triples["age_T2"]).values.astype(float)
    hum_age_months = hum_age_months[np.isfinite(hum_age_months)]

    mon0 = _ensure_conf_cols(monkey_triples.copy(), method_cfg.conf_cols, method_cfg.missing_token)
    if "id" not in mon0.columns and "monkey_id" in mon0.columns:
        mon0 = mon0.rename(columns={"monkey_id": "id"})

    mon, age_meta = _filter_monkeys_by_human_age_window(mon0, hum_age_months, age_ratio, reverse_cfg)
    mon["id"] = mon["id"].astype(str)
    mon["age_spline_months"] = pd.to_numeric(mon["age_T2"], errors="coerce").astype(float) * age_ratio
    mon[ratio_col] = safe_log_ratio(
        pd.to_numeric(mon[base_t2_col], errors="coerce"),
        pd.to_numeric(mon[base_t1_col], errors="coerce"),
        eps=method_cfg.ratio_eps,
    )

    req_mon = [
        "id", "age_spline_months", "dt12_months_used", "dt23_months_used",
        r12_raw_col, r23_raw_col, base_t1_col, base_t2_col,
    ] + list(method_cfg.conf_cols)
    if include_c_ratio:
        req_mon.append(ratio_col)
    mon = mon.dropna(subset=req_mon).copy()

    if reverse_cfg.filter_extreme_rates:
        mon = filter_extreme_rows_for_metric(
            mon,
            r12_raw_col,
            r23_raw_col,
            reverse_cfg.rate_abs_max,
            reverse_cfg.dt_used_min,
        )

    mon, spline_cols = add_age_spline_with_spec(mon, "age_spline_months", spline_spec)
    conf_mon = _make_monkey_conf_for_transform(mon, human_ref_conf, method_cfg)

    r12_mon = pd.to_numeric(mon[r12_raw_col], errors="coerce").values.astype(float)
    r23_mon = pd.to_numeric(mon[r23_raw_col], errors="coerce").values.astype(float)
    mon[f"{r12_raw_col}_resid"] = r12_mon - human_rr.hat_prev(conf_mon)
    mon[f"{r23_raw_col}_resid"] = r23_mon - human_rr.hat_next(conf_mon)

    baseline_col_used = base_t2_col
    if include_baseline_t2 and residualize_baseline_t2 and human_br is not None:
        mon[f"{base_t2_col}_resid"] = human_br.transform(
            conf_mon,
            pd.to_numeric(mon[base_t2_col], errors="coerce"),
        )
        baseline_col_used = f"{base_t2_col}_resid"

    X_mon_full, names_full = build_design_matrix_rate12_to_rate23(
        mon,
        r12_col=f"{r12_raw_col}_resid",
        spline_cols=spline_cols,
        include_baseline_t2=include_baseline_t2,
        baseline_t2_col=baseline_col_used,
        include_c_ratio=include_c_ratio,
        ratio_col=ratio_col if include_c_ratio else None,
        use_interactions=True,
    )
    y_mon = mon[f"{r23_raw_col}_resid"].values.astype(float)

    keep_mask_full = compute_keep_mask_low_variance(X_mon_full, names_full, reverse_cfg)
    X_mon_kept, _ = apply_keep_mask(X_mon_full, names_full, keep_mask_full)

    pd.DataFrame({"name": names_full, "keep": keep_mask_full.astype(int)}).to_csv(
        hs_dir / f"{out_prefix}_macaqueRule_feature_keep_mask.csv",
        index=False,
    )

    groups = mon["id"].values
    n_groups = pd.Series(groups).nunique()
    n_splits = int(min(reverse_cfg.n_splits, n_groups))
    if n_splits < 2:
        raise RuntimeError(f"[{region}-{metric}] Not enough distinct monkeys for GroupKFold.")
    cv = GroupKFold(n_splits=n_splits)

    best_alpha = None
    best_score = -np.inf
    for a in reverse_cfg.ridge_alphas:
        scores = []
        for tr, te in cv.split(X_mon_kept, y_mon, groups=groups):
            Xtr = X_mon_kept[tr]
            Xte = X_mon_kept[te]
            if reverse_cfg.clip_features_in_monkey_train:
                lo, hi = clip_bounds_from_train(Xtr, reverse_cfg.clip_lo_q, reverse_cfg.clip_hi_q)
                Xtr = clip_X(Xtr, lo, hi)
                Xte = clip_X(Xte, lo, hi)
            model = Pipeline([
                ("scaler", make_scaler(reverse_cfg)),
                ("ridge", Ridge(alpha=float(a), fit_intercept=True)),
            ])
            model.fit(Xtr, y_mon[tr])
            pred = model.predict(Xte)
            scores.append(float(r2_score(y_mon[te], pred)))
        s = float(np.nanmean(scores)) if len(scores) > 0 else -np.inf
        if s > best_score:
            best_score = s
            best_alpha = float(a)

    oof_pred = np.full_like(y_mon, np.nan, dtype=float)
    oof_fold = np.full(len(y_mon), -1, dtype=int)
    for fold_index, (tr, te) in enumerate(
        cv.split(X_mon_kept, y_mon, groups=groups),
        start=1,
    ):
        Xtr = X_mon_kept[tr]
        Xte = X_mon_kept[te]
        if reverse_cfg.clip_features_in_monkey_train:
            lo, hi = clip_bounds_from_train(Xtr, reverse_cfg.clip_lo_q, reverse_cfg.clip_hi_q)
            Xtr = clip_X(Xtr, lo, hi)
            Xte = clip_X(Xte, lo, hi)
        m = Pipeline([
            ("scaler", make_scaler(reverse_cfg)),
            ("ridge", Ridge(alpha=float(best_alpha), fit_intercept=True)),
        ])
        m.fit(Xtr, y_mon[tr])
        oof_pred[te] = m.predict(Xte)
        oof_fold[te] = fold_index

    oof_valid = np.isfinite(y_mon) & np.isfinite(oof_pred) & (oof_fold > 0)
    if int(oof_valid.sum()) != len(y_mon):
        raise RuntimeError(
            f"[{region}-{metric}] Incomplete macaque OOF predictions: "
            f"{int(oof_valid.sum())}/{len(y_mon)}."
        )
    pooled_oof_r2 = float(r2_score(y_mon[oof_valid], oof_pred[oof_valid]))
    pooled_oof_pearson_r = float(
        pearsonr(y_mon[oof_valid], oof_pred[oof_valid])[0]
    )
    pooled_oof_spearman_r = float(
        spearmanr(y_mon[oof_valid], oof_pred[oof_valid])[0]
    )
    pd.DataFrame(
        {
            "id": mon.loc[oof_valid, "id"].astype(str).to_numpy(),
            "region": region,
            "metric": metric,
            "fold": oof_fold[oof_valid],
            "observed_r23_resid": y_mon[oof_valid],
            "predicted_r23_resid": oof_pred[oof_valid],
            "error_r23_resid": y_mon[oof_valid] - oof_pred[oof_valid],
        }
    ).to_csv(
        hs_dir / f"{out_prefix}_macaque_rule_oof_predictions.csv",
        index=False,
    )

    err_center, err_scale, err_scale_method = robust_center_scale(
        y_mon - oof_pred,
        reverse_cfg=reverse_cfg,
        y_ref=y_mon,
    )

    X_mon_final = X_mon_kept.copy()
    if reverse_cfg.clip_features_in_monkey_train:
        lo_full, hi_full = clip_bounds_from_train(X_mon_final, reverse_cfg.clip_lo_q, reverse_cfg.clip_hi_q)
        X_mon_final = clip_X(X_mon_final, lo_full, hi_full)

    macaque_model = Pipeline([
        ("scaler", make_scaler(reverse_cfg)),
        ("ridge", Ridge(alpha=float(best_alpha), fit_intercept=True)),
    ])
    macaque_model.fit(X_mon_final, y_mon)

    meta = {
        "region": region,
        "metric": metric,
        "variant": method_cfg.variant_label,
        "best_alpha": float(best_alpha),
        "best_cv_r2_monkey_resid": float(best_score),
        "pooled_oof_r2_monkey_resid": pooled_oof_r2,
        "pooled_oof_pearson_r_monkey_resid": pooled_oof_pearson_r,
        "pooled_oof_spearman_r_monkey_resid": pooled_oof_spearman_r,
        "n_oof_monkey": int(oof_valid.sum()),
        "err_center": float(err_center),
        "err_scale": float(err_scale),
        "err_scale_method": str(err_scale_method),
        **age_meta,
    }
    pd.Series(meta).to_csv(hs_dir / f"{out_prefix}_macaque_rule_model_meta.csv")
    joblib.dump(macaque_model, hs_dir / f"{out_prefix}_macaque_rule_model.joblib")


    hum = _ensure_conf_cols(human_triples.copy(), method_cfg.conf_cols, method_cfg.missing_token)
    if "id" not in hum.columns and "subject_id" in hum.columns:
        hum = hum.rename(columns={"subject_id": "id"})

    hum["id"] = hum["id"].astype(str)
    hum["age_spline_months"] = detect_and_convert_age_to_months(hum["age_T2"]).astype(float)
    hum[ratio_col] = safe_log_ratio(
        pd.to_numeric(hum[base_t2_col], errors="coerce"),
        pd.to_numeric(hum[base_t1_col], errors="coerce"),
        eps=method_cfg.ratio_eps,
    )

    req_h = [
        "id", "scan_1", "scan_2", "scan_3",
        "age_1", "age_2", "age_3", "age_T2",
        "dt12_months_used", "dt23_months_used",
        r12_raw_col, r23_raw_col, base_t1_col, base_t2_col,
    ] + list(method_cfg.conf_cols)
    if include_c_ratio:
        req_h.append(ratio_col)
    hum = hum.dropna(subset=req_h).copy()

    if reverse_cfg.filter_extreme_rates:
        hum = filter_extreme_rows_for_metric(
            hum,
            r12_raw_col,
            r23_raw_col,
            reverse_cfg.rate_abs_max,
            reverse_cfg.dt_used_min,
        )

    hum, _ = add_age_spline_with_spec(hum, "age_spline_months", spline_spec)
    conf_hum = hum[list(method_cfg.conf_cols)].copy()
    r12_h = pd.to_numeric(hum[r12_raw_col], errors="coerce").values.astype(float)
    r23_h = pd.to_numeric(hum[r23_raw_col], errors="coerce").values.astype(float)

    hum[f"{r12_raw_col}_resid"] = r12_h - human_rr.hat_prev(conf_hum)
    hum[f"{r23_raw_col}_resid"] = r23_h - human_rr.hat_next(conf_hum)

    baseline_col_used_h = base_t2_col
    if include_baseline_t2 and residualize_baseline_t2 and human_br is not None:
        hum[f"{base_t2_col}_resid"] = human_br.transform(
            conf_hum,
            pd.to_numeric(hum[base_t2_col], errors="coerce"),
        )
        baseline_col_used_h = f"{base_t2_col}_resid"

    X_h_full, names_h_full = build_design_matrix_rate12_to_rate23(
        hum,
        r12_col=f"{r12_raw_col}_resid",
        spline_cols=spline_cols,
        include_baseline_t2=include_baseline_t2,
        baseline_t2_col=baseline_col_used_h,
        include_c_ratio=include_c_ratio,
        ratio_col=ratio_col if include_c_ratio else None,
        use_interactions=True,
    )
    keep_mask_use = load_feature_mask_csv(
        hs_dir / f"{out_prefix}_macaqueRule_feature_keep_mask.csv",
        expected_names=names_h_full,
    )
    X_h_kept, _ = apply_keep_mask(X_h_full, names_h_full, keep_mask_use)

    ood_df = compute_ood_metrics_from_monkey_train(X_h_kept, X_mon_kept, reverse_cfg)

    pred_h_res_no_clip = macaque_model.predict(X_h_kept)
    pred_h_res_clip = np.full_like(pred_h_res_no_clip, np.nan, dtype=float)
    if reverse_cfg.clip_features_in_human_infer:
        lo_full2, hi_full2 = clip_bounds_from_train(X_mon_kept, reverse_cfg.clip_lo_q, reverse_cfg.clip_hi_q)
        pred_h_res_clip = macaque_model.predict(clip_X(X_h_kept, lo_full2, hi_full2))

    hs_dev_res_no_clip = hum[f"{r23_raw_col}_resid"].values.astype(float) - pred_h_res_no_clip
    hs_dev_z_no_clip = (hs_dev_res_no_clip - err_center) / (err_scale + 1e-12)

    hs_dev_res_clip = np.full_like(hs_dev_res_no_clip, np.nan, dtype=float)
    hs_dev_z_clip = np.full_like(hs_dev_z_no_clip, np.nan, dtype=float)
    if reverse_cfg.clip_features_in_human_infer:
        hs_dev_res_clip = hum[f"{r23_raw_col}_resid"].values.astype(float) - pred_h_res_clip
        hs_dev_z_clip = (hs_dev_res_clip - err_center) / (err_scale + 1e-12)

    human_long = pd.DataFrame(
        {
            "id": hum["id"].values,
            "scan_1": hum["scan_1"].astype(str).values,
            "scan_2": hum["scan_2"].astype(str).values,
            "scan_3": hum["scan_3"].astype(str).values,
            "age_1": pd.to_numeric(hum["age_1"], errors="coerce").values,
            "age_2": pd.to_numeric(hum["age_2"], errors="coerce").values,
            "age_3": pd.to_numeric(hum["age_3"], errors="coerce").values,
            "age_T2": pd.to_numeric(hum["age_T2"], errors="coerce").values,
            "age_T2_months": pd.to_numeric(hum["age_spline_months"], errors="coerce").values,
            "dt23_months_used": pd.to_numeric(hum["dt23_months_used"], errors="coerce").values,
            "region": region,
            "metric": metric,
            "r23_raw": r23_h,
            "r23_hat_conf": human_rr.hat_next(conf_hum),
            "r23_resid": hum[f"{r23_raw_col}_resid"].values.astype(float),
            "pred_r23_resid_macaque_no_clip": pred_h_res_no_clip,
            "HS_dev_resid_no_clip": hs_dev_res_no_clip,
            "HS_dev_z_no_clip": hs_dev_z_no_clip,
            "pred_r23_resid_macaque_clip": pred_h_res_clip,
            "HS_dev_resid_clip": hs_dev_res_clip,
            "HS_dev_z_clip": hs_dev_z_clip,
            "HS_dev_abs_resid_no_clip": np.abs(hs_dev_res_no_clip),
            "HS_dev_abs_z_no_clip": np.abs(hs_dev_z_no_clip),
            "base_T1": pd.to_numeric(hum[base_t1_col], errors="coerce").values,
            "base_T2": pd.to_numeric(hum[base_t2_col], errors="coerce").values,
        }
    )

    for c in method_cfg.conf_cols:
        human_long[c] = hum[c].astype(str).values

    human_long = pd.concat([human_long.reset_index(drop=True), ood_df.reset_index(drop=True)], axis=1)

    oof_path = predict_out_dir / f"{out_prefix}_human_oof_predictions.csv"
    if reverse_cfg.add_human_rule_deviation and oof_path.exists():
        try:
            oof2 = _rename_oof_cols_for_merge(pd.read_csv(oof_path))
            oof2[["id", "scan_1", "scan_2", "scan_3"]] = oof2[["id", "scan_1", "scan_2", "scan_3"]].astype(str)
            merged = human_long.merge(oof2, on=["id", "scan_1", "scan_2", "scan_3"], how="left")
            hd_dev = pd.to_numeric(merged["r23_resid_humanOOF"], errors="coerce") - pd.to_numeric(
                merged["pred_r23_resid_humanOOF"], errors="coerce"
            )
            cen_h, sc_h, meth_h = robust_center_scale(
                hd_dev.values.astype(float),
                reverse_cfg=reverse_cfg,
                y_ref=hd_dev.values.astype(float),
            )
            merged["HD_dev_resid"] = hd_dev.values.astype(float)
            merged["HD_dev_z"] = (merged["HD_dev_resid"] - cen_h) / (sc_h + 1e-12)
            merged["HD_dev_scale_method"] = str(meth_h)
            human_long = merged
        except Exception as e:
            human_long["HD_dev_resid"] = np.nan
            human_long["HD_dev_z"] = np.nan
            human_long["HD_dev_scale_method"] = f"oof_merge_failed:{type(e).__name__}"

    human_long.to_csv(hs_dir / f"{out_prefix}_human_specificity_long.csv", index=False)
    return human_long, meta

def make_hsdi_tables(hs_long_all: pd.DataFrame, value_col: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = hs_long_all.copy()

    key_cols = [
        "id",
        "scan_1",
        "scan_2",
        "scan_3",
        "age_1",
        "age_2",
        "age_3",
        "age_T2",
    ]
    for key_col in key_cols:
        if key_col not in df.columns:
            raise KeyError(
                f"Missing key column required for HSDI aggregation: {key_col}"
            )

    wide = df.pivot_table(
        index=key_cols,
        columns=["metric", "region"],
        values=value_col,
        aggfunc="mean",
    ).reset_index()
    wide = _flatten_pivot_columns(wide, key_cols)

    for m in METRICS:
        for r in REGIONS:
            col = f"HSdevZ_{m}_{r}"
            if col not in wide.columns:
                wide[col] = np.nan

    cell_cols = [f"HSdevZ_{m}_{r}" for m in METRICS for r in REGIONS]
    wide["HSDI_global"] = wide[cell_cols].mean(axis=1, skipna=True)
    wide["HSDI_contrast_CT"] = (
        wide["HSdevZ_CT_high_expansion"] - wide["HSdevZ_CT_low_expansion"]
    )
    wide["HSDI_contrast_SA"] = (
        wide["HSdevZ_SA_high_expansion"] - wide["HSdevZ_SA_low_expansion"]
    )
    wide["HSDI_contrast_mean"] = wide[["HSDI_contrast_CT", "HSDI_contrast_SA"]].mean(axis=1, skipna=True)
    wide["HSDI_global_abs"] = np.abs(wide["HSDI_global"].astype(float))
    wide["HSDI_contrast_mean_abs"] = np.abs(wide["HSDI_contrast_mean"].astype(float))

    for oc in ["OOD_frac_q", "OOD_max_abs_z", "OOD_n_feat_absz_ge_thresh"]:
        if oc not in df.columns:
            df[oc] = np.nan

    ood_agg = (
        df.groupby(key_cols, as_index=False)
        .agg(
            OOD_frac_q_mean=("OOD_frac_q", "mean"),
            OOD_frac_q_max=("OOD_frac_q", "max"),
            OOD_max_abs_z_mean=("OOD_max_abs_z", "mean"),
            OOD_max_abs_z_max=("OOD_max_abs_z", "max"),
            OOD_n_feat_absz_ge_thresh_mean=("OOD_n_feat_absz_ge_thresh", "mean"),
            OOD_n_feat_absz_ge_thresh_max=("OOD_n_feat_absz_ge_thresh", "max"),
        )
    )
    wide = wide.merge(ood_agg, on=key_cols, how="left")

    subj_mean = wide.groupby("id", as_index=False)[
        cell_cols
        + [
            "HSDI_global",
            "HSDI_contrast_CT",
            "HSDI_contrast_SA",
            "HSDI_contrast_mean",
            "HSDI_global_abs",
            "HSDI_contrast_mean_abs",
            "OOD_frac_q_mean",
            "OOD_frac_q_max",
            "OOD_max_abs_z_mean",
            "OOD_max_abs_z_max",
            "OOD_n_feat_absz_ge_thresh_mean",
            "OOD_n_feat_absz_ge_thresh_max",
        ]
    ].mean()

    subj_maxabs = wide.copy()
    subj_maxabs["HSDI_global_abs"] = np.abs(subj_maxabs["HSDI_global"].astype(float))
    subj_maxabs["HSDI_contrast_mean_abs"] = np.abs(subj_maxabs["HSDI_contrast_mean"].astype(float))
    subj_maxabs = subj_maxabs.groupby("id", as_index=False).agg(
        {
            "HSDI_global_abs": "max",
            "HSDI_contrast_mean_abs": "max",
            "OOD_frac_q_max": "max",
            "OOD_max_abs_z_max": "max",
            "OOD_n_feat_absz_ge_thresh_max": "max",
        }
    )
    return wide, subj_mean, subj_maxabs

def validate_step1_inputs(method_cfg: PredictMethodConfig) -> None:
    required = [HUMAN_TRIPLES_CSV, MONKEY_TRIPLES_CSV, SPLINE_SPEC_CSV]

    for region in REGIONS:
        for metric in METRICS:
            out_prefix = f"R12toR23_{region}_{metric}_{method_cfg.file_tag}"
            required.extend(
                [
                    FINALFIG_DIR / f"{out_prefix}_human_rate_residualizer.joblib",
                    FINALFIG_DIR / f"{out_prefix}_human_ref_confounds.csv",
                ]
            )

    missing = [p for p in required if not p.exists()]
    if missing:
        missing_text = "\n".join(f"  - {p}" for p in missing)
        raise FileNotFoundError(
            "Missing required Figure 6 source inputs:\n" + missing_text
        )

    spline_spec = _read_spline_spec(SPLINE_SPEC_CSV)
    include_baseline_t2 = bool(int(spline_spec["include_baseline_t2"]))
    residualize_baseline_t2 = bool(int(spline_spec["residualize_baseline_t2"]))

    if include_baseline_t2 and residualize_baseline_t2:
        baseline_required = []
        for region in REGIONS:
            for metric in METRICS:
                out_prefix = f"R12toR23_{region}_{metric}_{method_cfg.file_tag}"
                baseline_required.append(
                    FINALFIG_DIR / f"{out_prefix}_human_baseline_residualizer.joblib"
                )
        missing_baseline = [p for p in baseline_required if not p.exists()]
        if missing_baseline:
            missing_text = "\n".join(f"  - {p}" for p in missing_baseline)
            raise FileNotFoundError(
                "spline_spec.csv requests baseline T2 residualization, but the "
                "following baseline residualizers are missing:\n" + missing_text
            )

    oof_required = []
    for region in REGIONS:
        for metric in METRICS:
            out_prefix = f"R12toR23_{region}_{metric}_{method_cfg.file_tag}"
            oof_required.append(
                FINALFIG_DIR / f"{out_prefix}_human_oof_predictions.csv"
            )

    missing_oof = [p for p in oof_required if not p.exists()]
    if missing_oof:
        missing_text = "\n".join(f"  - {p}" for p in missing_oof)
        raise FileNotFoundError(
            "Missing human OOF prediction tables required for the complete "
            "Figure 6 rerun:\n" + missing_text
        )

    expected_variant = {
        "include_baseline_t2": 1,
        "residualize_baseline_t2": 1,
        "include_c_ratio": 0,
        "use_zero_intercept_spline": 0,
    }
    mismatches = {
        key: (int(spline_spec[key]), expected)
        for key, expected in expected_variant.items()
        if int(spline_spec[key]) != int(expected)
    }
    if mismatches:
        raise ValueError(
            "spline_spec.csv does not match the expected model variant. "
            f"Observed/expected mismatches: {mismatches}"
        )

def main() -> None:
    if METHOD_NAME not in METHODS:
        raise ValueError(f"Unknown METHOD_NAME: {METHOD_NAME}")

    method_cfg = METHODS[METHOD_NAME]

    validate_step1_inputs(method_cfg)
    _read_spline_spec(SPLINE_SPEC_CSV)


    human_triples = pd.read_csv(HUMAN_TRIPLES_CSV)
    monkey_triples = pd.read_csv(MONKEY_TRIPLES_CSV)
    human_triples.columns = [str(c).strip() for c in human_triples.columns]
    monkey_triples.columns = [str(c).strip() for c in monkey_triples.columns]


    monkey_triples["age_3"] = pd.to_numeric(
        monkey_triples["age_3"],
        errors="coerce",
    )

    monkey_triples = monkey_triples.loc[
        monkey_triples["age_3"].le(65.0)
    ].copy()


    if "sex" in human_triples.columns:
        human_triples["sex"] = normalize_sex_codes(human_triples["sex"], dataset="human")
    if "sex" in monkey_triples.columns:
        monkey_triples["sex"] = normalize_sex_codes(monkey_triples["sex"], dataset="macaque")

    hs_long_list = []
    meta_list = []
    for region in REGIONS:
        for metric in METRICS:
            hs_long, meta = fit_macaque_rule_and_score_humans(
                human_triples=human_triples,
                monkey_triples=monkey_triples,
                region=region,
                metric=metric,
                predict_out_dir=PREDICT_DIR,
                hs_dir=HS_DIR,
                method_cfg=method_cfg,
                reverse_cfg=REVERSE_CFG,
            )
            hs_long_list.append(hs_long)
            meta_list.append(meta)

    hs_long_all = pd.concat(hs_long_list, ignore_index=True)
    hs_long_all.to_csv(HS_DIR / "ALL_human_specificity_long.csv", index=False)
    pd.DataFrame(meta_list).to_csv(HS_DIR / "ALL_macaque_rule_model_meta.csv", index=False)

    wide_by_triple, subj_mean, subj_maxabs = make_hsdi_tables(hs_long_all, "HS_dev_z_no_clip")
    wide_by_triple.to_csv(HS_DIR / "HSDI_wide_byTriple_noClip.csv", index=False)
    subj_mean.to_csv(HS_DIR / "HSDI_subjectMean_noClip.csv", index=False)
    subj_maxabs.to_csv(HS_DIR / "HSDI_subjectMaxAbs_noClip.csv", index=False)

    save_json(
        {
            "method_cfg": asdict(method_cfg),
            "reverse_cfg": asdict(REVERSE_CFG),
            "predict_dir": str(PREDICT_DIR),
            "hs_dir": str(HS_DIR),
        },
        HS_DIR / "step4_newstyle_config_snapshot.json",
    )

    with open(HS_DIR / "README_step4_newstyle.txt", "w", encoding="utf-8") as f:
        f.write("Human-specific deviation analysis.\n")
        f.write(f"METHOD_NAME = {METHOD_NAME}\n")
        f.write(f"PREDICT_DIR = {str(PREDICT_DIR)}\n")
        f.write("Initial macaque age-match buffer is 6 months.\n")
        f.write("All spline basis columns enter every enabled interaction block.\n")
        f.write("HSDI_wide_byTriple_noClip uses complete scan1-scan2-scan3 triple keys.\n")
        f.write("Low/high-expansion and noCratio definitions are retained.\n")

if __name__ == "__main__":
    main()
