"""Inputs: CROSS_SPECIES_FEATURE_DIR/*.csv. Outputs: CROSS_SPECIES_OUTPUT_DIR/prediction_outputs/."""

from __future__ import annotations

import os
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = Path(os.environ.get("CROSS_SPECIES_OUTPUT_DIR", str(PACKAGE_ROOT / "outputs")))
FEATURE_DIR = Path(os.environ.get("CROSS_SPECIES_FEATURE_DIR", str(PACKAGE_ROOT / "data" / "private" / "features")))
PREDICTION_OUT_DIR = OUTPUT_ROOT / "prediction_outputs"
PREDICTION_OUT_DIR.mkdir(parents=True, exist_ok=True)

import numpy as np

import pandas as pd


from typing import Tuple, List, Dict, Any, Optional

from patsy import dmatrix

from scipy.stats import spearmanr, pearsonr

from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.linear_model import ElasticNet, Ridge

from sklearn.pipeline import Pipeline

from sklearn.model_selection import GroupKFold

from sklearn.metrics import r2_score

import joblib

BASE_DIR = FEATURE_DIR

RUN_NAME = "fig5_spc_transition_rerun_20260723_baselineT2_noCratio"

RUN_DIR = BASE_DIR / RUN_NAME

PREDICTION_OUT_DIR = OUTPUT_ROOT / 'prediction_outputs'

HUMAN_CSV = BASE_DIR / "allSubjects_human_features_expansion.csv"

MONKEY_CSV = BASE_DIR / "allSubjects_macaque_features_expansion_completed.csv"

AGE_RATIO = 2.83

MONKEY_RATE_DT_SCALE = AGE_RATIO

EXPANSION_GROUPS = ("low_expansion", "high_expansion")

METRICS = ["CT", "SA"]

INCLUDE_BASELINE_T2 = True

RESIDUALIZE_BASELINE_T2 = True

INCLUDE_C_RATIO = False

MODEL_VARIANT = "baselineT2_Cratio" if INCLUDE_C_RATIO else "baselineT2_noCratio"

FILE_TAG = "Cratio" if INCLUDE_C_RATIO else "noCratio"

RATIO_EPS = 1e-6

CONF_COLS = ["sex", "site", "scanner_manufacturer", "scanner_model"]

MISSING_TOKEN = "__MISSING__"

SEX_CANONICAL_MALE = "male"

SEX_CANONICAL_FEMALE = "female"

MONKEY_ADJUST_SEX = True

MONKEY_USE_HUMAN_REF_FOR_SCANNER_SITE = True

SPLINE_DEGREE = 3

N_SPLINE_BASIS = 5

N_INTERNAL_KNOTS = max(0, N_SPLINE_BASIS - SPLINE_DEGREE - 1)

BOUND_EPS = 1e-6

BOUND_BUF = 1e-3

EPS = 1e-8

N_SPLITS = 5

ALPHAS = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]

L1_RATIOS = [0.0, 0.1, 0.3, 0.5]

VAR_STD_MIN = 1e-3

CONST_STD_MIN = 1e-12

CLIP_FEATURES_TO_HUMAN_QUANTILES = True

CLIP_LO_Q = 0.001

CLIP_HI_Q = 0.999

FILTER_EXTREME_RATES = True

RATE_ABS_MAX = 5.0

DT_USED_MIN = 1.0

RIDGE_ALPHA_RATE = 0.2

RIDGE_ALPHA_BASELINE = 1.0

ENABLE_HAT_CLIP = True

HAT_CLIP_Q_LO = 0.001

HAT_CLIP_Q_HI = 0.999

HAT_CLIP_FALLBACK_ABS = 100.0

ENABLE_MONKEY_CALIBRATION = True

CALIBRATION_N_SPLITS = 3

CALIBRATE_SPACE = "raw"

def safe_corr(x: np.ndarray, y: np.ndarray, method: str) -> float:

    x = np.asarray(x, float)
    y = np.asarray(y, float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return np.nan
    xv = x[mask]
    yv = y[mask]
    if np.nanstd(xv) < 1e-12 or np.nanstd(yv) < 1e-12:
        return np.nan
    if method == "pearson":
        return float(pearsonr(xv, yv)[0])
    if method == "spearman":
        res = spearmanr(xv, yv)
        if hasattr(res, "correlation"):
            return float(res.correlation)
        return float(res[0])
    raise ValueError("method must be 'pearson' or 'spearman'")

def detect_and_convert_age_to_months(series: pd.Series) -> pd.Series:

    vals = pd.to_numeric(series, errors="coerce").astype(float)
    med = vals.dropna().median()
    if np.isnan(med):
        return vals
    if med < 30:
        return vals * 12.0
    return vals

def spc_rate(x1: np.ndarray, x2: np.ndarray, dt_months: np.ndarray) -> np.ndarray:

    x1 = np.asarray(x1, float)
    x2 = np.asarray(x2, float)
    dt = np.asarray(dt_months, float)
    return (200.0 * (x2 - x1) / (x1 + x2 + EPS)) / (dt + EPS)

def _effective_n_splits(n_splits: int, groups: np.ndarray) -> int:
    n_groups = pd.Series(groups).nunique()
    return int(min(n_splits, n_groups))

def mode_or_missing(s: pd.Series) -> str:

    s = s.dropna().astype(str)
    if s.empty:
        return MISSING_TOKEN
    vc = s.value_counts()
    if vc.empty:
        return MISSING_TOKEN
    return str(vc.index[0])

def safe_log_ratio(num: np.ndarray, den: np.ndarray, eps: float = 1e-6) -> np.ndarray:

    num = np.asarray(num, float)
    den = np.asarray(den, float)
    return np.log((num + eps) / (den + eps))

def normalize_expansion_group(series: pd.Series) -> pd.Series:

    normalized = (
        series.fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
        .str.replace("-", "_", regex=False)
        .str.replace(" ", "_", regex=False)
    )
    aliases = {
        "NEG": "low_expansion",
        "NEGATIVE": "low_expansion",
        "LOW": "low_expansion",
        "LOW_EXPANSION": "low_expansion",
        "POS": "high_expansion",
        "POSITIVE": "high_expansion",
        "HIGH": "high_expansion",
        "HIGH_EXPANSION": "high_expansion",
    }
    return normalized.map(aliases)

def normalize_sex_codes(series: pd.Series, dataset: str) -> pd.Series:


    ds = str(dataset).strip().lower()
    if ds not in {"human", "macaque", "monkey"}:
        raise ValueError("dataset must be 'human' or 'macaque'")

    if ds == "human":
        code_map = {"1": SEX_CANONICAL_MALE, "2": SEX_CANONICAL_FEMALE}
    else:
        code_map = {"0": SEX_CANONICAL_FEMALE, "1": SEX_CANONICAL_MALE}

    alias_map = {
        "m": SEX_CANONICAL_MALE,
        "male": SEX_CANONICAL_MALE,
        "man": SEX_CANONICAL_MALE,
        "boy": SEX_CANONICAL_MALE,
        "f": SEX_CANONICAL_FEMALE,
        "female": SEX_CANONICAL_FEMALE,
        "woman": SEX_CANONICAL_FEMALE,
        "girl": SEX_CANONICAL_FEMALE,
    }

    out = series.copy()

    def _norm_one(x: Any) -> str:
        if pd.isna(x):
            return MISSING_TOKEN
        s = str(x).strip()
        if s == "":
            return MISSING_TOKEN
        sl = s.lower()
        if sl in alias_map:
            return alias_map[sl]
        if s in code_map:
            return code_map[s]
        if sl in code_map:
            return code_map[sl]
        try:
            fv = float(s)
            if np.isfinite(fv) and float(int(fv)) == fv:
                key = str(int(fv))
                if key in code_map:
                    return code_map[key]
        except Exception:
            pass
        return sl

    return out.map(_norm_one).astype(str)

def filter_extreme_rows_for_metric(
    df: pd.DataFrame,
    r12_col: str,
    r23_col: str,
    dt12_col: str = "dt12_months_used",
    dt23_col: str = "dt23_months_used",
    rate_abs_max: float = RATE_ABS_MAX,
    dt_used_min: float = DT_USED_MIN,
) -> pd.DataFrame:

    out = df.copy()
    r12 = pd.to_numeric(out[r12_col], errors="coerce").astype(float)
    r23 = pd.to_numeric(out[r23_col], errors="coerce").astype(float)
    dt12 = pd.to_numeric(out[dt12_col], errors="coerce").astype(float)
    dt23 = pd.to_numeric(out[dt23_col], errors="coerce").astype(float)

    ok = np.isfinite(r12) & np.isfinite(r23) & np.isfinite(dt12) & np.isfinite(dt23)
    ok &= (np.abs(r12) <= float(rate_abs_max)) & (np.abs(r23) <= float(rate_abs_max))
    ok &= (dt12 >= float(dt_used_min)) & (dt23 >= float(dt_used_min))
    return out.loc[ok].reset_index(drop=True)

def compute_feature_keep_mask_from_Xtr(
    Xtr: np.ndarray,
    names: List[str],
    var_std_min: float = VAR_STD_MIN,
    const_std_min: float = CONST_STD_MIN,
) -> Tuple[np.ndarray, np.ndarray]:


    Xtr = np.asarray(Xtr, float)
    std_vec = np.nanstd(Xtr, axis=0)

    keep = np.ones(len(names), dtype=bool)
    keep &= np.isfinite(std_vec) & (std_vec >= const_std_min)

    is_interaction = np.array([("*" in nm) for nm in names], dtype=bool)
    keep &= (~is_interaction) | (std_vec >= var_std_min)

    if keep.sum() < 1:
        raise RuntimeError(
            f"All features were removed by variance filtering. "
            f"Try lowering VAR_STD_MIN={var_std_min:g}."
        )

    return keep, std_vec

def apply_keep_mask(X: np.ndarray, names: List[str], keep: np.ndarray) -> Tuple[np.ndarray, List[str]]:

    X = np.asarray(X, float)
    keep = np.asarray(keep, bool)
    if X.shape[1] != keep.size:
        raise ValueError(f"Mask length {keep.size} does not match X columns {X.shape[1]}.")
    new_names = [nm for nm, k in zip(names, keep) if k]
    return X[:, keep], new_names

def save_feature_mask_csv(out_path: Path, names: List[str], keep: np.ndarray, std_vec: Optional[np.ndarray] = None) -> None:

    keep = np.asarray(keep, bool)
    df = pd.DataFrame({"name": names, "keep": keep.astype(int)})
    if std_vec is not None:
        df["std"] = np.asarray(std_vec, float)
    df.to_csv(out_path, index=False)

def load_feature_mask_csv(mask_path: Path, expected_names: List[str]) -> np.ndarray:

    df = pd.read_csv(mask_path)
    if "name" not in df.columns or "keep" not in df.columns:
        raise ValueError(f"Invalid mask CSV: {mask_path}")

    keep_map = {str(nm): int(k) for nm, k in zip(df["name"].astype(str), df["keep"].astype(int))}

    keep = []
    missing = []
    for nm in expected_names:
        if nm not in keep_map:
            missing.append(nm)
            keep.append(0)
        else:
            keep.append(1 if keep_map[nm] == 1 else 0)

    if missing:
        raise RuntimeError(
            f"Feature names in mask file do not match current design matrix. "
            f"Missing {len(missing)} names, e.g. {missing[:5]}"
        )

    keep = np.asarray(keep, dtype=bool)
    if keep.sum() < 1:
        raise RuntimeError("Loaded feature mask keeps zero features.")
    return keep

def save_feature_clip_stats_csv(
    out_path: Path,
    kept_names: List[str],
    X_kept_human: np.ndarray,
    q_lo: float = CLIP_LO_Q,
    q_hi: float = CLIP_HI_Q,
) -> None:

    X = np.asarray(X_kept_human, float)
    rows = []
    for j, nm in enumerate(kept_names):
        col = X[:, j]
        col = col[np.isfinite(col)]
        if col.size == 0:
            lo = np.nan
            hi = np.nan
        else:
            lo = float(np.quantile(col, q_lo))
            hi = float(np.quantile(col, q_hi))
            if hi < lo:
                lo, hi = hi, lo
            if np.isclose(hi, lo):
                hi = lo + 1e-12
        rows.append({"name": nm, "q_lo": lo, "q_hi": hi})
    pd.DataFrame(rows).to_csv(out_path, index=False)

def load_feature_clip_stats_csv(stats_path: Path, expected_names: List[str]) -> Tuple[np.ndarray, np.ndarray]:

    df = pd.read_csv(stats_path)
    if not {"name", "q_lo", "q_hi"}.issubset(df.columns):
        raise ValueError(f"Invalid clip-stats CSV: {stats_path}")

    mp_lo = {str(nm): float(v) for nm, v in zip(df["name"].astype(str), df["q_lo"].astype(float))}
    mp_hi = {str(nm): float(v) for nm, v in zip(df["name"].astype(str), df["q_hi"].astype(float))}

    lo = []
    hi = []
    missing = []
    for nm in expected_names:
        if nm not in mp_lo or nm not in mp_hi:
            missing.append(nm)
            lo.append(np.nan)
            hi.append(np.nan)
        else:
            lo.append(mp_lo[nm])
            hi.append(mp_hi[nm])

    if missing:
        raise RuntimeError(
            f"Clip-stats names do not match current kept feature names. "
            f"Missing {len(missing)} names, e.g. {missing[:5]}"
        )

    return np.asarray(lo, float), np.asarray(hi, float)

def clip_X_by_bounds(X: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:

    X = np.asarray(X, float)
    lo = np.asarray(lo, float).reshape(1, -1)
    hi = np.asarray(hi, float).reshape(1, -1)
    return np.minimum(np.maximum(X, lo), hi)

def make_onehot_encoder(drop_first: bool = True) -> OneHotEncoder:


    drop = "first" if drop_first else None
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop=drop)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False, drop=drop)

def _save_human_hat_stats(
    out_prefix: str,
    rr: "RateResidualizer",
    conf_df: pd.DataFrame,
    q_lo: float,
    q_hi: float,
) -> None:

    r12_hat = rr.hat_prev(conf_df)
    r23_hat = rr.hat_next(conf_df)
    stats = {
        "r12_hat_mean": float(np.mean(r12_hat)),
        "r12_hat_std": float(np.std(r12_hat) + 1e-12),
        "r23_hat_mean": float(np.mean(r23_hat)),
        "r23_hat_std": float(np.std(r23_hat) + 1e-12),
        "r12_hat_p_lo": float(np.quantile(r12_hat, q_lo)),
        "r12_hat_p_hi": float(np.quantile(r12_hat, q_hi)),
        "r23_hat_p_lo": float(np.quantile(r23_hat, q_lo)),
        "r23_hat_p_hi": float(np.quantile(r23_hat, q_hi)),
        "q_lo": float(q_lo),
        "q_hi": float(q_hi),
    }
    pd.Series(stats).to_csv(PREDICTION_OUT_DIR / f"{out_prefix}_human_hat_stats.csv")

def _clip_hat_with_human_stats(
    out_prefix: str,
    r12_hat: np.ndarray,
    r23_hat: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:


    meta: Dict[str, float] = {}
    f = PREDICTION_OUT_DIR / f"{out_prefix}_human_hat_stats.csv"

    if ENABLE_HAT_CLIP and f.exists():
        hs = pd.read_csv(f, header=None, index_col=0).iloc[:, 0].to_dict()

        r12_lo = float(hs.get("r12_hat_p_lo", -HAT_CLIP_FALLBACK_ABS))
        r12_hi = float(hs.get("r12_hat_p_hi", HAT_CLIP_FALLBACK_ABS))
        r23_lo = float(hs.get("r23_hat_p_lo", -HAT_CLIP_FALLBACK_ABS))
        r23_hi = float(hs.get("r23_hat_p_hi", HAT_CLIP_FALLBACK_ABS))

        meta.update({
            "clip_used": 1.0,
            "r12_lo": r12_lo,
            "r12_hi": r12_hi,
            "r23_lo": r23_lo,
            "r23_hi": r23_hi,
        })
        return (
            np.clip(r12_hat, r12_lo, r12_hi),
            np.clip(r23_hat, r23_lo, r23_hi),
            meta,
        )

    meta.update({
        "clip_used": 1.0 if ENABLE_HAT_CLIP else 0.0,
        "r12_lo": -HAT_CLIP_FALLBACK_ABS,
        "r12_hi": HAT_CLIP_FALLBACK_ABS,
        "r23_lo": -HAT_CLIP_FALLBACK_ABS,
        "r23_hi": HAT_CLIP_FALLBACK_ABS,
    })
    return (
        np.clip(r12_hat, -HAT_CLIP_FALLBACK_ABS, HAT_CLIP_FALLBACK_ABS),
        np.clip(r23_hat, -HAT_CLIP_FALLBACK_ABS, HAT_CLIP_FALLBACK_ABS),
        meta,
    )

def _fit_linear_calibrator(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:

    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    ok = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[ok]
    y_pred = y_pred[ok]

    if y_true.size < 2:
        return (0.0, 1.0)

    X = np.column_stack([np.ones_like(y_pred), y_pred])
    coef, _, _, _ = np.linalg.lstsq(X, y_true, rcond=None)
    a = float(coef[0])
    b = float(coef[1])
    return a, b

def _oof_linear_calibration_groupkfold(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
    n_splits: int,
) -> Tuple[np.ndarray, Dict[str, float]]:


    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    groups = np.asarray(groups)

    ok = np.isfinite(y_true) & np.isfinite(y_pred) & pd.notna(groups)
    idx_all = np.where(ok)[0]
    if idx_all.size < 5:
        return np.full_like(y_pred, np.nan, dtype=float), {"cal_oof_done": 0.0}

    y_true2 = y_true[ok]
    y_pred2 = y_pred[ok]
    groups2 = groups[ok].astype(str)

    n_splits_eff = _effective_n_splits(n_splits, groups2)
    if n_splits_eff < 2:
        return np.full_like(y_pred, np.nan, dtype=float), {"cal_oof_done": 0.0}

    cv = GroupKFold(n_splits=n_splits_eff)
    cal_oof = np.full_like(y_pred2, np.nan, dtype=float)
    a_list: List[float] = []
    b_list: List[float] = []

    for tr, te in cv.split(y_pred2, y_true2, groups=groups2):
        a, b = _fit_linear_calibrator(y_true2[tr], y_pred2[tr])
        cal_oof[te] = a + b * y_pred2[te]
        a_list.append(a)
        b_list.append(b)

    out = np.full_like(y_pred, np.nan, dtype=float)
    out[idx_all] = cal_oof
    return out, {
        "cal_oof_done": 1.0,
        "a_mean": float(np.mean(a_list)) if a_list else np.nan,
        "b_mean": float(np.mean(b_list)) if b_list else np.nan,
        "n_splits_eff": float(n_splits_eff),
    }

class RateResidualizer:


    def __init__(self, conf_cols: List[str], ridge_alpha: float = 1.0):
        self.conf_cols = list(conf_cols)
        self.ridge_alpha = float(ridge_alpha)
        self.encoder: Optional[OneHotEncoder] = None
        self.lr_prev: Optional[Ridge] = None
        self.lr_next: Optional[Ridge] = None

    def _prep_conf(self, conf_df: pd.DataFrame) -> pd.DataFrame:
        out = conf_df.copy()
        for c in self.conf_cols:
            if c not in out.columns:
                out[c] = MISSING_TOKEN
            out[c] = out[c].astype(str).fillna(MISSING_TOKEN).str.strip()
        return out[self.conf_cols]

    def fit(self, conf_df: pd.DataFrame, r_prev: np.ndarray, r_next: np.ndarray) -> "RateResidualizer":
        conf_df = self._prep_conf(conf_df)

        self.encoder = make_onehot_encoder(drop_first=True)
        Z = self.encoder.fit_transform(conf_df)

        self.lr_prev = Ridge(alpha=self.ridge_alpha, fit_intercept=True)
        self.lr_next = Ridge(alpha=self.ridge_alpha, fit_intercept=True)

        self.lr_prev.fit(Z, r_prev.reshape(-1, 1))
        self.lr_next.fit(Z, r_next.reshape(-1, 1))
        return self

    def hat_prev(self, conf_df: pd.DataFrame) -> np.ndarray:
        if self.encoder is None or self.lr_prev is None:
            raise RuntimeError("Residualizer not fitted.")
        conf_df = self._prep_conf(conf_df)
        Z = self.encoder.transform(conf_df)
        return self.lr_prev.predict(Z).ravel()

    def hat_next(self, conf_df: pd.DataFrame) -> np.ndarray:
        if self.encoder is None or self.lr_next is None:
            raise RuntimeError("Residualizer not fitted.")
        conf_df = self._prep_conf(conf_df)
        Z = self.encoder.transform(conf_df)
        return self.lr_next.predict(Z).ravel()

    def transform_prev(self, conf_df: pd.DataFrame, r_prev: np.ndarray) -> np.ndarray:
        return np.asarray(r_prev, float) - self.hat_prev(conf_df)

    def transform_next(self, conf_df: pd.DataFrame, r_next: np.ndarray) -> np.ndarray:
        return np.asarray(r_next, float) - self.hat_next(conf_df)

class SingleVarResidualizer:


    def __init__(self, conf_cols: List[str], ridge_alpha: float = 1.0):
        self.conf_cols = list(conf_cols)
        self.ridge_alpha = float(ridge_alpha)
        self.encoder: Optional[OneHotEncoder] = None
        self.lr: Optional[Ridge] = None

    def _prep_conf(self, conf_df: pd.DataFrame) -> pd.DataFrame:
        out = conf_df.copy()
        for c in self.conf_cols:
            if c not in out.columns:
                out[c] = MISSING_TOKEN
            out[c] = out[c].astype(str).fillna(MISSING_TOKEN).str.strip()
        return out[self.conf_cols]

    def fit(self, conf_df: pd.DataFrame, x: np.ndarray) -> "SingleVarResidualizer":
        conf_df = self._prep_conf(conf_df)
        self.encoder = make_onehot_encoder(drop_first=True)
        Z = self.encoder.fit_transform(conf_df)
        self.lr = Ridge(alpha=self.ridge_alpha, fit_intercept=True)
        self.lr.fit(Z, x.reshape(-1, 1))
        return self

    def hat(self, conf_df: pd.DataFrame) -> np.ndarray:
        if self.encoder is None or self.lr is None:
            raise RuntimeError("Residualizer not fitted.")
        conf_df = self._prep_conf(conf_df)
        Z = self.encoder.transform(conf_df)
        return self.lr.predict(Z).ravel()

    def transform(self, conf_df: pd.DataFrame, x: np.ndarray) -> np.ndarray:
        return np.asarray(x, float) - self.hat(conf_df)

def _coerce_confound_columns(df: pd.DataFrame, conf_cols: List[str]) -> pd.DataFrame:


    out = df.copy()
    for c in conf_cols:
        if c not in out.columns:
            out[c] = MISSING_TOKEN
        out[c] = out[c].fillna(MISSING_TOKEN).astype(str).str.strip()
        out.loc[out[c] == "", c] = MISSING_TOKEN
    return out

def _make_monkey_conf_for_transform(
    df: pd.DataFrame,
    human_ref_conf: Dict[str, str],
) -> pd.DataFrame:


    conf = df.reindex(columns=CONF_COLS).copy()

    for c in CONF_COLS:
        if c not in conf.columns:
            conf[c] = MISSING_TOKEN
        conf[c] = conf[c].fillna(MISSING_TOKEN).astype(str).str.strip()
        conf.loc[conf[c] == "", c] = MISSING_TOKEN

    if not MONKEY_ADJUST_SEX:
        conf["sex"] = human_ref_conf.get("sex", MISSING_TOKEN)

    if MONKEY_USE_HUMAN_REF_FOR_SCANNER_SITE:
        conf["site"] = human_ref_conf.get("site", MISSING_TOKEN)
        conf["scanner_manufacturer"] = human_ref_conf.get(
            "scanner_manufacturer", MISSING_TOKEN
        )
        conf["scanner_model"] = human_ref_conf.get("scanner_model", MISSING_TOKEN)

    return conf

def standardize_human_long(df: pd.DataFrame) -> pd.DataFrame:


    df = df.copy()

    required = ["subject_id", "eventname", "hemi", "region", "CT", "SA"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"[HUMAN] Missing required column: {c}")

    if "age_months" in df.columns:
        df["age_months"] = detect_and_convert_age_to_months(df["age_months"])
    elif "age_years" in df.columns:
        df["age_months"] = pd.to_numeric(df["age_years"], errors="coerce").astype(float) * 12.0
    else:
        raise ValueError("[HUMAN] Need age_months or age_years")

    df["subject_id"] = df["subject_id"].astype(str).str.strip()
    df["scan_id"] = df["eventname"].astype(str).str.strip()
    df["hemi"] = df["hemi"].astype(str).str.lower().str.strip()
    df["expansion_group"] = normalize_expansion_group(df["region"])

    df = df[df["hemi"] == "bilateral"].copy()
    df = df[df["expansion_group"].isin(list(EXPANSION_GROUPS))].copy()

    df["CT"] = pd.to_numeric(df["CT"], errors="coerce").astype(float)
    df["SA"] = pd.to_numeric(df["SA"], errors="coerce").astype(float)

    df = _coerce_confound_columns(df, CONF_COLS)
    df["sex"] = normalize_sex_codes(df["sex"], dataset="human")

    out_cols = ["subject_id", "scan_id", "age_months", "hemi", "expansion_group"] + METRICS + CONF_COLS
    return df[out_cols].dropna(subset=["subject_id", "scan_id", "age_months", "expansion_group"])

def standardize_monkey_long(df: pd.DataFrame) -> pd.DataFrame:


    df = df.copy()

    required = [
        "scan_name", "monkey_id", "age", "hemi", "region",
        "CT_mean_aw", "SA_total"
    ]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"[MONKEY] Missing required column: {c}")

    df["age_months"] = detect_and_convert_age_to_months(df["age"])
    df["monkey_id"] = df["monkey_id"].astype(str).str.strip()
    df["scan_id"] = df["scan_name"].astype(str).str.strip()
    df["hemi"] = df["hemi"].astype(str).str.lower().str.strip()
    df["expansion_group"] = normalize_expansion_group(df["region"])

    df = df[df["hemi"] == "bilateral"].copy()
    df = df[df["expansion_group"].isin(list(EXPANSION_GROUPS))].copy()

    df["CT"] = pd.to_numeric(df["CT_mean_aw"], errors="coerce").astype(float)
    df["SA"] = pd.to_numeric(df["SA_total"], errors="coerce").astype(float)


    df = _coerce_confound_columns(df, CONF_COLS)
    df["sex"] = normalize_sex_codes(df["sex"], dataset="macaque")

    out_cols = ["monkey_id", "scan_id", "age_months", "hemi", "expansion_group"] + METRICS + CONF_COLS
    return df[out_cols].dropna(subset=["monkey_id", "scan_id", "age_months", "expansion_group"])

def pivot_long_to_scan_wide_with_confounds(
    df_long: pd.DataFrame,
    id_col: str,
    scan_col: str,
    age_col: str,
    region_col: str,
    metrics: List[str],
    conf_cols: List[str],
) -> pd.DataFrame:

    df = df_long.copy()
    df = df.dropna(subset=[id_col, scan_col, age_col, region_col]).copy()

    for c in conf_cols:
        if c not in df.columns:
            df[c] = MISSING_TOKEN
        df[c] = df[c].astype(str).fillna(MISSING_TOKEN)

    df = df.sort_values([id_col, scan_col, region_col, age_col]).drop_duplicates(
        subset=[id_col, scan_col, region_col],
        keep="first",
    )

    wide_metrics = df.pivot_table(
        index=[id_col, scan_col, age_col],
        columns=region_col,
        values=metrics,
        aggfunc="first",
    )
    wide_metrics.columns = [f"{m}_{r}" for (m, r) in wide_metrics.columns.to_list()]
    wide_metrics = wide_metrics.reset_index()

    wide_conf = df.groupby([id_col, scan_col, age_col], as_index=False)[conf_cols].first()
    wide = pd.merge(wide_metrics, wide_conf, on=[id_col, scan_col, age_col], how="left")

    for r in EXPANSION_GROUPS:
        for m in metrics:
            col = f"{m}_{r}"
            if col not in wide.columns:
                wide[col] = np.nan

    for c in conf_cols:
        if c not in wide.columns:
            wide[c] = MISSING_TOKEN
        wide[c] = wide[c].astype(str).fillna(MISSING_TOKEN)

    return wide

def build_triples_rate12_to_rate23(
    scan_wide: pd.DataFrame,
    id_col: str,
    scan_col: str,
    age_col: str,
    metrics: List[str],
    conf_cols: List[str],
    dt_scale: float = 1.0,
) -> pd.DataFrame:

    df = scan_wide.copy()
    df[age_col] = pd.to_numeric(df[age_col], errors="coerce").astype(float)
    df = df.dropna(subset=[id_col, scan_col, age_col]).copy()

    for c in conf_cols:
        if c not in df.columns:
            df[c] = MISSING_TOKEN
        df[c] = df[c].astype(str).fillna(MISSING_TOKEN)

    rows: List[Dict[str, Any]] = []
    for sid, g in df.groupby(id_col, sort=False):
        g = g.sort_values(age_col).reset_index(drop=True)
        if len(g) < 3:
            continue

        for i in range(len(g) - 2):
            r1 = g.iloc[i]
            r2 = g.iloc[i + 1]
            r3 = g.iloc[i + 2]

            age1 = float(r1[age_col])
            age2 = float(r2[age_col])
            age3 = float(r3[age_col])

            dt12_native = age2 - age1
            dt23_native = age3 - age2
            if not (np.isfinite(dt12_native) and np.isfinite(dt23_native)):
                continue
            if dt12_native <= 0 or dt23_native <= 0:
                continue

            dt12_used = dt12_native * float(dt_scale)
            dt23_used = dt23_native * float(dt_scale)

            sample: Dict[str, Any] = {
                "id": str(sid),
                "scan_1": str(r1[scan_col]),
                "scan_2": str(r2[scan_col]),
                "scan_3": str(r3[scan_col]),
                "age_1": age1,
                "age_2": age2,
                "age_3": age3,
                "age_T2": age2,
                "dt12_months_used": dt12_used,
                "dt23_months_used": dt23_used,
                "dt23_months_native": dt23_native,
            }

            for c in conf_cols:
                sample[c] = str(r2.get(c, MISSING_TOKEN)) if pd.notna(r2.get(c, np.nan)) else MISSING_TOKEN

            for reg in EXPANSION_GROUPS:
                for m in metrics:
                    col = f"{m}_{reg}"
                    x1 = float(pd.to_numeric(r1.get(col, np.nan), errors="coerce"))
                    x2 = float(pd.to_numeric(r2.get(col, np.nan), errors="coerce"))
                    x3 = float(pd.to_numeric(r3.get(col, np.nan), errors="coerce"))

                    sample[f"{m}_{reg}_base_T1"] = x1
                    sample[f"{m}_{reg}_base_T2"] = x2

                    if np.isfinite(x1) and np.isfinite(x2) and dt12_used > 0:
                        sample[f"{m}_{reg}_r12"] = spc_rate([x1], [x2], [dt12_used])[0]
                    else:
                        sample[f"{m}_{reg}_r12"] = np.nan

                    if np.isfinite(x2) and np.isfinite(x3) and dt23_used > 0:
                        sample[f"{m}_{reg}_r23"] = spc_rate([x2], [x3], [dt23_used])[0]
                    else:
                        sample[f"{m}_{reg}_r23"] = np.nan

            rows.append(sample)

    return pd.DataFrame(rows)

def _make_knots_strictly_increasing(knots: np.ndarray, lb: float, ub: float) -> np.ndarray:

    if knots.size == 0:
        return knots.astype(float)

    k = np.sort(knots.astype(float))
    k = np.clip(k, lb + BOUND_EPS, ub - BOUND_EPS)

    for i in range(1, len(k)):
        if k[i] <= k[i - 1]:
            k[i] = k[i - 1] + 10.0 * BOUND_EPS

    if k[-1] >= ub - BOUND_EPS:
        k = np.linspace(lb + 2.0 * BOUND_EPS, ub - 2.0 * BOUND_EPS, num=len(k))

    return k

def fit_spline_spec_from_human_ages(human_ages: np.ndarray, lower_bound: float, upper_bound: float) -> Dict[str, Any]:

    ages = np.asarray(human_ages, float)
    ages = ages[np.isfinite(ages)]
    if ages.size < 10:
        raise ValueError("Not enough human ages to fit spline spec.")
    lb = float(lower_bound)
    ub = float(upper_bound)

    if N_INTERNAL_KNOTS <= 0:
        knots = np.array([], dtype=float)
    else:
        qs = np.linspace(0, 1, N_INTERNAL_KNOTS + 2)[1:-1]
        knots = np.quantile(ages, qs).astype(float)
        knots = _make_knots_strictly_increasing(knots, lb, ub)

    return {"knots": knots, "lb": lb, "ub": ub}

def add_age_spline_with_spec(df: pd.DataFrame, age_col: str, spec: Dict[str, Any]) -> Tuple[pd.DataFrame, List[str]]:

    ages = pd.to_numeric(df[age_col], errors="coerce").astype(float).values
    lb = float(spec["lb"])
    ub = float(spec["ub"])
    ages = np.clip(ages, lb + BOUND_EPS, ub - BOUND_EPS)

    dm = dmatrix(
        "bs(age, knots=knots, degree=deg, include_intercept=False, lower_bound=lb, upper_bound=ub)",
        {"age": ages, "knots": spec["knots"], "deg": SPLINE_DEGREE, "lb": lb, "ub": ub},
        return_type="dataframe",
    )
    spline_cols = [f"age_spline_{i:d}" for i in range(dm.shape[1])]
    dm.columns = spline_cols
    out = pd.concat([df.reset_index(drop=True), dm.reset_index(drop=True)], axis=1)
    return out, spline_cols

def build_design_matrix_rate12_to_rate23(
    df: pd.DataFrame,
    r12_col: str,
    spline_cols: List[str],
    include_baseline_t2: bool,
    baseline_t2_col: Optional[str],
    include_c_ratio: bool,
    ratio_col: Optional[str],
    use_interactions: bool = True,
) -> Tuple[np.ndarray, List[str]]:

    parts: List[np.ndarray] = []
    names: List[str] = []

    xr = pd.to_numeric(df[r12_col], errors="coerce").values.astype(float).reshape(-1, 1)
    parts.append(xr)
    names.append(r12_col)

    xdt = pd.to_numeric(df["dt23_months_used"], errors="coerce").values.astype(float).reshape(-1, 1)
    parts.append(xdt)
    names.append("dt23_months_used")

    xs = df[spline_cols].apply(pd.to_numeric, errors="coerce").values.astype(float)
    parts.append(xs)
    names.extend(spline_cols)

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

    X = np.concatenate(parts, axis=1)
    return X, names

def train_single_metric_rate12_to_rate23(
    human_triples: pd.DataFrame,
    expansion_group: str,
    metric: str,
    spline_spec: Dict[str, Any],
    out_prefix: str,
) -> Dict[str, Any]:

    df = human_triples.copy()
    df["age_spline_months"] = df["age_T2"].astype(float)

    r12_raw_col = f"{metric}_{expansion_group}_r12"
    r23_raw_col = f"{metric}_{expansion_group}_r23"
    base_t1_raw_col = f"{metric}_{expansion_group}_base_T1"
    base_t2_raw_col = f"{metric}_{expansion_group}_base_T2"
    ratio_col = f"{metric}_{expansion_group}_logratio_T2T1"

    if INCLUDE_C_RATIO:
        df[ratio_col] = safe_log_ratio(
            pd.to_numeric(df[base_t2_raw_col], errors="coerce").values.astype(float),
            pd.to_numeric(df[base_t1_raw_col], errors="coerce").values.astype(float),
            eps=RATIO_EPS,
        )

    req = [
        "id", "age_spline_months", "dt12_months_used", "dt23_months_used",
        r12_raw_col, r23_raw_col, base_t1_raw_col, base_t2_raw_col,
    ] + CONF_COLS
    if INCLUDE_C_RATIO:
        req.append(ratio_col)
    df = df.dropna(subset=req).copy()

    if FILTER_EXTREME_RATES:
        before_n = len(df)
        df = filter_extreme_rows_for_metric(
            df=df,
            r12_col=r12_raw_col,
            r23_col=r23_raw_col,
            dt12_col="dt12_months_used",
            dt23_col="dt23_months_used",
            rate_abs_max=RATE_ABS_MAX,
            dt_used_min=DT_USED_MIN,
        )
        after_n = len(df)
        pd.Series({"before": before_n, "after": after_n}).to_csv(
            PREDICTION_OUT_DIR / f"{out_prefix}_human_qc_counts.csv"
        )

    df, spline_cols = add_age_spline_with_spec(df, "age_spline_months", spline_spec)

    ok = np.isfinite(pd.to_numeric(df[r12_raw_col], errors="coerce").values)
    ok &= np.isfinite(pd.to_numeric(df[r23_raw_col], errors="coerce").values)
    ok &= np.isfinite(pd.to_numeric(df["dt12_months_used"], errors="coerce").values)
    ok &= np.isfinite(pd.to_numeric(df["dt23_months_used"], errors="coerce").values)
    ok &= np.isfinite(pd.to_numeric(df[base_t1_raw_col], errors="coerce").values)
    ok &= np.isfinite(pd.to_numeric(df[base_t2_raw_col], errors="coerce").values)
    if INCLUDE_C_RATIO:
        ok &= np.isfinite(pd.to_numeric(df[ratio_col], errors="coerce").values)
    ok &= np.isfinite(df[spline_cols].apply(pd.to_numeric, errors="coerce").values).all(axis=1)
    for c in CONF_COLS:
        ok &= df[c].notna().values
    df = df.loc[ok].reset_index(drop=True)

    human_ref_conf = {
        "sex": mode_or_missing(df["sex"]),
        "site": mode_or_missing(df["site"]),
        "scanner_manufacturer": mode_or_missing(df["scanner_manufacturer"]),
        "scanner_model": mode_or_missing(df["scanner_model"]),
    }
    pd.Series(human_ref_conf).to_csv(PREDICTION_OUT_DIR / f"{out_prefix}_human_ref_confounds.csv")

    groups = df["id"].astype(str).values
    n_splits_eff = _effective_n_splits(N_SPLITS, groups)
    if n_splits_eff < 2:
        raise RuntimeError(f"Not enough distinct subjects for GroupKFold (need >=2, got {n_splits_eff}).")
    cv = GroupKFold(n_splits=n_splits_eff)

    best_score = -np.inf
    best_params = {"alpha": np.nan, "l1_ratio": np.nan}

    for alpha in ALPHAS:
        for l1_ratio in L1_RATIOS:
            fold_scores: List[float] = []
            for tr, te in cv.split(df, groups=groups):
                df_tr = df.iloc[tr].reset_index(drop=True)
                df_te = df.iloc[te].reset_index(drop=True)

                r12_tr = pd.to_numeric(df_tr[r12_raw_col], errors="coerce").values.astype(float)
                r23_tr = pd.to_numeric(df_tr[r23_raw_col], errors="coerce").values.astype(float)
                r12_te = pd.to_numeric(df_te[r12_raw_col], errors="coerce").values.astype(float)
                r23_te = pd.to_numeric(df_te[r23_raw_col], errors="coerce").values.astype(float)

                rr = RateResidualizer(CONF_COLS, ridge_alpha=RIDGE_ALPHA_RATE).fit(
                    df_tr[CONF_COLS], r12_tr, r23_tr
                )
                r12_tr_res = rr.transform_prev(df_tr[CONF_COLS], r12_tr)
                r23_tr_res = rr.transform_next(df_tr[CONF_COLS], r23_tr)
                r12_te_res = rr.transform_prev(df_te[CONF_COLS], r12_te)
                r23_te_res = rr.transform_next(df_te[CONF_COLS], r23_te)

                df_tr2 = df_tr.copy()
                df_te2 = df_te.copy()
                r12_res_col = f"{r12_raw_col}_resid"
                df_tr2[r12_res_col] = r12_tr_res
                df_te2[r12_res_col] = r12_te_res

                baseline_col_used = None
                br = None
                if INCLUDE_BASELINE_T2:
                    baseline_col_used = base_t2_raw_col
                    if RESIDUALIZE_BASELINE_T2:
                        br = SingleVarResidualizer(CONF_COLS, ridge_alpha=RIDGE_ALPHA_BASELINE).fit(
                            df_tr[CONF_COLS],
                            pd.to_numeric(df_tr[base_t2_raw_col], errors="coerce").values.astype(float),
                        )
                        df_tr2[f"{base_t2_raw_col}_resid"] = br.transform(
                            df_tr[CONF_COLS],
                            pd.to_numeric(df_tr[base_t2_raw_col], errors="coerce").values.astype(float),
                        )
                        df_te2[f"{base_t2_raw_col}_resid"] = br.transform(
                            df_te[CONF_COLS],
                            pd.to_numeric(df_te[base_t2_raw_col], errors="coerce").values.astype(float),
                        )
                        baseline_col_used = f"{base_t2_raw_col}_resid"

                ratio_used = ratio_col if INCLUDE_C_RATIO else None

                Xtr, names = build_design_matrix_rate12_to_rate23(
                    df_tr2,
                    r12_col=r12_res_col,
                    spline_cols=spline_cols,
                    include_baseline_t2=INCLUDE_BASELINE_T2,
                    baseline_t2_col=baseline_col_used,
                    include_c_ratio=INCLUDE_C_RATIO,
                    ratio_col=ratio_used,
                    use_interactions=True,
                )
                Xte, _ = build_design_matrix_rate12_to_rate23(
                    df_te2,
                    r12_col=r12_res_col,
                    spline_cols=spline_cols,
                    include_baseline_t2=INCLUDE_BASELINE_T2,
                    baseline_t2_col=baseline_col_used,
                    include_c_ratio=INCLUDE_C_RATIO,
                    ratio_col=ratio_used,
                    use_interactions=True,
                )

                keep_mask, _ = compute_feature_keep_mask_from_Xtr(Xtr, names)
                Xtr_k, _ = apply_keep_mask(Xtr, names, keep_mask)
                Xte_k, _ = apply_keep_mask(Xte, names, keep_mask)

                if CLIP_FEATURES_TO_HUMAN_QUANTILES:
                    lo = np.quantile(Xtr_k, CLIP_LO_Q, axis=0)
                    hi = np.quantile(Xtr_k, CLIP_HI_Q, axis=0)
                    Xtr_k = clip_X_by_bounds(Xtr_k, lo, hi)
                    Xte_k = clip_X_by_bounds(Xte_k, lo, hi)

                model = Pipeline([
                    ("scaler", StandardScaler()),
                    ("model", ElasticNet(
                        alpha=float(alpha),
                        l1_ratio=float(l1_ratio),
                        max_iter=50000,
                        random_state=0,
                    )),
                ])
                model.fit(Xtr_k, r23_tr_res)
                pred = model.predict(Xte_k)
                fold_scores.append(float(r2_score(r23_te_res, pred)))

            score = float(np.nanmean(fold_scores)) if fold_scores else -np.inf
            if score > best_score:
                best_score = score
                best_params = {"alpha": float(alpha), "l1_ratio": float(l1_ratio)}

    fold_rows: List[Dict[str, Any]] = []
    oof_rows: List[pd.DataFrame] = []

    for fold_idx, (tr, te) in enumerate(cv.split(df, groups=groups), start=1):
        df_tr = df.iloc[tr].reset_index(drop=True)
        df_te = df.iloc[te].reset_index(drop=True)

        r12_tr = pd.to_numeric(df_tr[r12_raw_col], errors="coerce").values.astype(float)
        r23_tr = pd.to_numeric(df_tr[r23_raw_col], errors="coerce").values.astype(float)
        r12_te = pd.to_numeric(df_te[r12_raw_col], errors="coerce").values.astype(float)
        r23_te = pd.to_numeric(df_te[r23_raw_col], errors="coerce").values.astype(float)

        rr = RateResidualizer(CONF_COLS, ridge_alpha=RIDGE_ALPHA_RATE).fit(
            df_tr[CONF_COLS], r12_tr, r23_tr
        )
        r12_tr_res = rr.transform_prev(df_tr[CONF_COLS], r12_tr)
        r23_tr_res = rr.transform_next(df_tr[CONF_COLS], r23_tr)
        r12_te_res = rr.transform_prev(df_te[CONF_COLS], r12_te)
        r23_te_res = rr.transform_next(df_te[CONF_COLS], r23_te)

        df_tr2 = df_tr.copy()
        df_te2 = df_te.copy()
        r12_res_col = f"{r12_raw_col}_resid"
        df_tr2[r12_res_col] = r12_tr_res
        df_te2[r12_res_col] = r12_te_res

        baseline_col_used = None
        br = None
        if INCLUDE_BASELINE_T2:
            baseline_col_used = base_t2_raw_col
            if RESIDUALIZE_BASELINE_T2:
                br = SingleVarResidualizer(CONF_COLS, ridge_alpha=RIDGE_ALPHA_BASELINE).fit(
                    df_tr[CONF_COLS],
                    pd.to_numeric(df_tr[base_t2_raw_col], errors="coerce").values.astype(float),
                )
                df_tr2[f"{base_t2_raw_col}_resid"] = br.transform(
                    df_tr[CONF_COLS],
                    pd.to_numeric(df_tr[base_t2_raw_col], errors="coerce").values.astype(float),
                )
                df_te2[f"{base_t2_raw_col}_resid"] = br.transform(
                    df_te[CONF_COLS],
                    pd.to_numeric(df_te[base_t2_raw_col], errors="coerce").values.astype(float),
                )
                baseline_col_used = f"{base_t2_raw_col}_resid"

        ratio_used = ratio_col if INCLUDE_C_RATIO else None

        Xtr, names = build_design_matrix_rate12_to_rate23(
            df_tr2,
            r12_col=r12_res_col,
            spline_cols=spline_cols,
            include_baseline_t2=INCLUDE_BASELINE_T2,
            baseline_t2_col=baseline_col_used,
            include_c_ratio=INCLUDE_C_RATIO,
            ratio_col=ratio_used,
            use_interactions=True,
        )
        Xte, _ = build_design_matrix_rate12_to_rate23(
            df_te2,
            r12_col=r12_res_col,
            spline_cols=spline_cols,
            include_baseline_t2=INCLUDE_BASELINE_T2,
            baseline_t2_col=baseline_col_used,
            include_c_ratio=INCLUDE_C_RATIO,
            ratio_col=ratio_used,
            use_interactions=True,
        )

        keep_mask, _ = compute_feature_keep_mask_from_Xtr(Xtr, names)
        Xtr_k, _ = apply_keep_mask(Xtr, names, keep_mask)
        Xte_k, _ = apply_keep_mask(Xte, names, keep_mask)

        if CLIP_FEATURES_TO_HUMAN_QUANTILES:
            lo = np.quantile(Xtr_k, CLIP_LO_Q, axis=0)
            hi = np.quantile(Xtr_k, CLIP_HI_Q, axis=0)
            Xtr_k = clip_X_by_bounds(Xtr_k, lo, hi)
            Xte_k = clip_X_by_bounds(Xte_k, lo, hi)

        model = Pipeline([
            ("scaler", StandardScaler()),
            ("model", ElasticNet(
                alpha=best_params["alpha"],
                l1_ratio=best_params["l1_ratio"],
                max_iter=50000,
                random_state=0,
            )),
        ])
        model.fit(Xtr_k, r23_tr_res)
        pred_res = model.predict(Xte_k)

        fold_rows.append({
            "fold": fold_idx,
            "variant": MODEL_VARIANT,
            "expansion_group": expansion_group,
            "metric": metric,
            "target": r23_raw_col,
            "r2_resid": float(r2_score(r23_te_res, pred_res)),
            "pearson_r_resid": safe_corr(r23_te_res, pred_res, method="pearson"),
            "spearman_r_resid": safe_corr(r23_te_res, pred_res, method="spearman"),
            "n_test": int(len(te)),
            "n_feat_kept": int(np.sum(keep_mask)),
        })

        meta = df_te[[
            "id", "scan_1", "scan_2", "scan_3",
            "age_1", "age_2", "age_3",
            "dt12_months_used", "dt23_months_used",
        ] + CONF_COLS].copy()
        meta["variant"] = MODEL_VARIANT
        meta["expansion_group"] = expansion_group
        meta["metric"] = metric
        meta["target"] = r23_raw_col
        meta["r12_raw"] = r12_te
        meta["r12_resid"] = r12_te_res
        meta["r23_raw"] = r23_te
        meta["r23_resid"] = r23_te_res
        meta["pred_r23_resid"] = pred_res
        meta["base_T1"] = pd.to_numeric(df_te[base_t1_raw_col], errors="coerce").values.astype(float)
        meta["base_T2"] = pd.to_numeric(df_te[base_t2_raw_col], errors="coerce").values.astype(float)
        if INCLUDE_C_RATIO:
            meta["logratio_T2T1"] = pd.to_numeric(df_te[ratio_col], errors="coerce").values.astype(float)
        oof_rows.append(meta)

    metrics_df = pd.DataFrame(fold_rows)
    metrics_df = pd.concat([
        metrics_df,
        pd.DataFrame([{
            "fold": "mean_overall",
            "variant": MODEL_VARIANT,
            "expansion_group": expansion_group,
            "metric": metric,
            "target": r23_raw_col,
            "r2_resid": float(metrics_df["r2_resid"].mean()),
            "pearson_r_resid": float(metrics_df["pearson_r_resid"].mean()),
            "spearman_r_resid": float(metrics_df["spearman_r_resid"].mean()),
            "n_test": int(metrics_df["n_test"].sum()),
            "n_feat_kept": float(metrics_df["n_feat_kept"].mean()),
        }]),
    ], axis=0, ignore_index=True)
    oof_df = pd.concat(oof_rows, axis=0, ignore_index=True) if oof_rows else pd.DataFrame()

    r12_all = pd.to_numeric(df[r12_raw_col], errors="coerce").values.astype(float)
    r23_all = pd.to_numeric(df[r23_raw_col], errors="coerce").values.astype(float)

    final_rr = RateResidualizer(CONF_COLS, ridge_alpha=RIDGE_ALPHA_RATE).fit(
        df[CONF_COLS], r12_all, r23_all
    )
    r12_all_res = final_rr.transform_prev(df[CONF_COLS], r12_all)
    r23_all_res = final_rr.transform_next(df[CONF_COLS], r23_all)

    _save_human_hat_stats(
        out_prefix=out_prefix,
        rr=final_rr,
        conf_df=df[CONF_COLS],
        q_lo=HAT_CLIP_Q_LO,
        q_hi=HAT_CLIP_Q_HI,
    )

    df_all2 = df.copy()
    r12_res_col = f"{r12_raw_col}_resid"
    df_all2[r12_res_col] = r12_all_res

    baseline_col_used = None
    final_br = None
    if INCLUDE_BASELINE_T2:
        baseline_col_used = base_t2_raw_col
        if RESIDUALIZE_BASELINE_T2:
            final_br = SingleVarResidualizer(CONF_COLS, ridge_alpha=RIDGE_ALPHA_BASELINE).fit(
                df[CONF_COLS],
                pd.to_numeric(df[base_t2_raw_col], errors="coerce").values.astype(float),
            )
            df_all2[f"{base_t2_raw_col}_resid"] = final_br.transform(
                df[CONF_COLS],
                pd.to_numeric(df[base_t2_raw_col], errors="coerce").values.astype(float),
            )
            baseline_col_used = f"{base_t2_raw_col}_resid"

    ratio_used = ratio_col if INCLUDE_C_RATIO else None
    Xall, names_all = build_design_matrix_rate12_to_rate23(
        df_all2,
        r12_col=r12_res_col,
        spline_cols=spline_cols,
        include_baseline_t2=INCLUDE_BASELINE_T2,
        baseline_t2_col=baseline_col_used,
        include_c_ratio=INCLUDE_C_RATIO,
        ratio_col=ratio_used,
        use_interactions=True,
    )

    keep_all, std_all = compute_feature_keep_mask_from_Xtr(Xall, names_all)
    Xall_k, kept_names_all = apply_keep_mask(Xall, names_all, keep_all)

    save_feature_mask_csv(
        PREDICTION_OUT_DIR / f"{out_prefix}_feature_keep_mask.csv",
        names_all,
        keep_all,
        std_vec=std_all,
    )

    save_feature_clip_stats_csv(
        PREDICTION_OUT_DIR / f"{out_prefix}_feature_clip_stats.csv",
        kept_names_all,
        Xall_k,
        q_lo=CLIP_LO_Q,
        q_hi=CLIP_HI_Q,
    )

    if CLIP_FEATURES_TO_HUMAN_QUANTILES:
        lo = np.quantile(Xall_k, CLIP_LO_Q, axis=0)
        hi = np.quantile(Xall_k, CLIP_HI_Q, axis=0)
        Xall_k = clip_X_by_bounds(Xall_k, lo, hi)

    final_model = Pipeline([
        ("scaler", StandardScaler()),
        ("model", ElasticNet(
            alpha=best_params["alpha"],
            l1_ratio=best_params["l1_ratio"],
            max_iter=50000,
            random_state=0,
        )),
    ])
    final_model.fit(Xall_k, r23_all_res)

    metrics_df.to_csv(PREDICTION_OUT_DIR / f"{out_prefix}_human_cv_metrics.csv", index=False)
    oof_df.to_csv(PREDICTION_OUT_DIR / f"{out_prefix}_human_oof_predictions.csv", index=False)
    pd.Series(best_params).to_csv(PREDICTION_OUT_DIR / f"{out_prefix}_best_params.csv")
    joblib.dump(final_model, PREDICTION_OUT_DIR / f"{out_prefix}_human_model.joblib")
    joblib.dump(final_rr, PREDICTION_OUT_DIR / f"{out_prefix}_human_rate_residualizer.joblib")
    if final_br is not None:
        joblib.dump(final_br, PREDICTION_OUT_DIR / f"{out_prefix}_human_baseline_residualizer.joblib")

    pd.Series({
        "variant": MODEL_VARIANT,
        "expansion_group": expansion_group,
        "metric": metric,
        "include_baseline_T2": int(INCLUDE_BASELINE_T2),
        "include_c_ratio": int(INCLUDE_C_RATIO),
        "ratio_eps": float(RATIO_EPS),
        "residualize_baseline_T2": int(RESIDUALIZE_BASELINE_T2),
        "best_alpha": float(best_params["alpha"]),
        "best_l1_ratio": float(best_params["l1_ratio"]),
        "VAR_STD_MIN": float(VAR_STD_MIN),
        "CONST_STD_MIN": float(CONST_STD_MIN),
        "CLIP_FEATURES_TO_HUMAN_QUANTILES": int(CLIP_FEATURES_TO_HUMAN_QUANTILES),
        "CLIP_LO_Q": float(CLIP_LO_Q),
        "CLIP_HI_Q": float(CLIP_HI_Q),
        "FILTER_EXTREME_RATES": int(FILTER_EXTREME_RATES),
        "RATE_ABS_MAX": float(RATE_ABS_MAX),
        "DT_USED_MIN": float(DT_USED_MIN),
        "residualizer": "onehot_drop_first + ridge",
        "ridge_alpha_rate": float(RIDGE_ALPHA_RATE),
        "ridge_alpha_baseline": float(RIDGE_ALPHA_BASELINE),
        "hat_clip_enabled": int(ENABLE_HAT_CLIP),
        "hat_clip_q_lo": float(HAT_CLIP_Q_LO),
        "hat_clip_q_hi": float(HAT_CLIP_Q_HI),
        "monkey_calibration_enabled": int(ENABLE_MONKEY_CALIBRATION),
        "calibration_n_splits": int(CALIBRATION_N_SPLITS),
        "calibrate_space": CALIBRATE_SPACE,
        "n_features_total": int(len(names_all)),
        "n_features_kept": int(np.sum(keep_all)),
    }).to_csv(PREDICTION_OUT_DIR / f"{out_prefix}_model_spec.csv")

    return {
        "model": final_model,
        "rate_residualizer": final_rr,
        "baseline_residualizer": final_br,
        "metrics_df": metrics_df,
        "oof_df": oof_df,
        "human_ref_conf": human_ref_conf,
        "spline_cols": spline_cols,
    }

def apply_single_metric_rate12_to_rate23_to_monkey(
    monkey_triples: pd.DataFrame,
    expansion_group: str,
    metric: str,
    spline_spec: Dict[str, Any],
    trained_model: Pipeline,
    trained_rate_residualizer: RateResidualizer,
    trained_baseline_residualizer: Optional[SingleVarResidualizer],
    human_ref_conf: Dict[str, str],
    out_prefix: str,
) -> pd.DataFrame:

    df = monkey_triples.copy()
    df["age_spline_months"] = df["age_T2"].astype(float) * AGE_RATIO

    r12_raw_col = f"{metric}_{expansion_group}_r12"
    r23_raw_col = f"{metric}_{expansion_group}_r23"
    base_t1_raw_col = f"{metric}_{expansion_group}_base_T1"
    base_t2_raw_col = f"{metric}_{expansion_group}_base_T2"
    ratio_col = f"{metric}_{expansion_group}_logratio_T2T1"

    if INCLUDE_C_RATIO:
        df[ratio_col] = safe_log_ratio(
            pd.to_numeric(df[base_t2_raw_col], errors="coerce").values.astype(float),
            pd.to_numeric(df[base_t1_raw_col], errors="coerce").values.astype(float),
            eps=RATIO_EPS,
        )

    req = [
        "id", "age_spline_months", "dt12_months_used", "dt23_months_used",
        r12_raw_col, r23_raw_col, base_t1_raw_col, base_t2_raw_col,
    ] + CONF_COLS
    if INCLUDE_C_RATIO:
        req.append(ratio_col)
    df = df.dropna(subset=req).copy()

    if FILTER_EXTREME_RATES:
        before_n = len(df)
        df = filter_extreme_rows_for_metric(
            df=df,
            r12_col=r12_raw_col,
            r23_col=r23_raw_col,
            dt12_col="dt12_months_used",
            dt23_col="dt23_months_used",
            rate_abs_max=RATE_ABS_MAX,
            dt_used_min=DT_USED_MIN,
        )
        after_n = len(df)
        pd.Series({"before": before_n, "after": after_n}).to_csv(
            PREDICTION_OUT_DIR / f"{out_prefix}_monkey_qc_counts.csv"
        )

    df, spline_cols = add_age_spline_with_spec(df, "age_spline_months", spline_spec)

    r12 = pd.to_numeric(df[r12_raw_col], errors="coerce").values.astype(float)
    r23 = pd.to_numeric(df[r23_raw_col], errors="coerce").values.astype(float)

    conf_used = _make_monkey_conf_for_transform(df, human_ref_conf)
    r12_hat_raw = trained_rate_residualizer.hat_prev(conf_used)
    r23_hat_raw = trained_rate_residualizer.hat_next(conf_used)
    r12_hat, r23_hat, clip_meta = _clip_hat_with_human_stats(out_prefix, r12_hat_raw, r23_hat_raw)

    r12_res = r12 - r12_hat
    r23_res = r23 - r23_hat

    df2 = df.copy()
    r12_res_col = f"{r12_raw_col}_resid"
    df2[r12_res_col] = r12_res

    baseline_col_used = None
    if INCLUDE_BASELINE_T2:
        baseline_col_used = base_t2_raw_col
        if RESIDUALIZE_BASELINE_T2:
            if trained_baseline_residualizer is None:
                raise RuntimeError("Baseline residualizer is required but not provided.")
            b = pd.to_numeric(df2[base_t2_raw_col], errors="coerce").values.astype(float)
            df2[f"{base_t2_raw_col}_resid"] = trained_baseline_residualizer.transform(conf_used, b)
            baseline_col_used = f"{base_t2_raw_col}_resid"

    ratio_used = ratio_col if INCLUDE_C_RATIO else None
    X, names = build_design_matrix_rate12_to_rate23(
        df2,
        r12_col=r12_res_col,
        spline_cols=spline_cols,
        include_baseline_t2=INCLUDE_BASELINE_T2,
        baseline_t2_col=baseline_col_used,
        include_c_ratio=INCLUDE_C_RATIO,
        ratio_col=ratio_used,
        use_interactions=True,
    )

    mask_path = PREDICTION_OUT_DIR / f"{out_prefix}_feature_keep_mask.csv"
    if not mask_path.exists():
        raise FileNotFoundError(f"Missing feature mask file: {mask_path}")
    keep = load_feature_mask_csv(mask_path, expected_names=names)
    X_k, kept_names = apply_keep_mask(X, names, keep)

    if CLIP_FEATURES_TO_HUMAN_QUANTILES:
        stats_path = PREDICTION_OUT_DIR / f"{out_prefix}_feature_clip_stats.csv"
        if not stats_path.exists():
            raise FileNotFoundError(f"Missing feature clip-stats file: {stats_path}")
        lo, hi = load_feature_clip_stats_csv(stats_path, expected_names=kept_names)
        X_k = clip_X_by_bounds(X_k, lo, hi)

    pred_res = trained_model.predict(X_k)
    pred_raw = pred_res + r23_hat

    pred_raw_cal_fit_all = np.full_like(pred_raw, np.nan, dtype=float)
    pred_raw_cal_oof = np.full_like(pred_raw, np.nan, dtype=float)
    cal_meta: Dict[str, float] = {"cal_enabled": float(int(ENABLE_MONKEY_CALIBRATION))}

    if ENABLE_MONKEY_CALIBRATION and CALIBRATE_SPACE == "raw":
        a, b = _fit_linear_calibrator(y_true=r23, y_pred=pred_raw)
        pred_raw_cal_fit_all = a + b * pred_raw

        groups = df2["id"].astype(str).values
        pred_raw_cal_oof, oof_meta = _oof_linear_calibration_groupkfold(
            y_true=r23,
            y_pred=pred_raw,
            groups=groups,
            n_splits=CALIBRATION_N_SPLITS,
        )

        cal_meta.update({
            "a_fit_all": float(a),
            "b_fit_all": float(b),
        })
        for k, v in oof_meta.items():
            try:
                cal_meta[f"oof_{k}"] = float(v)
            except Exception:
                cal_meta[f"oof_{k}"] = np.nan

        pd.Series({"a": float(a), "b": float(b)}).to_csv(
            PREDICTION_OUT_DIR / f"{out_prefix}_monkey_linear_calibrator_ab.csv"
        )

    met_rows: List[Dict[str, Any]] = []
    met_rows.append({
        "expansion_group": expansion_group,
        "metric": metric,
        "variant": MODEL_VARIANT,
        "target": r23_raw_col,
        "n": int(len(df2)),
        "n_feat_kept": int(np.sum(keep)),
        "clip_feat_to_human_q": int(CLIP_FEATURES_TO_HUMAN_QUANTILES),
        "filter_extreme_rates": int(FILTER_EXTREME_RATES),
        "space": "resid",
        "calibration": "none",
        "r2": float(r2_score(r23_res, pred_res)),
        "pearson_r": safe_corr(r23_res, pred_res, method="pearson"),
        "spearman_r": safe_corr(r23_res, pred_res, method="spearman"),
    })
    met_rows.append({
        "expansion_group": expansion_group,
        "metric": metric,
        "variant": MODEL_VARIANT,
        "target": r23_raw_col,
        "n": int(len(df2)),
        "n_feat_kept": int(np.sum(keep)),
        "clip_feat_to_human_q": int(CLIP_FEATURES_TO_HUMAN_QUANTILES),
        "filter_extreme_rates": int(FILTER_EXTREME_RATES),
        "space": "raw",
        "calibration": "none",
        "r2": float(r2_score(r23, pred_raw)),
        "pearson_r": safe_corr(r23, pred_raw, method="pearson"),
        "spearman_r": safe_corr(r23, pred_raw, method="spearman"),
    })

    if np.isfinite(pred_raw_cal_fit_all).sum() > 0:
        met_rows.append({
            "expansion_group": expansion_group,
            "metric": metric,
            "variant": MODEL_VARIANT,
            "target": r23_raw_col,
            "n": int(len(df2)),
            "n_feat_kept": int(np.sum(keep)),
            "clip_feat_to_human_q": int(CLIP_FEATURES_TO_HUMAN_QUANTILES),
            "filter_extreme_rates": int(FILTER_EXTREME_RATES),
            "space": "raw",
            "calibration": "linear_ab_fit_all",
            "r2": float(r2_score(r23, pred_raw_cal_fit_all)),
            "pearson_r": safe_corr(r23, pred_raw_cal_fit_all, method="pearson"),
            "spearman_r": safe_corr(r23, pred_raw_cal_fit_all, method="spearman"),
        })

    if np.isfinite(pred_raw_cal_oof).sum() > 0:
        ok_oof = np.isfinite(r23) & np.isfinite(pred_raw_cal_oof)
        if ok_oof.sum() >= 3:
            met_rows.append({
                "expansion_group": expansion_group,
                "metric": metric,
                "variant": MODEL_VARIANT,
                "target": r23_raw_col,
                "n": int(ok_oof.sum()),
                "n_feat_kept": int(np.sum(keep)),
                "clip_feat_to_human_q": int(CLIP_FEATURES_TO_HUMAN_QUANTILES),
                "filter_extreme_rates": int(FILTER_EXTREME_RATES),
                "space": "raw",
                "calibration": "linear_ab_oof_groupkfold",
                "r2": float(r2_score(r23[ok_oof], pred_raw_cal_oof[ok_oof])),
                "pearson_r": safe_corr(r23[ok_oof], pred_raw_cal_oof[ok_oof], method="pearson"),
                "spearman_r": safe_corr(r23[ok_oof], pred_raw_cal_oof[ok_oof], method="spearman"),
            })

    met = pd.DataFrame(met_rows)
    for k, v in clip_meta.items():
        met[k] = float(v)
    for k, v in cal_meta.items():
        try:
            met[k] = float(v)
        except Exception:
            met[k] = np.nan

    met.to_csv(PREDICTION_OUT_DIR / f"{out_prefix}_monkey_metrics.csv", index=False)

    out = df2[[
        "id", "scan_1", "scan_2", "scan_3",
        "age_1", "age_2", "age_3",
        "dt12_months_used", "dt23_months_used",
    ] + CONF_COLS].copy()
    out["expansion_group"] = expansion_group
    out["metric"] = metric
    out["variant"] = MODEL_VARIANT

    out["base_T1"] = pd.to_numeric(df2[base_t1_raw_col], errors="coerce").values.astype(float)
    out["base_T2"] = pd.to_numeric(df2[base_t2_raw_col], errors="coerce").values.astype(float)
    if INCLUDE_C_RATIO:
        out["logratio_T2T1"] = pd.to_numeric(df2[ratio_col], errors="coerce").values.astype(float)

    out["r12_raw"] = r12
    out["r12_hat_conf_raw"] = r12_hat_raw
    out["r12_hat_conf"] = r12_hat
    out["r12_resid"] = r12_res

    out["r23_raw"] = r23
    out["r23_hat_conf_raw"] = r23_hat_raw
    out["r23_hat_conf"] = r23_hat
    out["r23_resid"] = r23_res

    out["pred_r23_resid"] = pred_res
    out["pred_r23_raw"] = pred_raw
    out["err_resid"] = r23_res - pred_res
    out["err_raw"] = r23 - pred_raw

    if np.isfinite(pred_raw_cal_fit_all).sum() > 0:
        out["pred_r23_raw_cal_fit_all"] = pred_raw_cal_fit_all
        out["err_raw_cal_fit_all"] = r23 - pred_raw_cal_fit_all
    if np.isfinite(pred_raw_cal_oof).sum() > 0:
        out["pred_r23_raw_cal_oof"] = pred_raw_cal_oof
        out["err_raw_cal_oof"] = r23 - pred_raw_cal_oof

    out.to_csv(PREDICTION_OUT_DIR / f"{out_prefix}_monkey_obs_pred_resid_raw.csv", index=False)
    return met

def run_prediction_pipeline() -> None:
    missing_inputs = [path for path in (HUMAN_CSV, MONKEY_CSV) if not path.exists()]
    if missing_inputs:
        missing_text = "\n".join(str(path) for path in missing_inputs)
        raise FileNotFoundError(f"Missing required input file(s):\n{missing_text}")


    human_long = standardize_human_long(pd.read_csv(HUMAN_CSV))
    monkey_long = standardize_monkey_long(pd.read_csv(MONKEY_CSV))

    human_wide = pivot_long_to_scan_wide_with_confounds(
        df_long=human_long,
        id_col="subject_id",
        scan_col="scan_id",
        age_col="age_months",
        region_col="expansion_group",
        metrics=METRICS,
        conf_cols=CONF_COLS,
    )
    monkey_wide = pivot_long_to_scan_wide_with_confounds(
        df_long=monkey_long,
        id_col="monkey_id",
        scan_col="scan_id",
        age_col="age_months",
        region_col="expansion_group",
        metrics=METRICS,
        conf_cols=CONF_COLS,
    )

    human_triples = build_triples_rate12_to_rate23(
        scan_wide=human_wide,
        id_col="subject_id",
        scan_col="scan_id",
        age_col="age_months",
        metrics=METRICS,
        conf_cols=CONF_COLS,
        dt_scale=1.0,
    )
    monkey_triples = build_triples_rate12_to_rate23(
        scan_wide=monkey_wide,
        id_col="monkey_id",
        scan_col="scan_id",
        age_col="age_months",
        metrics=METRICS,
        conf_cols=CONF_COLS,
        dt_scale=MONKEY_RATE_DT_SCALE,
    )

    human_triples.to_csv(PREDICTION_OUT_DIR / "human_triples_rate12_to_rate23.csv", index=False)
    monkey_triples.to_csv(PREDICTION_OUT_DIR / "monkey_triples_rate12_to_rate23.csv", index=False)

    if human_triples.empty or monkey_triples.empty:
        raise RuntimeError("Triple samples are empty. Need >=3 timepoints per subject and complete expansion_group values.")

    human_age_t2 = pd.to_numeric(human_triples["age_T2"], errors="coerce").values.astype(float)
    monkey_age_t2_equiv = pd.to_numeric(monkey_triples["age_T2"], errors="coerce").values.astype(float) * AGE_RATIO

    lb = float(min(np.nanmin(human_age_t2), np.nanmin(monkey_age_t2_equiv)) - BOUND_BUF)
    ub = float(max(np.nanmax(human_age_t2), np.nanmax(monkey_age_t2_equiv)) + BOUND_BUF)

    spline_spec = fit_spline_spec_from_human_ages(human_age_t2, lower_bound=lb, upper_bound=ub)
    pd.Series({
        "degree": SPLINE_DEGREE,
        "N_SPLINE_BASIS": N_SPLINE_BASIS,
        "N_INTERNAL_KNOTS": N_INTERNAL_KNOTS,
        "lb": spline_spec["lb"],
        "ub": spline_spec["ub"],
        "knots": ",".join([f"{k:.6f}" for k in spline_spec["knots"]]),
        "AGE_RATIO": AGE_RATIO,
        "MONKEY_RATE_DT_SCALE": MONKEY_RATE_DT_SCALE,
        "include_baseline_T2": int(INCLUDE_BASELINE_T2),
        "include_c_ratio": int(INCLUDE_C_RATIO),
        "ratio_eps": float(RATIO_EPS),
        "residualize_baseline_T2": int(RESIDUALIZE_BASELINE_T2),
        "monkey_adjust_sex": int(MONKEY_ADJUST_SEX),
        "monkey_use_human_ref_scanner_site": int(MONKEY_USE_HUMAN_REF_FOR_SCANNER_SITE),
        "confounds": ",".join(CONF_COLS),
        "VAR_STD_MIN": float(VAR_STD_MIN),
        "CONST_STD_MIN": float(CONST_STD_MIN),
        "CLIP_FEATURES_TO_HUMAN_QUANTILES": int(CLIP_FEATURES_TO_HUMAN_QUANTILES),
        "CLIP_LO_Q": float(CLIP_LO_Q),
        "CLIP_HI_Q": float(CLIP_HI_Q),
        "FILTER_EXTREME_RATES": int(FILTER_EXTREME_RATES),
        "RATE_ABS_MAX": float(RATE_ABS_MAX),
        "DT_USED_MIN": float(DT_USED_MIN),
        "ridge_alpha_rate": float(RIDGE_ALPHA_RATE),
        "ridge_alpha_baseline": float(RIDGE_ALPHA_BASELINE),
        "hat_clip_enabled": int(ENABLE_HAT_CLIP),
        "hat_clip_q_lo": float(HAT_CLIP_Q_LO),
        "hat_clip_q_hi": float(HAT_CLIP_Q_HI),
        "hat_clip_fallback_abs": float(HAT_CLIP_FALLBACK_ABS),
        "monkey_calibration_enabled": int(ENABLE_MONKEY_CALIBRATION),
        "calibration_n_splits": int(CALIBRATION_N_SPLITS),
        "calibrate_space": CALIBRATE_SPACE,
    }).to_csv(PREDICTION_OUT_DIR / "spline_spec.csv")

    monkey_triples = monkey_triples.rename(columns={"monkey_id": "id"})

    all_human_metrics: List[pd.DataFrame] = []
    all_monkey_metrics: List[pd.DataFrame] = []

    for expansion_group in EXPANSION_GROUPS:
        for metric in METRICS:
            out_prefix = f"R12toR23_{expansion_group}_{metric}_{FILE_TAG}"

            res = train_single_metric_rate12_to_rate23(
                human_triples=human_triples,
                expansion_group=expansion_group,
                metric=metric,
                spline_spec=spline_spec,
                out_prefix=out_prefix,
            )
            all_human_metrics.append(res["metrics_df"].copy())

            met_mon = apply_single_metric_rate12_to_rate23_to_monkey(
                monkey_triples=monkey_triples,
                expansion_group=expansion_group,
                metric=metric,
                spline_spec=spline_spec,
                trained_model=res["model"],
                trained_rate_residualizer=res["rate_residualizer"],
                trained_baseline_residualizer=res["baseline_residualizer"],
                human_ref_conf=res["human_ref_conf"],
                out_prefix=out_prefix,
            )
            all_monkey_metrics.append(met_mon.copy())

    if all_human_metrics:
        pd.concat(all_human_metrics, axis=0, ignore_index=True).to_csv(
            PREDICTION_OUT_DIR / f"ALL_human_cv_metrics_R12toR23_{FILE_TAG}.csv",
            index=False,
        )
    if all_monkey_metrics:
        pd.concat(all_monkey_metrics, axis=0, ignore_index=True).to_csv(
            PREDICTION_OUT_DIR / f"ALL_monkey_metrics_R12toR23_{FILE_TAG}.csv",
            index=False,
        )

    with open(PREDICTION_OUT_DIR / "README.txt", "w", encoding="utf-8") as f:
        f.write(f"Variant: {MODEL_VARIANT}\n")
        f.write("Expansion groups: low_expansion and high_expansion.\n")
        f.write("Human: fold-wise residualization on rates using confounds at T2.\n")
        predictor_text = "r12_resid + dt23 + age(T2) spline + baseline(T2)"
        if INCLUDE_C_RATIO:
            predictor_text += " + logratio(T2/T1)"
        f.write(
            "Model predicts r23_resid from "
            f"{predictor_text}, with spline interactions.\n"
        )
        f.write("Stability additions used:\n")
        f.write(f"  - A: drop interaction columns with std < {VAR_STD_MIN:g}, and any columns with std < {CONST_STD_MIN:g}\n")
        f.write(f"  - B: clip kept features to human quantiles q=[{CLIP_LO_Q}, {CLIP_HI_Q}] before StandardScaler\n")
        f.write(f"  - C: filter rows by |r12|,|r23| <= {RATE_ABS_MAX:g} and dt12,dt23 >= {DT_USED_MIN:g}\n")
        f.write("Residualizer stabilized: OneHotEncoder(drop='first') + Ridge.\n")
        f.write("Monkey: site/scanner fields can be fixed to human reference levels; sex can be kept or fixed.\n")
        f.write("Monkey: hats are clipped using human hat quantiles to prevent catastrophic raw-space bias.\n")
        f.write("Monkey: optional linear calibration y=a+b*pred_raw fitted on monkey truth; also outputs GroupKFold OOF calibrated preds.\n")
        f.write(f"MONKEY_ADJUST_SEX={int(MONKEY_ADJUST_SEX)}, MONKEY_USE_HUMAN_REF_FOR_SCANNER_SITE={int(MONKEY_USE_HUMAN_REF_FOR_SCANNER_SITE)}\n")
        f.write(f"RIDGE_ALPHA_RATE={float(RIDGE_ALPHA_RATE)}, RIDGE_ALPHA_BASELINE={float(RIDGE_ALPHA_BASELINE)}\n")
        f.write(f"HAT_CLIP: enabled={int(ENABLE_HAT_CLIP)}, q_lo={float(HAT_CLIP_Q_LO)}, q_hi={float(HAT_CLIP_Q_HI)}, fallback_abs={float(HAT_CLIP_FALLBACK_ABS)}\n")
        f.write(f"CALIBRATION: enabled={int(ENABLE_MONKEY_CALIBRATION)}, n_splits={int(CALIBRATION_N_SPLITS)}, space={CALIBRATE_SPACE}\n")

if __name__ == "__main__":
    run_prediction_pipeline()
