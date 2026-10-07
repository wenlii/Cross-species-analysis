"""Inputs: CROSS_SPECIES_PREDICTION_DIR/*.csv. Outputs: CROSS_SPECIES_OUTPUT_DIR/figure5/."""

from __future__ import annotations

import os
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = Path(os.environ.get("CROSS_SPECIES_OUTPUT_DIR", str(PACKAGE_ROOT / "outputs")))
PREDICTION_OUT_DIR = Path(os.environ.get("CROSS_SPECIES_PREDICTION_DIR", str(OUTPUT_ROOT / "prediction_outputs")))
FIGURE_OUT_DIR = OUTPUT_ROOT / "figure5"
MODEL_VARIANT = "baselineT2_noCratio"
FILE_TAG = "noCratio"


from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import pandas as pd

import matplotlib as mpl

import matplotlib.pyplot as plt

import scienceplots


from matplotlib.patches import Rectangle


from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from scipy.stats import fisher_exact, pearsonr, spearmanr, t as student_t

plt.style.use(["science", "nature", "no-latex"])

mpl.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.transparent": False,
    "font.family": ["Arial", "DejaVu Sans"],
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "mathtext.fontset": "dejavusans",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "axes.linewidth": 0.8,
    "axes.labelsize": 8.5,
    "axes.titlesize": 9.0,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "xtick.major.width": 0.75,
    "ytick.major.width": 0.75,
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
    "lines.linewidth": 1.5,
    "lines.markersize": 3.5,
    "legend.frameon": False,
    "legend.fontsize": 7.1,
    "legend.title_fontsize": 7.3,
    "figure.dpi": 150,
    "savefig.dpi": 600,
    "xtick.top": False,
    "ytick.right": False,
    "xtick.minor.top": False,
    "ytick.minor.right": False,
})

PRED_DIR = PREDICTION_OUT_DIR

TRIPLES_CSV = PRED_DIR / "human_triples_rate12_to_rate23.csv"

EXPANSION_GROUPS = ["low_expansion", "high_expansion"]

EPS_SIGN = 0.0

PAD_FRAC = 0.05

QUAD_SCATTER = "#5A8EC1"

QUAD_BAR_COLORS = ["#2E5B88", "#5A8EC1", "#83B1DA", "#B8D0E8"]

LOW_EXPANSION_GRAY = "#9A9A9A"

HUMAN_HIGH_EXPANSION_BLUE = "#4C78A8"

MONKEY_HIGH_EXPANSION_ORANGE = "#B2795C"

REF_LINE_COLOR = "#A8A8A8"

PRED_CORR_KIND = "spearman"

HUMAN_SPACE = "resid"

MONKEY_SPACE = "raw"

HUMAN_EXPECTED_VARIANT = MODEL_VARIANT

MONKEY_EXPECTED_VARIANT = None

AGE_RATIO = 2.83

BIN_WIDTH_MONTHS_HUMAN = 24.0

BIN_WIDTH_MONTHS_MONKEY_EQ = 24.0

MONKEY_MIN_T2_EQ_MONTHS = 120.0

MERGE_LAST_TWO_MONKEY_BINS = False

MIN_BIN_N_HUMAN = 20

MIN_BIN_N_MONKEY = 10

N_BOOT = 100

RANDOM_SEED = 0

SCORE_KIND = "spearman"

WINSORIZE_ERRORS = True

WINSOR_Q_LO = 0.05

WINSOR_Q_HI = 0.95

VIOLIN_XLABEL_ROTATION = 0

VIOLIN_XLABEL_HA = "center"

VIOLIN_SIGMA_LOW_DX = -0.22

VIOLIN_SIGMA_HIGH_DX = 0.22

VIOLIN_SIGMA_Y_FRAC = 0.11

VIOLIN_EXTRA_BOTTOM_FRAC = 0.28

VIOLIN_EXTRA_TOP_FRAC = 0.03

def style_axis(ax: plt.Axes, keep_right: bool = False, keep_top: bool = False) -> None:
    if not keep_top:
        ax.spines["top"].set_visible(False)
    if not keep_right:
        ax.spines["right"].set_visible(False)
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")
    ax.tick_params(
        axis="both",
        which="both",
        direction="out",
        length=3.0,
        width=0.75,
        pad=2,
        top=False,
        right=False,
        labeltop=False,
        labelright=False,
    )

def read_r12_r23(df: pd.DataFrame, metric: str, expansion_group: str) -> Tuple[np.ndarray, np.ndarray]:
    r12 = pd.to_numeric(df[f"{metric}_{expansion_group}_r12"], errors="coerce").astype(float).to_numpy()
    r23 = pd.to_numeric(df[f"{metric}_{expansion_group}_r23"], errors="coerce").astype(float).to_numpy()
    return r12, r23

def sign3(x: np.ndarray, eps: float) -> np.ndarray:
    x = np.asarray(x, float)
    out = np.zeros_like(x, dtype=int)
    out[x > eps] = 1
    out[x < -eps] = -1
    return out

def sign2(x: np.ndarray, eps: float) -> np.ndarray:
    return sign3(x, eps)

def set_limits_with_pad(ax: plt.Axes, x: np.ndarray, y: np.ndarray, same_xy: bool = False) -> None:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    okx = np.isfinite(x)
    oky = np.isfinite(y)
    if same_xy:
        both = np.concatenate([x[okx], y[oky]]) if okx.sum() + oky.sum() > 0 else np.array([])
        if both.size == 0:
            return
        lo = float(np.min(both))
        hi = float(np.max(both))
        pad = PAD_FRAC * (hi - lo + 1e-12)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        return

    if okx.sum() > 0:
        lo = float(np.min(x[okx]))
        hi = float(np.max(x[okx]))
        pad = PAD_FRAC * (hi - lo + 1e-12)
        ax.set_xlim(lo - pad, hi + pad)
    if oky.sum() > 0:
        lo = float(np.min(y[oky]))
        hi = float(np.max(y[oky]))
        pad = PAD_FRAC * (hi - lo + 1e-12)
        ax.set_ylim(lo - pad, hi + pad)

def expansion_group_colors(species: str) -> Tuple[str, str]:
    if species == "Human":
        return LOW_EXPANSION_GRAY, HUMAN_HIGH_EXPANSION_BLUE
    return LOW_EXPANSION_GRAY, MONKEY_HIGH_EXPANSION_ORANGE

def plot_quadrant_panel(ax: plt.Axes, r12: np.ndarray, r23: np.ndarray, title: str) -> None:
    ok = np.isfinite(r12) & np.isfinite(r23)
    r12 = r12[ok]
    r23 = r23[ok]

    ax.scatter(r12, r23, s=8, color=QUAD_SCATTER, alpha=0.36, edgecolors="none", rasterized=True)
    ax.axhline(0.0, linestyle="--", linewidth=0.9, color=REF_LINE_COLOR)
    ax.axvline(0.0, linestyle="--", linewidth=0.9, color=REF_LINE_COLOR)
    ax.set_title(title, pad=4)
    ax.set_xlabel("W1→W2 SPC rate (r12)")
    ax.set_ylabel("W2→W3 SPC rate (r23)")
    style_axis(ax)
    ax.tick_params(axis="both", which="both", top=False, right=False, labeltop=False, labelright=False)

    s12 = sign2(r12, EPS_SIGN)
    s23 = sign2(r23, EPS_SIGN)
    mask = (s12 != 0) & (s23 != 0)
    s12b, s23b = s12[mask], s23[mask]

    a = int(((s12b == 1) & (s23b == 1)).sum())
    b = int(((s12b == 1) & (s23b == -1)).sum())
    c = int(((s12b == -1) & (s23b == 1)).sum())
    d = int(((s12b == -1) & (s23b == -1)).sum())
    table = np.array([[a, b], [c, d]], dtype=float)

    fisher_p = np.nan
    try:
        _, fisher_p = fisher_exact(table)
    except Exception:
        pass

    props = np.array([a, b, c, d], dtype=float)
    props = props / props.sum() if props.sum() > 0 else np.zeros_like(props)

    axins = inset_axes(ax, width="32%", height="32%", loc="upper right", borderpad=0.8)
    bars = axins.bar([0, 1, 2, 3], props, color=QUAD_BAR_COLORS, width=0.8)
    for bbar in bars:
        bbar.set_linewidth(0)
    axins.set_xticks([0, 1, 2, 3])
    axins.set_xticklabels(["++", "+-", "-+", "--"], fontsize=6)
    axins.tick_params(axis="y", labelsize=6, length=2, top=False, right=False, labeltop=False, labelright=False)
    axins.set_title("Quadrants", fontsize=6.2, pad=1.5)
    style_axis(axins)

    if np.isfinite(fisher_p):
        ax.text(
            0.03,
            0.04,
            f"Fisher p={fisher_p:.2e}",
            transform=ax.transAxes,
            fontsize=6.8,
            va="bottom",
            ha="left",
        )

    set_limits_with_pad(ax, r12, r23, same_xy=False)

def safe_spearman_stats(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, int]:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    n = int(ok.sum())
    if n < 3:
        return np.nan, np.nan, n
    xv = x[ok]
    yv = y[ok]
    if np.std(xv) < 1e-12 or np.std(yv) < 1e-12:
        return np.nan, np.nan, n
    res = spearmanr(xv, yv)
    rho = float(res.correlation) if hasattr(res, "correlation") else float(res[0])
    pval = float(res.pvalue) if hasattr(res, "pvalue") else float(res[1])
    return rho, pval, n

def safe_pearson_stats(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, int]:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    n = int(ok.sum())
    if n < 3:
        return np.nan, np.nan, n
    xv = x[ok]
    yv = y[ok]
    if np.std(xv) < 1e-12 or np.std(yv) < 1e-12:
        return np.nan, np.nan, n
    rho, pval = pearsonr(xv, yv)
    return float(rho), float(pval), n

def prediction_corr_stats(x: np.ndarray, y: np.ndarray, kind: str) -> Tuple[float, float, int]:
    if kind == "pearson":
        return safe_pearson_stats(x, y)
    return safe_spearman_stats(x, y)

def linear_fit_mean_ci(
    x: np.ndarray,
    y: np.ndarray,
    x_grid: np.ndarray,
    alpha: float = 0.05,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]
    y = y[ok]
    n = x.size
    if n < 3:
        return None
    x_mean = x.mean()
    y_mean = y.mean()
    sxx = np.sum((x - x_mean) ** 2)
    if sxx <= 0:
        return None
    b1 = np.sum((x - x_mean) * (y - y_mean)) / sxx
    b0 = y_mean - b1 * x_mean
    y_fit = b0 + b1 * x
    resid = y - y_fit
    s_err = np.sqrt(np.sum(resid ** 2) / (n - 2))
    tval = float(student_t.ppf(1.0 - alpha / 2.0, df=n - 2))
    se_mean = s_err * np.sqrt(1.0 / n + (x_grid - x_mean) ** 2 / sxx)
    y_hat = b0 + b1 * x_grid
    ci_low = y_hat - tval * se_mean
    ci_high = y_hat + tval * se_mean
    return y_hat, ci_low, ci_high

def find_human_oof_file(pred_dir: Path, expansion_group: str, metric: str) -> Path:
    p = pred_dir / f"R12toR23_{expansion_group}_{metric}_{FILE_TAG}_human_oof_predictions.csv"
    if p.exists():
        return p
    cand = sorted(pred_dir.glob(f"R12toR23_{expansion_group}_{metric}_*_human_oof_predictions.csv"))
    if len(cand) == 0:
        raise FileNotFoundError(f"Cannot find human OOF file for {expansion_group}-{metric} in {pred_dir}")
    return cand[0]

def find_monkey_pred_file(pred_dir: Path, expansion_group: str, metric: str) -> Path:
    p = pred_dir / f"R12toR23_{expansion_group}_{metric}_{FILE_TAG}_monkey_obs_pred_resid_raw.csv"
    if p.exists():
        return p
    cand = sorted(pred_dir.glob(f"R12toR23_{expansion_group}_{metric}_*_monkey_obs_pred_resid_raw.csv"))
    if len(cand) == 0:
        cand = sorted(pred_dir.glob(f"*{expansion_group}*{metric}*monkey*obs_pred*.csv"))
    if len(cand) == 0:
        raise FileNotFoundError(f"Cannot find monkey prediction file for {expansion_group}-{metric} in {pred_dir}")
    return cand[0]

def load_human_metric_expansion_group(metric: str, expansion_group: str) -> pd.DataFrame:
    f = find_human_oof_file(PRED_DIR, expansion_group, metric)
    df = pd.read_csv(f)
    need = ["metric", "expansion_group", "r23_resid", "pred_r23_resid"]
    for c in need:
        if c not in df.columns:
            raise ValueError(f"Missing '{c}' in {f}")
    return pd.DataFrame({
        "metric": df["metric"].astype(str),
        "expansion_group": df["expansion_group"].astype(str),
        "obs": pd.to_numeric(df["r23_resid"], errors="coerce"),
        "pred": pd.to_numeric(df["pred_r23_resid"], errors="coerce"),
    })

def load_monkey_metric_expansion_group(metric: str, expansion_group: str, space: str) -> pd.DataFrame:
    f = find_monkey_pred_file(PRED_DIR, expansion_group, metric)
    df = pd.read_csv(f)
    if space == "raw":
        obs_col, pred_col = "r23_raw", "pred_r23_raw"
    elif space == "resid":
        obs_col, pred_col = "r23_resid", "pred_r23_resid"
    else:
        raise ValueError("space must be 'raw' or 'resid'")
    need = ["metric", "expansion_group", obs_col, pred_col]
    for c in need:
        if c not in df.columns:
            raise ValueError(f"Missing '{c}' in {f}")
    return pd.DataFrame({
        "metric": df["metric"].astype(str),
        "expansion_group": df["expansion_group"].astype(str),
        "obs": pd.to_numeric(df[obs_col], errors="coerce"),
        "pred": pd.to_numeric(df[pred_col], errors="coerce"),
    })

def safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return np.nan
    if np.std(x[ok]) < 1e-12 or np.std(y[ok]) < 1e-12:
        return np.nan
    return float(spearmanr(x[ok], y[ok]).correlation)

def safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return np.nan
    if np.std(x[ok]) < 1e-12 or np.std(y[ok]) < 1e-12:
        return np.nan
    return float(pearsonr(x[ok], y[ok])[0])

def winsorize_errors(err: np.ndarray, q_lo: float, q_hi: float) -> np.ndarray:
    err = np.asarray(err, float)
    ok = np.isfinite(err)
    if ok.sum() < 3:
        return err
    lo = np.quantile(err[ok], q_lo)
    hi = np.quantile(err[ok], q_hi)
    return np.clip(err, lo, hi)

def score_from_obs_pred(obs: np.ndarray, pred: np.ndarray, kind: str) -> float:
    obs = np.asarray(obs, float)
    pred = np.asarray(pred, float)
    ok = np.isfinite(obs) & np.isfinite(pred)
    obs = obs[ok]
    pred = pred[ok]
    if obs.size < 1:
        return np.nan
    if kind == "spearman":
        return safe_spearman(obs, pred)
    if kind == "pearson":
        return safe_pearson(obs, pred)
    err = obs - pred
    if WINSORIZE_ERRORS:
        err = winsorize_errors(err, WINSOR_Q_LO, WINSOR_Q_HI)
    if kind == "mae":
        return float(np.mean(np.abs(err)))
    if kind == "rmse":
        return float(np.sqrt(np.mean(err ** 2)))
    raise ValueError("kind must be 'spearman', 'pearson', 'mae', or 'rmse'")

def bootstrap_scores(obs: np.ndarray, pred: np.ndarray, kind: str, n_boot: int, rng: np.random.Generator) -> List[float]:
    obs = np.asarray(obs, float)
    pred = np.asarray(pred, float)
    ok = np.isfinite(obs) & np.isfinite(pred)
    obs = obs[ok]
    pred = pred[ok]
    n = obs.size
    if kind in ("spearman", "pearson") and n < 3:
        return []
    if n < 1:
        return []
    scores: List[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        s = score_from_obs_pred(obs[idx], pred[idx], kind)
        if np.isfinite(s):
            scores.append(float(s))
    return scores

def make_age_bins(values: np.ndarray, bin_width: float, min_edge: Optional[float] = None) -> pd.Categorical:
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return pd.Categorical([])
    lo = np.floor(v.min() / bin_width) * bin_width if min_edge is None else float(min_edge)
    vmax = float(np.nanmax(v))
    hi = lo + bin_width if vmax <= lo else lo + int(np.ceil((vmax - lo) / bin_width)) * bin_width
    edges = np.arange(lo, hi + bin_width, bin_width)
    labels = [f"{int(edges[i])}-{int(edges[i + 1])}" for i in range(len(edges) - 1)]
    return pd.cut(values, bins=edges, labels=labels, include_lowest=True, right=False, ordered=True)

def format_num(x: float) -> str:
    if not np.isfinite(x):
        return "nan"
    ax = abs(x)
    if ax >= 1000 or (ax > 0 and ax < 0.001):
        return f"{x:.1e}"
    return f"{x:.3f}"

def _parse_bin_label(label: str) -> Tuple[float, float]:
    parts = str(label).split("-")
    if len(parts) != 2:
        raise ValueError(f"Invalid bin label: {label}")
    return float(parts[0]), float(parts[1])

def merge_last_two_bins(cat: pd.Categorical) -> pd.Categorical:
    if not hasattr(cat, "categories"):
        return cat
    cats = [str(c) for c in cat.categories]
    if len(cats) < 2:
        return cat
    second_last = cats[-2]
    last = cats[-1]
    lo2, _ = _parse_bin_label(second_last)
    _, hi3 = _parse_bin_label(last)
    merged_label = f"{int(lo2)}-{int(hi3)}"
    mapper = {second_last: merged_label, last: merged_label}
    vals = pd.Series(cat.astype(object))
    new_vals = vals.map(lambda v: np.nan if pd.isna(v) else mapper.get(str(v), str(v)))
    new_categories = cats[:-2] + [merged_label]
    return pd.Categorical(new_vals, categories=new_categories, ordered=True)

def pick_obs_pred_cols(df: pd.DataFrame, space: str) -> Tuple[str, str]:
    if space == "resid":
        obs_candidates = ["r23_resid"]
        pred_candidates = ["pred_r23_resid", "pred_r23_res"]
    elif space == "raw":
        obs_candidates = ["r23_raw"]
        pred_candidates = ["pred_r23_raw", "pred_r23"]
    elif space == "raw_cal_oof":
        obs_candidates = ["r23_raw"]
        pred_candidates = ["pred_r23_raw_cal_oof"]
    else:
        raise ValueError("space must be 'raw', 'raw_cal_oof', or 'resid'")
    obs_col = next((c for c in obs_candidates if c in df.columns), None)
    pred_col = next((c for c in pred_candidates if c in df.columns), None)
    if obs_col is None or pred_col is None:
        raise ValueError(f"Cannot find obs/pred columns for space='{space}'.")
    return obs_col, pred_col

def pick_age_col(df: pd.DataFrame) -> str:
    if "age_T2" in df.columns:
        return "age_T2"
    if "age_2" in df.columns:
        return "age_2"
    raise ValueError("Missing age_T2/age_2 in file.")

def apply_variant_filter(df: pd.DataFrame, species: str) -> pd.DataFrame:
    if "variant" not in df.columns:
        return df.copy()
    expected_variant = HUMAN_EXPECTED_VARIANT if species == "Human" else MONKEY_EXPECTED_VARIANT
    if expected_variant is None:
        return df.copy()
    mask = df["variant"].astype(str) == expected_variant
    if mask.any():
        return df.loc[mask].copy()
    return df.copy()

def get_bin_width_and_min_edge(species: str) -> Tuple[float, Optional[float]]:
    if species == "Monkey":
        return BIN_WIDTH_MONTHS_MONKEY_EQ, float(MONKEY_MIN_T2_EQ_MONTHS)
    return BIN_WIDTH_MONTHS_HUMAN, None

def build_panel_data(
    species: str,
    metric: str,
    expansion_group: str,
    df: pd.DataFrame,
    eval_space: str,
    std_space: str,
    min_bin_n: int,
    rng: np.random.Generator,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = apply_variant_filter(df, species)
    age_col = pick_age_col(df)
    age_native = pd.to_numeric(df[age_col], errors="coerce").values.astype(float)
    age_months = np.maximum(age_native * AGE_RATIO, float(MONKEY_MIN_T2_EQ_MONTHS)) if species == "Monkey" else age_native
    obs_col, pred_col = pick_obs_pred_cols(df, eval_space)
    std_obs_col, _ = pick_obs_pred_cols(df, std_space)

    tmp = pd.DataFrame({
        "species": species,
        "metric": metric,
        "expansion_group": expansion_group,
        "age_months": age_months,
        "age_native_months": age_native,
        "obs": pd.to_numeric(df[obs_col], errors="coerce").values.astype(float),
        "pred": pd.to_numeric(df[pred_col], errors="coerce").values.astype(float),
        "y_for_std": pd.to_numeric(df[std_obs_col], errors="coerce").values.astype(float),
    }).copy()

    bin_width, bin_min_edge = get_bin_width_and_min_edge(species)
    tmp["age_bin"] = make_age_bins(tmp["age_months"].values, bin_width=bin_width, min_edge=bin_min_edge)
    if species == "Monkey" and MERGE_LAST_TWO_MONKEY_BINS:
        tmp["age_bin"] = merge_last_two_bins(tmp["age_bin"])

    tmp = tmp.dropna(subset=["age_bin", "obs", "pred", "y_for_std"]).copy()
    bin_counts = tmp.groupby("age_bin", observed=True).size().rename("n").reset_index()
    good_bins = set(bin_counts.loc[bin_counts["n"] >= min_bin_n, "age_bin"].astype(str).tolist())
    tmp = tmp[tmp["age_bin"].astype(str).isin(good_bins)].copy()

    violin_rows: List[Dict[str, Any]] = []
    anno_rows: List[Dict[str, Any]] = []
    cats = tmp["age_bin"].cat.categories
    for b in cats:
        b_str = str(b)
        if b_str not in good_bins:
            continue
        sub = tmp[tmp["age_bin"] == b].copy()
        scores = bootstrap_scores(sub["obs"].values, sub["pred"].values, SCORE_KIND, N_BOOT, rng)
        native_age = sub["age_native_months"].to_numpy(dtype=float)
        native_age = native_age[np.isfinite(native_age)]
        if native_age.size == 0:
            raise RuntimeError(f"No finite native ages for {species}-{metric}, {b_str}")
        for s in scores:
            violin_rows.append({
                "species": species,
                "metric": metric,
                "expansion_group": expansion_group,
                "age_bin": b_str,
                "score": float(s),
                "n_in_bin": int(sub.shape[0]),
            })
        y_std = float(np.std(sub["y_for_std"].values[np.isfinite(sub["y_for_std"].values)], ddof=1)) if sub.shape[0] >= 2 else np.nan
        anno_rows.append({
            "species": species,
            "metric": metric,
            "expansion_group": expansion_group,
            "age_bin": b_str,
            "n_in_bin": int(sub.shape[0]),
            "std_y": y_std,
            "age_native_min_months": float(np.min(native_age)),
            "age_native_max_months": float(np.max(native_age)),
            "age_human_equivalent_min_months": float(np.min(native_age) * AGE_RATIO),
            "age_human_equivalent_max_months": float(np.max(native_age) * AGE_RATIO),
        })
    return pd.DataFrame(violin_rows), pd.DataFrame(anno_rows)

def build_violin_tables() -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(RANDOM_SEED)
    long_parts: List[pd.DataFrame] = []
    anno_parts: List[pd.DataFrame] = []

    for metric in ["SA", "CT"]:
        for expansion_group in EXPANSION_GROUPS:
            human_df = pd.read_csv(find_human_oof_file(PRED_DIR, expansion_group, metric))
            ld, ad = build_panel_data(
                species="Human",
                metric=metric,
                expansion_group=expansion_group,
                df=human_df,
                eval_space=HUMAN_SPACE,
                std_space=HUMAN_SPACE,
                min_bin_n=MIN_BIN_N_HUMAN,
                rng=rng,
            )
            if not ld.empty:
                long_parts.append(ld)
            if not ad.empty:
                anno_parts.append(ad)

            monkey_df = pd.read_csv(find_monkey_pred_file(PRED_DIR, expansion_group, metric))
            ld, ad = build_panel_data(
                species="Monkey",
                metric=metric,
                expansion_group=expansion_group,
                df=monkey_df,
                eval_space=MONKEY_SPACE,
                std_space="raw",
                min_bin_n=MIN_BIN_N_MONKEY,
                rng=rng,
            )
            if not ld.empty:
                long_parts.append(ld)
            if not ad.empty:
                anno_parts.append(ad)

    if not long_parts or not anno_parts:
        raise RuntimeError("No violin data available after binning/filtering.")
    return pd.concat(long_parts, ignore_index=True), pd.concat(anno_parts, ignore_index=True)

def score_ylabel(kind: str) -> str:
    if kind == "spearman":
        return "Spearman r (obs vs pred)"
    if kind == "pearson":
        return "Pearson r (obs vs pred)"
    if kind == "mae":
        return "MAE"
    if kind == "rmse":
        return "RMSE"
    return kind

def _horizontal_intersections(vertices: np.ndarray, y0: float) -> np.ndarray:
    xs = []
    for i in range(len(vertices) - 1):
        x1, y1 = vertices[i]
        x2, y2 = vertices[i + 1]
        if not (min(y1, y2) <= y0 <= max(y1, y2)):
            continue
        if y1 == y2:
            xs.extend([x1, x2])
            continue
        t = (y0 - y1) / (y2 - y1)
        if 0.0 <= t <= 1.0:
            xs.append(x1 + t * (x2 - x1))
    if not xs:
        return np.array([])
    xs = np.unique(np.round(xs, 6))
    return np.sort(xs)

def _violin_line_extent(body, pos: float, y0: float, side: str, width: float) -> Tuple[float, float]:
    verts = body.get_paths()[0].vertices
    xs = _horizontal_intersections(verts, y0)
    if xs.size < 2:
        if side == "left":
            return pos - width * 0.34, pos - width * 0.08
        return pos + width * 0.08, pos + width * 0.34

    x_left = float(xs.min())
    x_right = float(xs.max())
    if side == "left":
        span = max(pos - x_left, width * 0.02)
        return x_left + 0.12 * span, pos - 0.12 * span
    span = max(x_right - pos, width * 0.02)
    return pos + 0.12 * span, x_right - 0.12 * span

def _violin_half(
    ax: plt.Axes,
    datasets: List[np.ndarray],
    positions: np.ndarray,
    color: str,
    side: str,
    width: float = 0.88,
) -> None:
    parts = ax.violinplot(
        datasets,
        positions=positions,
        widths=width,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    for body, pos, arr in zip(parts["bodies"], positions, datasets):
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.40)
        body.set_linewidth(0.8)
        if side == "left":
            clip = Rectangle((pos - width / 2.0, -1e6), width / 2.0, 2e6, transform=ax.transData)
        else:
            clip = Rectangle((pos, -1e6), width / 2.0, 2e6, transform=ax.transData)
        body.set_clip_path(clip)

        finite = np.asarray(arr, float)
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            continue
        q1, med, q3 = np.percentile(finite, [25, 50, 75])
        for qv, lw in [(q1, 0.9), (med, 1.2), (q3, 0.9)]:
            x0, x1 = _violin_line_extent(body, pos, qv, side, width)
            ax.plot([x0, x1], [qv, qv], color=color, linewidth=lw, solid_capstyle="round")

def plot_split_violin_panel(
    ax: plt.Axes,
    df_long: pd.DataFrame,
    df_anno: pd.DataFrame,
    species: str,
    metric: str,
) -> None:
    low_color, high_color = expansion_group_colors(species)

    d = df_long[(df_long["species"] == species) & (df_long["metric"] == metric)].copy()
    a = df_anno[(df_anno["species"] == species) & (df_anno["metric"] == metric)].copy()
    if d.empty:
        ax.axis("off")
        return

    bin_order = sorted(d["age_bin"].unique(), key=lambda s: float(str(s).split("-")[0]))
    positions = np.arange(1, len(bin_order) + 1, dtype=float)

    low_data = []
    high_data = []
    for b in bin_order:
        low_data.append(d[(d["age_bin"] == b) & (d["expansion_group"] == "low_expansion")]["score"].to_numpy(dtype=float))
        high_data.append(d[(d["age_bin"] == b) & (d["expansion_group"] == "high_expansion")]["score"].to_numpy(dtype=float))

    _violin_half(ax, low_data, positions, low_color, side="left", width=0.90)
    _violin_half(ax, high_data, positions, high_color, side="right", width=0.90)

    if SCORE_KIND in ("spearman", "pearson"):
        ax.axhline(0.0, linestyle="--", linewidth=0.9, color=REF_LINE_COLOR, zorder=0)

    ax.set_xticks(positions)
    ax.set_xticklabels(bin_order, rotation=VIOLIN_XLABEL_ROTATION, ha=VIOLIN_XLABEL_HA)
    ax.set_xlim(0.4, len(bin_order) + 0.6)
    ax.set_xlabel("W2 age bin (human-equivalent months)")
    ax.set_ylabel(score_ylabel(SCORE_KIND))
    ax.set_title(f"{species} | {metric}", pad=4)
    style_axis(ax)

    base_y_min, base_y_max = ax.get_ylim()
    base_y_range = max(base_y_max - base_y_min, 1e-12)
    extra_bottom = VIOLIN_EXTRA_BOTTOM_FRAC * base_y_range
    extra_top = VIOLIN_EXTRA_TOP_FRAC * base_y_range
    ax.set_ylim(base_y_min - extra_bottom, base_y_max + extra_top)

    y_text = base_y_min - VIOLIN_SIGMA_Y_FRAC * base_y_range
    for i, b in enumerate(bin_order, start=1):
        suba = a[a["age_bin"] == b]
        for expansion_group, dx in [("low_expansion", VIOLIN_SIGMA_LOW_DX), ("high_expansion", VIOLIN_SIGMA_HIGH_DX)]:
            row = suba[suba["expansion_group"] == expansion_group]
            if row.empty:
                continue
            sd = float(row["std_y"].iloc[0]) if np.isfinite(row["std_y"].iloc[0]) else np.nan
            nbin = int(row["n_in_bin"].iloc[0])
            ax.text(
                i + dx,
                y_text,
                f"σ={format_num(sd)}\nn={nbin}",
                ha="center",
                va="top",
                fontsize=5.8,
                color="#444444",
                linespacing=0.95,
                clip_on=False,
            )


from matplotlib.ticker import MaxNLocator

from datetime import datetime, timezone

import json

from hashlib import sha256

from PIL import Image


FIG5_0826_DPI = 600

FIG5_0826_WIDTH_MM = 180.0

FIG5_0826_HEIGHT_MM = 255.0

FIG5_0826_WIDTH_PX = round(FIG5_0826_WIDTH_MM / 25.4 * FIG5_0826_DPI)

FIG5_0826_HEIGHT_PX = round(FIG5_0826_HEIGHT_MM / 25.4 * FIG5_0826_DPI)

FIG5_0826_SIZE_IN = (
    FIG5_0826_WIDTH_PX / FIG5_0826_DPI,
    np.nextafter(FIG5_0826_HEIGHT_PX / FIG5_0826_DPI, np.inf),
)

FIG5_0826_STEM = "FIG5_human_reference_spc_transition_clean_labels_v3_20260908"

FIG5_0826_PNG = Path(FIGURE_OUT_DIR) / f"{FIG5_0826_STEM}.png"

FIG5_0826_AUDIT = Path(FIGURE_OUT_DIR) / f"{FIG5_0826_STEM}_audit.json"

COLOR_HUMAN_HIGH_0826 = "#4C78A8"

COLOR_HUMAN_LOW_0826 = "#96A6BD"

COLOR_MACAQUE_HIGH_0826 = "#B2795C"

COLOR_MACAQUE_LOW_0826 = "#B9A79E"

COLOR_TEXT_0826 = "#202020"

COLOR_AXIS_0826 = "#333333"

mpl.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.transparent": False,
        "font.family": ["Arial", "DejaVu Sans"],
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "mathtext.fontset": "dejavusans",
        "font.size": 7.0,
        "axes.titlesize": 7.8,
        "axes.labelsize": 7.0,
        "xtick.labelsize": 6.4,
        "ytick.labelsize": 6.4,
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.top": False,
        "ytick.right": False,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "svg.hashsalt": "cross-species-fig5-0826",
    }
)

def fig5_0826_group_color(species: str, expansion_group: str) -> str:
    if species == "Human":
        return (
            COLOR_HUMAN_LOW_0826
            if expansion_group == "low_expansion"
            else COLOR_HUMAN_HIGH_0826
        )
    return (
        COLOR_MACAQUE_LOW_0826
        if expansion_group == "low_expansion"
        else COLOR_MACAQUE_HIGH_0826
    )

def fig5_0826_violin_colors(species: str):
    return (
        fig5_0826_group_color(species, "low_expansion"),
        fig5_0826_group_color(species, "high_expansion"),
    )

def fig5_0826_style_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
        spine.set_color(COLOR_AXIS_0826)
    ax.tick_params(
        axis="both",
        which="major",
        direction="out",
        labelsize=6.4,
        width=0.55,
        length=2.6,
        pad=2.0,
        top=False,
        right=False,
    )

def fig5_0826_load_data():
    triples = pd.read_csv(TRIPLES_CSV)
    required_quadrant_columns = ["id"]
    for metric in ("CT", "SA"):
        for expansion_group in ("low_expansion", "high_expansion"):
            required_quadrant_columns.extend(
                [
                    f"{metric}_{expansion_group}_r12",
                    f"{metric}_{expansion_group}_r23",
                ]
            )

    missing_columns = [
        column for column in required_quadrant_columns if column not in triples.columns
    ]
    if missing_columns:
        raise ValueError("Missing quadrant-panel columns: " + ", ".join(missing_columns))

    input_rows = int(len(triples))
    null_counts = {
        column: int(triples[column].isna().sum())
        for column in required_quadrant_columns
    }
    complete_quadrant_mask = triples[required_quadrant_columns].notna().all(axis=1)
    triples = triples.loc[complete_quadrant_mask].copy()
    duplicate_id_rows = int(triples["id"].duplicated(keep=False).sum())
    if duplicate_id_rows:
        raise ValueError(
            f"Expected one human triple per subject, found {duplicate_id_rows} "
            "rows belonging to duplicated subject IDs"
        )

    human_prediction = pd.concat(
        [
            load_human_metric_expansion_group(metric, expansion_group)
            for metric in ("CT", "SA")
            for expansion_group in ("low_expansion", "high_expansion")
        ],
        ignore_index=True,
    )
    macaque_prediction = pd.concat(
        [
            load_monkey_metric_expansion_group(
                metric,
                expansion_group,
                MONKEY_SPACE,
            )
            for metric in ("CT", "SA")
            for expansion_group in ("low_expansion", "high_expansion")
        ],
        ignore_index=True,
    )
    violin_long, violin_annotation = build_violin_tables()

    data_audit = {
        "human_triple_rows_before_complete_case_filter": input_rows,
        "human_triple_rows_after_complete_case_filter": int(len(triples)),
        "human_triple_id_nulls": int(triples["id"].isna().sum()),
        "human_triple_duplicate_id_rows": duplicate_id_rows,
        "quadrant_column_null_counts_before_filter": null_counts,
        "joins_performed": False,
        "subject_key": "id",
        "quadrant_axes": ["r12", "r23"],
    }
    return {
        "triples": triples,
        "human_prediction": human_prediction,
        "macaque_prediction": macaque_prediction,
        "violin_long": violin_long,
        "violin_annotation": violin_annotation,
        "audit": data_audit,
    }

def fig5_0826_superscript_integer(value: int) -> str:
    return str(value).translate(str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹"))

def fig5_0826_format_scientific_p(value: float) -> str:
    if not np.isfinite(value):
        return "P = NA"
    if value <= 0:
        return "P < 10⁻³⁰⁰"
    exponent = int(np.floor(np.log10(value)))
    mantissa = value / (10.0 ** exponent)
    return f"P = {mantissa:.2f} × 10{fig5_0826_superscript_integer(exponent)}"

def fig5_0826_quadrant_fisher_p(r12, r23) -> float:
    r12 = np.asarray(r12, dtype=float)
    r23 = np.asarray(r23, dtype=float)
    finite = np.isfinite(r12) & np.isfinite(r23)
    s12 = sign2(r12[finite], EPS_SIGN)
    s23 = sign2(r23[finite], EPS_SIGN)
    keep = (s12 != 0) & (s23 != 0)
    s12 = s12[keep]
    s23 = s23[keep]
    table = np.array(
        [
            [
                ((s12 == 1) & (s23 == 1)).sum(),
                ((s12 == 1) & (s23 == -1)).sum(),
            ],
            [
                ((s12 == -1) & (s23 == 1)).sum(),
                ((s12 == -1) & (s23 == -1)).sum(),
            ],
        ],
        dtype=float,
    )
    try:
        return float(fisher_exact(table)[1])
    except Exception:
        return float("nan")

def fig5_0826_plot_quadrant_panel(
    ax: plt.Axes,
    triples: pd.DataFrame,
    metric: str,
    expansion_group: str,
    title: str,
):
    r12, r23 = read_r12_r23(triples, metric, expansion_group)
    finite = np.isfinite(r12) & np.isfinite(r23)
    r12 = np.asarray(r12[finite], dtype=float)
    r23 = np.asarray(r23[finite], dtype=float)

    before = set(ax.figure.axes)
    original_quad_scatter = globals()["QUAD_SCATTER"]
    globals()["QUAD_SCATTER"] = fig5_0826_group_color("Human", expansion_group)
    try:
        plot_quadrant_panel(ax, r12, r23, title)
    finally:
        globals()["QUAD_SCATTER"] = original_quad_scatter

    added = [
        axis
        for axis in ax.figure.axes
        if axis not in before and axis is not ax
    ]
    if len(added) != 1:
        raise RuntimeError(
            f"Expected one quadrant inset for {title}, found {len(added)}"
        )
    inset = added[0]

    ax.figure.canvas.draw()
    parent = ax.get_position()
    inset.set_axes_locator(None)
    inset.set_position(
        [
            parent.x0 + 0.560 * parent.width,
            parent.y0 + 0.610 * parent.height,
            0.400 * parent.width,
            0.320 * parent.height,
        ]
    )
    inset.set_title("Quadrant proportion", fontsize=5.4, pad=1.3)
    inset.set_xticks([])
    inset.tick_params(
        axis="x",
        which="both",
        bottom=False,
        top=False,
        labelbottom=False,
    )

    bars = sorted(
        [
            patch
            for patch in inset.patches
            if isinstance(patch, mpl.patches.Rectangle)
            and np.isfinite(patch.get_height())
            and patch.get_width() > 0
            and patch.get_height() >= 0
        ],
        key=lambda patch: patch.get_x() + patch.get_width() / 2.0,
    )
    quadrant_symbols = ("↑↑", "↑↓", "↓↑", "↓↓")
    if len(bars) != len(quadrant_symbols):
        raise RuntimeError(
            f"Expected four quadrant bars for {title}, found {len(bars)}"
        )
    if bars:
        y_bottom, y_top = inset.get_ylim()
        max_height = max(float(bar.get_height()) for bar in bars)
        occupied_height = max(max_height - float(y_bottom), np.finfo(float).eps)
        inset.set_ylim(
            float(y_bottom),
            max(float(y_top), max_height + 0.32 * occupied_height),
        )
    for bar, symbol in zip(bars, quadrant_symbols):
        inset.annotate(
            symbol,
            xy=(bar.get_x() + bar.get_width() / 2.0, bar.get_height()),
            xytext=(0.0, 1.2),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=5.2,
            color=COLOR_TEXT_0826,
            clip_on=False,
            annotation_clip=False,
        )
    inset.yaxis.set_major_locator(MaxNLocator(2))
    inset.tick_params(
        axis="y",
        which="major",
        labelsize=5.0,
        width=0.45,
        length=1.7,
        pad=1.0,
        right=False,
    )
    for spine in inset.spines.values():
        spine.set_linewidth(0.5)
        spine.set_color(COLOR_AXIS_0826)

    fisher_texts = [
        text for text in ax.texts if text.get_text().startswith("Fisher p=")
    ]
    if len(fisher_texts) != 1:
        raise RuntimeError(
            f"Expected one Fisher annotation for {title}, found {len(fisher_texts)}"
        )
    fisher_p = fig5_0826_quadrant_fisher_p(r12, r23)
    fisher_texts[0].set_text(
        fig5_0826_format_scientific_p(fisher_p)
    )
    fisher_texts[0].set_position((0.03, 0.035))
    fisher_texts[0].set_fontsize(5.6)
    fisher_texts[0].set_fontfamily("DejaVu Sans")
    fisher_texts[0].set_linespacing(0.95)

    ax.set_title("", loc="center")
    ax.set_title("", loc="right")
    ax.set_title(title, loc="left", fontweight="bold", pad=4.0, fontsize=7.8)
    ax.set_xlabel("")
    ax.set_ylabel("")
    fig5_0826_style_axis(ax)
    return {
        "n": int(r12.size),
        "r12": r12,
        "r23": r23,
        "fisher_p": fisher_p,
        "inset": inset,
    }

def fig5_0826_format_p(value: float) -> str:
    if not np.isfinite(value):
        return "P = NA"
    if value < 0.01:
        return "P < 0.01"
    return f"P = {value:.3f}"

def fig5_0826_prediction_xy(
    frame: pd.DataFrame,
    expansion_group: str,
):
    subset = frame[frame["expansion_group"].eq(expansion_group)]
    x = pd.to_numeric(subset["obs"], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(subset["pred"], errors="coerce").to_numpy(dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    return x[keep], y[keep]

def fig5_0826_plot_prediction_panel(
    ax: plt.Axes,
    frame: pd.DataFrame,
    species: str,
    metric: str,
    expansion_group: str,
):
    x, y = fig5_0826_prediction_xy(frame, expansion_group)
    color = fig5_0826_group_color(species, expansion_group)
    display_species = "Macaque" if species == "Monkey" else species

    ax.scatter(
        x,
        y,
        s=9.0,
        color=color,
        alpha=0.40,
        edgecolors="none",
        rasterized=True,
        zorder=2,
    )
    if x.size >= 3 and np.std(x) > 0:
        x_grid = np.linspace(float(np.min(x)), float(np.max(x)), 200)
        fit = linear_fit_mean_ci(x, y, x_grid, alpha=0.05)
        if fit is not None:
            y_hat, ci_low, ci_high = fit
            ax.fill_between(
                x_grid,
                ci_low,
                ci_high,
                color=color,
                alpha=0.13,
                linewidth=0.0,
                zorder=1,
            )
            ax.plot(x_grid, y_hat, color=color, linewidth=1.35, zorder=3)

    rho, p_value, n_value = prediction_corr_stats(x, y, PRED_CORR_KIND)
    symbol = "r" if PRED_CORR_KIND == "pearson" else "ρ"
    ax.text(
        0.03,
        0.96,
        f"{symbol} = {rho:.2f}\n{fig5_0826_format_p(p_value)}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=5.6,
        linespacing=1.05,
        color=COLOR_TEXT_0826,
    )
    expansion_label = {
        "low_expansion": "Low expansion",
        "high_expansion": "High expansion",
    }[expansion_group]
    title_lines = [f"{display_species} · {metric}", expansion_label]
    ax.set_title("", loc="right")
    ax.set_title(
        "\n".join(title_lines),
        loc="left",
        fontweight="bold",
        pad=3.0,
        fontsize=6.9,
        linespacing=0.94,
    )
    if species == "Monkey":
        right_title = ax.set_title(
            "Human→\nmacaque\nprediction",
            loc="right",
            fontweight="regular",
            pad=3.0,
            fontsize=5.0,
            linespacing=0.94,
            color="#555555",
        )
        right_title.set_x(1.10)
    fig5_0826_style_axis(ax)
    return {
        "n": int(n_value),
        "rho": float(rho),
        "p": float(p_value),
        "x": x,
        "y": y,
    }

def fig5_0826_padded_limits(values, pad_fraction=0.05):
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return (-1.0, 1.0)
    low = float(np.min(finite))
    high = float(np.max(finite))
    span = max(high - low, 1e-12)
    return (low - pad_fraction * span, high + pad_fraction * span)

def fig5_0826_format_age_range(low: float, high: float) -> str:
    if not (np.isfinite(low) and np.isfinite(high)):
        raise ValueError("Age-range endpoints must be finite")
    return f"{low:.0f}–{high:.0f}"

def fig5_0826_apply_macaque_native_age_ticks(
    ax: plt.Axes,
    violin_annotation: pd.DataFrame,
    metric: str,
):
    panel_annotation = violin_annotation[
        violin_annotation["species"].eq("Monkey")
        & violin_annotation["metric"].eq(metric)
    ].copy()
    required_columns = {
        "age_native_min_months",
        "age_native_max_months",
        "age_human_equivalent_min_months",
        "age_human_equivalent_max_months",
    }
    missing = sorted(required_columns.difference(panel_annotation.columns))
    if missing:
        raise ValueError("Missing macaque age-axis columns: " + ", ".join(missing))

    age_bins = [tick.get_text() for tick in ax.get_xticklabels()]
    tick_labels = []
    audit_rows = []
    for age_bin in age_bins:
        rows = panel_annotation[panel_annotation["age_bin"].astype(str).eq(age_bin)]
        if len(rows) != 2:
            raise RuntimeError(
                f"Expected low/high rows for Monkey-{metric}, {age_bin}; "
                f"found {len(rows)}"
            )
        native_low = float(rows["age_native_min_months"].min())
        native_high = float(rows["age_native_max_months"].max())
        human_low = float(rows["age_human_equivalent_min_months"].min())
        human_high = float(rows["age_human_equivalent_max_months"].max())
        native_label = fig5_0826_format_age_range(native_low, native_high)
        display_label = native_label
        tick_labels.append(display_label)
        audit_rows.append(
            {
                "analysis_bin_human_equivalent_months": age_bin,
                "macaque_actual_months": [native_low, native_high],
                "human_equivalent_months": [human_low, human_high],
                "display_label": display_label,
            }
        )

    positions = np.arange(1, len(age_bins) + 1, dtype=float)
    ax.set_xticks(positions)
    ax.set_xticklabels(tick_labels, rotation=0, ha="center", fontsize=6.2)
    ax.tick_params(axis="x", pad=1.2)
    return audit_rows

def fig5_0826_polish_violin_annotation(
    ax: plt.Axes,
    violin_long: pd.DataFrame,
    violin_annotation: pd.DataFrame,
    species: str,
    metric: str,
) -> None:

    for text in tuple(ax.texts):
        if text.get_text().startswith(("σ=", "SD =", "n =")):
            text.remove()

def fig5_0826_add_panel_letters(fig, axes, panel_rows):
    letters = {}
    for row in panel_rows:
        row_top = max(axes[panel].get_position().y1 for panel in row)
        letter_y = row_top + 0.010
        for panel in row:
            x_value = axes[panel].get_position().x0 - 0.026
            letters[panel] = fig.text(
                x_value,
                letter_y,
                panel,
                ha="left",
                va="bottom",
                fontsize=10.2,
                fontweight="bold",
                color="black",
            )
    return letters

def fig5_0826_file_record(path: Path):
    path = Path(path).resolve()
    stat = path.stat()
    return {
        "path": str(path),
        "bytes": int(stat.st_size),
        "sha256": sha256(path.read_bytes()).hexdigest(),
        "modified_utc": datetime.fromtimestamp(
            stat.st_mtime, timezone.utc
        ).isoformat(),
    }

def fig5_0826_input_paths():
    paths = [Path(TRIPLES_CSV)]
    for metric in ("CT", "SA"):
        for expansion_group in ("low_expansion", "high_expansion"):
            paths.append(find_human_oof_file(PRED_DIR, expansion_group, metric))
            paths.append(find_monkey_pred_file(PRED_DIR, expansion_group, metric))
    unique = []
    seen = set()
    for path in paths:
        resolved = Path(path).resolve()
        key = str(resolved).lower()
        if key not in seen:
            if not resolved.is_file() or resolved.stat().st_size == 0:
                raise FileNotFoundError(f"Missing or empty Figure 5 input: {resolved}")
            unique.append(resolved)
            seen.add(key)
    return unique

def build_fig5_0826():
    data = fig5_0826_load_data()
    fig = plt.figure(figsize=FIG5_0826_SIZE_IN, constrained_layout=False)
    content_left = 0.090
    content_right = 0.965

    row1 = fig.add_gridspec(
        1, 4, left=content_left, right=content_right,
        bottom=0.825647059, top=0.946725490, wspace=0.50,
    )
    row2 = fig.add_gridspec(
        1, 4, left=content_left, right=content_right,
        bottom=0.638607843, top=0.759686275, wspace=0.50,
    )
    row3 = fig.add_gridspec(
        1, 4, left=content_left, right=content_right,
        bottom=0.451568627, top=0.572647059, wspace=0.50,
    )
    row4 = fig.add_gridspec(
        1, 2, left=content_left, right=content_right,
        bottom=0.277121569, top=0.377858824, wspace=0.30,
    )
    row5 = fig.add_gridspec(
        1, 2, left=content_left, right=content_right,
        bottom=0.082333333, top=0.203411765, wspace=0.30,
    )

    axes = {}
    quadrant_specs = [
        ("a", "CT", "low_expansion", "CT · Low expansion"),
        ("b", "CT", "high_expansion", "CT · High expansion"),
        ("c", "SA", "low_expansion", "SA · Low expansion"),
        ("d", "SA", "high_expansion", "SA · High expansion"),
    ]
    quadrant_context = {}
    for column, (panel, metric, expansion_group, title) in enumerate(quadrant_specs):
        axes[panel] = fig.add_subplot(row1[0, column])
        quadrant_context[panel] = fig5_0826_plot_quadrant_panel(
            axes[panel], data["triples"], metric, expansion_group, title
        )

    prediction_columns = [
        ("Human", "CT"),
        ("Human", "SA"),
        ("Monkey", "CT"),
        ("Monkey", "SA"),
    ]
    prediction_context = {}
    prediction_panels = {
        "low_expansion": tuple("efgh"),
        "high_expansion": tuple("ijkl"),
    }
    for row_grid, expansion_group in (
        (row2, "low_expansion"),
        (row3, "high_expansion"),
    ):
        for column, (species, metric) in enumerate(prediction_columns):
            panel = prediction_panels[expansion_group][column]
            axes[panel] = fig.add_subplot(row_grid[0, column])
            source = (
                data["human_prediction"]
                if species == "Human"
                else data["macaque_prediction"]
            )
            metric_frame = source[source["metric"].eq(metric)].copy()
            prediction_context[panel] = fig5_0826_plot_prediction_panel(
                axes[panel], metric_frame, species, metric, expansion_group
            )

    for column, _ in enumerate(prediction_columns):
        low_panel = prediction_panels["low_expansion"][column]
        high_panel = prediction_panels["high_expansion"][column]
        x_values = np.concatenate(
            [
                prediction_context[low_panel]["x"],
                prediction_context[high_panel]["x"],
            ]
        )
        y_values = np.concatenate(
            [
                prediction_context[low_panel]["y"],
                prediction_context[high_panel]["y"],
            ]
        )
        x_limits = fig5_0826_padded_limits(x_values)
        y_limits = fig5_0826_padded_limits(y_values)
        for panel in (low_panel, high_panel):
            axes[panel].set_xlim(*x_limits)
            axes[panel].set_ylim(*y_limits)

    violin_specs = [
        ("m", row4[0, 0], "Human", "CT", "Human · CT"),
        ("n", row4[0, 1], "Monkey", "CT", "Macaque · CT"),
        ("o", row5[0, 0], "Human", "SA", "Human · SA"),
        ("p", row5[0, 1], "Monkey", "SA", "Macaque · SA"),
    ]
    macaque_age_axis_context = {}
    original_color_function = globals()["expansion_group_colors"]
    globals()["expansion_group_colors"] = fig5_0826_violin_colors
    try:
        for panel, grid_cell, species, metric, title in violin_specs:
            axes[panel] = fig.add_subplot(grid_cell)
            plot_split_violin_panel(
                axes[panel],
                data["violin_long"],
                data["violin_annotation"],
                species=species,
                metric=metric,
            )
            axes[panel].set_title("", loc="center")
            axes[panel].set_title("", loc="right")
            panel_title = title
            if species == "Monkey":
                panel_title = f"{title}\nCross-species prediction"
            axes[panel].set_title(
                panel_title,
                loc="left",
                fontweight="bold",
                pad=3.5,
                fontsize=7.8,
                linespacing=0.95,
            )
            axes[panel].set_xlabel("")
            axes[panel].set_ylabel("")
            fig5_0826_style_axis(axes[panel])
            fig5_0826_polish_violin_annotation(
                axes[panel],
                data["violin_long"],
                data["violin_annotation"],
                species,
                metric,
            )
            if species == "Monkey":
                macaque_age_axis_context[panel] = (
                    fig5_0826_apply_macaque_native_age_ticks(
                        axes[panel], data["violin_annotation"], metric
                    )
                )
    finally:
        globals()["expansion_group_colors"] = original_color_function

    for panel in tuple("abcdefghijklmnop"):
        axes[panel].set_xlabel("")
        axes[panel].set_ylabel("")

    fig.text(
        0.024, 0.885701961, "W2→W3 SPC rate (r23)", rotation=90,
        ha="center", va="center", rotation_mode="anchor",
        fontsize=7.0, color=COLOR_TEXT_0826,
    )
    for panel in tuple("abcd"):
        axes[panel].set_xlabel(
            "W1→W2\nSPC rate (r12)",
            fontsize=6.2, labelpad=2.5, color=COLOR_TEXT_0826,
            linespacing=1.05,
        )
    fig.text(
        0.024, 0.605627451, "Predicted W2→W3 SPC rate (r23)", rotation=90,
        ha="center", va="center", rotation_mode="anchor",
        fontsize=7.0, color=COLOR_TEXT_0826,
    )

    for panel in tuple("efghijkl"):
        axes[panel].set_xlabel(
            "Observed W2→W3\nSPC rate (r23)",
            fontsize=6.2, labelpad=2.5, color=COLOR_TEXT_0826,
            linespacing=1.05,
        )
    fig.text(
        0.024, 0.319741176, "Spearman ρ (observed vs. predicted)", rotation=90,
        ha="center", va="center", rotation_mode="anchor",
        fontsize=7.0, color=COLOR_TEXT_0826,
    )
    for panel, y_value, label in (
        ("m", 0.247094118, "Human W2 age bin (months)"),
        (
            "n", 0.247094118,
            "Macaque W2 age bin (months)",
        ),
    ):
        center_x = (axes[panel].get_position().x0 + axes[panel].get_position().x1) / 2
        fig.text(
            center_x, y_value, label, ha="center", va="top",
            fontsize=6.4, color=COLOR_TEXT_0826,
        )
    fig.text(
        0.024, 0.142388235, "Spearman ρ (observed vs. predicted)", rotation=90,
        ha="center", va="center", rotation_mode="anchor",
        fontsize=7.0, color=COLOR_TEXT_0826,
    )
    for panel, y_value, label in (
        ("o", 0.048431373, "Human W2 age bin (months)"),
        (
            "p", 0.048431373,
            "Macaque W2 age bin (months)",
        ),
    ):
        center_x = (axes[panel].get_position().x0 + axes[panel].get_position().x1) / 2
        fig.text(
            center_x, y_value, label, ha="center", va="top",
            fontsize=6.4, color=COLOR_TEXT_0826,
        )

    panel_rows = (
        tuple("abcd"),
        tuple("efgh"),
        tuple("ijkl"),
        tuple("mn"),
        tuple("op"),
    )
    letters = fig5_0826_add_panel_letters(fig, axes, panel_rows)

    legend_handles = [
        mpl.lines.Line2D(
            [0], [0], marker="o", linestyle="", markersize=4.0,
            color=COLOR_HUMAN_HIGH_0826, label="Human, high expansion",
        ),
        mpl.lines.Line2D(
            [0], [0], marker="o", linestyle="", markersize=4.0,
            color=COLOR_HUMAN_LOW_0826, label="Human, low expansion",
        ),
        mpl.lines.Line2D(
            [0], [0], marker="o", linestyle="", markersize=4.0,
            color=COLOR_MACAQUE_HIGH_0826, label="Macaque, high expansion",
        ),
        mpl.lines.Line2D(
            [0], [0], marker="o", linestyle="", markersize=4.0,
            color=COLOR_MACAQUE_LOW_0826, label="Macaque, low expansion",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.017435294),
        ncol=4,
        frameon=False,
        fontsize=6.4,
        handlelength=1.0,
        handletextpad=0.4,
        columnspacing=1.0,
    )

    fig.canvas.draw()

    audit = {
        "design": "fig5_0826_quadrant_a_d_and_split_prediction_2x4",
        "core_conclusion": (
            "Human transition organization is shown in r12-by-r23 quadrants, "
            "while low/high expansion prediction performance is shown in "
            "separate rows; macaque panels are explicitly labeled as "
            "human-to-macaque predictions and display native macaque age "
            "ranges only."
        ),
        "archetype": "quantitative grid",
        "backend": "python",
        "canvas_mm": [FIG5_0826_WIDTH_MM, FIG5_0826_HEIGHT_MM],
        "canvas_px": [FIG5_0826_WIDTH_PX, FIG5_0826_HEIGHT_PX],
        "panel_order": list("abcdefghijklmnop"),
        "prediction_layout": "2 rows x 4 columns",
        "prediction_row_order": ["low_expansion", "high_expansion"],
        "prediction_column_order": [
            "Human_CT", "Human_SA", "Macaque_CT", "Macaque_SA"
        ],
        "first_row_design": (
            "r12-by-r23 quadrant scatter with quadrant-proportion inset"
        ),
        "quadrant_axes": ["r12", "r23"],
        "quadrant_subject_counts": {
            panel: quadrant_context[panel]["n"] for panel in tuple("abcd")
        },
        "quadrant_fisher_p": {
            panel: quadrant_context[panel]["fisher_p"]
            for panel in tuple("abcd")
        },
        "prediction_statistics": {
            panel: {
                "n": prediction_context[panel]["n"],
                "rho": prediction_context[panel]["rho"],
                "p": prediction_context[panel]["p"],
            }
            for panel in tuple("efghijkl")
        },
        "data_checks": data["audit"],
        "upstream_covariates": [
            "sex", "site", "scanner manufacturer", "scanner model"
        ],
        "train_validation_logic": (
            "Human predictions are existing subject-grouped 5-fold out-of-fold "
            "predictions; macaque predictions are existing applications of the "
            "human longitudinal reference. No refit is performed."
        ),
        "scientific_values_changed": False,
        "quadrant_x_label_panels": list("abcd"),
        "inter_row_gap_increase_mm": 2.0,
        "data_panel_physical_dimensions_preserved": True,
        "violin_sd_n_displayed": False,
        "figure_footer_displayed": False,
        "quadrant_test_name_displayed": False,
        "quadrant_p_values_displayed": True,
        "prediction_x_label_panels": list("efghijkl"),
        "prediction_x_label_pad_pt": 2.5,
        "violin_sd_n_values": data["violin_annotation"].to_dict(orient="records"),
        "prediction_split_is_layout_only": True,
        "prediction_title_structure": {
            "left_line_1": "species and metric",
            "left_line_2": "low or high expansion",
            "right_line_1": "Human→",
            "right_line_2": "macaque",
            "right_line_3": "prediction",
            "right_descriptor_panels": list("ghkl"),
            "right_title_x": 1.10,
        },
        "cross_species_prediction_title_panels": list("ghkl"),
        "violin_cross_species_prediction_title_panels": ["n", "p"],
        "prediction_stat_annotation": {
            "line_order": ["spearman_rho", "p_value"],
            "n_displayed": False,
            "n_retained_in_audit": True,
        },
        "x_axis_caption_y": {
            "m": 0.247094118, "n": 0.247094118, "o": 0.048431373, "p": 0.048431373,
        },
        "macaque_age_axis": {
            "panels": ["n", "p"],
            "tick_line_order": ["macaque_actual_months"],
            "human_equivalent_age_displayed": False,
            "x_axis_caption": "Macaque W2 age bin (months)",
            "age_ratio": float(AGE_RATIO),
            "bin_membership_changed": False,
            "panel_labels": macaque_age_axis_context,
        },
        "panel_letter_positions": {
            panel: list(letters[panel].get_position()) for panel in letters
        },
    }
    return fig, axes, audit

def save_fig5_0826(fig, audit):
    Path(FIGURE_OUT_DIR).mkdir(parents=True, exist_ok=True)
    output_paths = {
        "png": FIG5_0826_PNG,


    }
    common = {
        "bbox_inches": fig.bbox_inches,
        "pad_inches": 0,
        "facecolor": "white",
        "transparent": False,
    }
    with mpl.rc_context({"savefig.bbox": None, "savefig.pad_inches": 0}):
        fig.savefig(output_paths["png"], dpi=FIG5_0826_DPI, **common)


    for path in output_paths.values():
        if not Path(path).is_file() or Path(path).stat().st_size == 0:
            raise RuntimeError(f"Missing or empty Figure 5 output: {path}")

    with Image.open(output_paths["png"]) as image:
        if image.size != (FIG5_0826_WIDTH_PX, FIG5_0826_HEIGHT_PX):
            raise RuntimeError(
                f"Unexpected PNG geometry: {image.size}; expected "
                f"{(FIG5_0826_WIDTH_PX, FIG5_0826_HEIGHT_PX)}"
            )
        png_geometry = {
            "width_px": int(image.width),
            "height_px": int(image.height),
            "corner_rgba": list(image.convert("RGBA").getpixel((0, 0))),
        }

    audit = dict(audit)
    audit["inputs"] = {
        str(path): fig5_0826_file_record(path)
        for path in fig5_0826_input_paths()
    }
    audit["png_geometry"] = png_geometry
    audit["outputs"] = {
        key: fig5_0826_file_record(path)
        for key, path in output_paths.items()
    }
    FIG5_0826_AUDIT.write_text(
        json.dumps(audit, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return output_paths, audit

if __name__ == "__main__":
    figure, axes, audit = build_fig5_0826()
    save_fig5_0826(figure, audit)
    plt.close(figure)
