"""Inputs: results/behavior_association/*.csv (or CROSS_SPECIES_ASSOCIATION_TABLE_DIR). Output: CROSS_SPECIES_OUTPUT_DIR/figure6/Fig6_colorbar_v2.png."""

from __future__ import annotations

import os
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = Path(os.environ.get("CROSS_SPECIES_OUTPUT_DIR", str(PACKAGE_ROOT / "outputs")))


import hashlib


from typing import Any, Mapping, Sequence

import matplotlib as mpl

import matplotlib.pyplot as plt

from matplotlib.colors import LinearSegmentedColormap

from matplotlib.lines import Line2D

from matplotlib.patches import Rectangle

from matplotlib.transforms import Bbox

import numpy as np

import pandas as pd


REPO_ROOT = PACKAGE_ROOT

INPUT_DIR = Path(os.environ.get('CROSS_SPECIES_ASSOCIATION_TABLE_DIR', str(PACKAGE_ROOT / 'results' / 'behavior_association')))

OUTPUT_DIR = OUTPUT_ROOT / 'figure6'

OUT_PNG = OUTPUT_DIR / "Fig6_colorbar_v2.png"

DPI = 600

RIGHT_MARGIN_MM = 1.0

INPUT_SPECS = {'cognition_primary': {'filename': 'fig6_cognition_primary_associations.csv', 'sha256': 'd0f7d34bc3a005a2e0d701acd7dca76f2455860f604c56264b0937fbf5ff387d', 'rows': 28}, 'cbcl_primary': {'filename': 'fig6_cbcl_primary_associations.csv', 'sha256': 'd358b4179c6b025d1287b4a851ec7d5d2831da9eb1527220e8780ec3e45e0bc3', 'rows': 16}, 'partial_bins': {'filename': 'fig6_primary_partial_bins.csv', 'sha256': 'c46f31bf39b821f58d222710758d76e796adacf4b97a0deb9025d7f2e1ebb7cc', 'rows': 64}, 'partial_lines': {'filename': 'fig6_primary_partial_lines.csv', 'sha256': '259a6f3b83e374a75450240517e5533ccb966ef8dd8b30b2dea6470cf23945fc', 'rows': 800}, 'partial_summary': {'filename': 'fig6_primary_partial_summary.csv', 'sha256': '13718dddeaf26a0526ef44a30399c79dacbed9e4b1009a6be7b03ce52eb44e83', 'rows': 4}, 'joint': {'filename': 'fig6_joint_hsdi_conditional.csv', 'sha256': 'e42af5f7d6aa54aa33b4347c7f6af56fb8ba9d3c0b5bf96610a393404be823a4', 'rows': 16}, 'sensitivity': {'filename': 'fig6_ood_tail_sensitivity.csv', 'sha256': 'baf60e96091c489a2426f48cf261e97283d1a3dedd1ca846ea0e2538fe5b8cdc', 'rows': 16}, 'loso': {'filename': 'fig6_leave_one_site_out.csv', 'sha256': 'a57638df635603c8c5abbcd173a648d97bbec6e54c9ebf29e000d72a3e495136', 'rows': 76}}

ALPHA = 0.05

EPS = 1e-12

HSDI_ORDER: tuple[str, ...] = (
    "HSdevZ_SA_high_expansion",
    "HSdevZ_SA_low_expansion",
    "HSdevZ_CT_high_expansion",
    "HSdevZ_CT_low_expansion",
)

HSDI_LABELS: Mapping[str, str] = {
    "HSdevZ_SA_high_expansion": "SA high-expansion",
    "HSdevZ_SA_low_expansion": "SA low-expansion",
    "HSdevZ_CT_high_expansion": "CT high-expansion",
    "HSdevZ_CT_low_expansion": "CT low-expansion",
}

HSDI_LANDSCAPE_LABELS: Mapping[str, str] = {
    "HSdevZ_SA_high_expansion": "SA\nhigh-expansion",
    "HSdevZ_SA_low_expansion": "SA\nlow-expansion",
    "HSdevZ_CT_high_expansion": "CT\nhigh-expansion",
    "HSdevZ_CT_low_expansion": "CT\nlow-expansion",
}

COGNITION_SPECS: tuple[Mapping[str, Any], ...] = (
    {
        "behavior_file": "nc_y_wisc.csv",
        "merged_file": "MERGED_nc_y_wisc_withHSDI_subinfo_tail.csv",
        "event_key": "baselineYear1Arm1",
        "outcome": "pea_wiscv_tss",
        "short_label": "WISC TSS\nBL",
        "display_label": "Matrix reasoning",
        "domain": "Reasoning",
    },
    {
        "behavior_file": "nc_y_nihtb.csv",
        "merged_file": "MERGED_nc_y_nihtb_withHSDI_subinfo_tail.csv",
        "event_key": "baselineYear1Arm1",
        "outcome": "nihtbx_fluidcomp_fc",
        "short_label": "NIH Fluid\nBL",
        "display_label": "Fluid cognition",
        "domain": "NIH Toolbox",
    },
    {
        "behavior_file": "nc_y_nihtb.csv",
        "merged_file": "MERGED_nc_y_nihtb_withHSDI_subinfo_tail.csv",
        "event_key": "baselineYear1Arm1",
        "outcome": "nihtbx_cryst_fc",
        "short_label": "NIH Cryst\nBL",
        "display_label": "Crystallized cognition at baseline",
        "domain": "NIH Toolbox",
    },
    {
        "behavior_file": "nc_y_nihtb.csv",
        "merged_file": "MERGED_nc_y_nihtb_withHSDI_subinfo_tail.csv",
        "event_key": "2YearFollowUpYArm1",
        "outcome": "nihtbx_cryst_fc",
        "short_label": "NIH Cryst\n2Y",
        "display_label": "Crystallized cognition",
        "domain": "NIH Toolbox",
    },
    {
        "behavior_file": "nc_y_ravlt.csv",
        "merged_file": "MERGED_nc_y_ravlt_withHSDI_subinfo_tail.csv",
        "event_key": "baselineYear1Arm1",
        "outcome": "pea_ravlt_ld_trial_vii_tc",
        "short_label": "RAVLT LD\nBL",
        "display_label": "Long-delay recall at baseline",
        "domain": "Verbal memory",
    },
    {
        "behavior_file": "nc_y_ravlt.csv",
        "merged_file": "MERGED_nc_y_ravlt_withHSDI_subinfo_tail.csv",
        "event_key": "2YearFollowUpYArm1",
        "outcome": "pea_ravlt_ld_trial_vii_tc",
        "short_label": "RAVLT LD\n2Y",
        "display_label": "Long-delay recall at 2-year follow-up",
        "domain": "Verbal memory",
    },
    {
        "behavior_file": "nc_y_smarte.csv",
        "merged_file": "MERGED_nc_y_smarte_withHSDI_subinfo_tail.csv",
        "event_key": "3YearFollowUpYArm1",
        "outcome": "smarte_ss_all_total_corr",
        "short_label": "SMARTE\n3Y",
        "display_label": "Mental arithmetic",
        "domain": "Arithmetic",
    },
)

CBCL_DISPLAY_ORDER: tuple[str, ...] = (
    "cbcl_scr_syn_totprob_r",
    "cbcl_scr_syn_internal_r",
    "cbcl_scr_syn_external_r",
    "cbcl_scr_syn_attention_r",
)

CBCL_LABELS: Mapping[str, tuple[str, str]] = {
    "cbcl_scr_syn_totprob_r": ("CBCL Total", "CBCL total problems"),
    "cbcl_scr_syn_internal_r": ("CBCL Internalizing", "CBCL internalizing problems"),
    "cbcl_scr_syn_external_r": ("CBCL Externalizing", "CBCL externalizing problems"),
    "cbcl_scr_syn_attention_r": ("CBCL Attention", "CBCL attention problems"),
}

REPRESENTATIVE_SPECS: tuple[Mapping[str, str], ...] = (
    {
        "source": "cognition",
        "outcome": "pea_wiscv_tss",
        "event_key": "baselineYear1Arm1",
        "short_label": "WISC BL",
        "display_label": "Matrix reasoning",
    },
    {
        "source": "cognition",
        "outcome": "nihtbx_cryst_fc",
        "event_key": "2YearFollowUpYArm1",
        "short_label": "NIH Cryst 2Y",
        "display_label": "Crystallized cognition",
    },
    {
        "source": "cbcl",
        "outcome": "cbcl_scr_syn_totprob_r",
        "event_key": "POOLED_ANNUAL",
        "short_label": "CBCL Total",
        "display_label": "CBCL total problems",
    },
    {
        "source": "cbcl",
        "outcome": "cbcl_scr_syn_attention_r",
        "event_key": "POOLED_ANNUAL",
        "short_label": "CBCL Attention",
        "display_label": "CBCL attention problems",
    },
)

BETA_CMAP = LinearSegmentedColormap.from_list(
    "fig6_beta",
    ["#4C78A8", "#F7F7F7", "#B2795C"],
)

COGNITION_COLOR = "#58799C"

CBCL_COLOR = "#8B5E78"

SAHIGH_COLOR = "#3B8D6D"

NEUTRAL_COLOR = "#777777"

def load_frozen_tables():
    tables = {}
    for key, spec in INPUT_SPECS.items():
        path = INPUT_DIR / spec["filename"]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != spec["sha256"] and INPUT_DIR.resolve() == (PACKAGE_ROOT / "data" / "fig6").resolve():
            raise ValueError(f"Frozen source changed: {path}")
        table = pd.read_csv(path, float_precision="round_trip")
        if len(table) != spec["rows"]:
            raise ValueError(f"Unexpected row count for {key}: {len(table)}")
        tables[key] = table
    return tables

def configure_publication_style() -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7.0,
            "axes.titlesize": 8.2,
            "axes.titleweight": "bold",
            "axes.labelsize": 7.0,
            "xtick.labelsize": 6.2,
            "ytick.labelsize": 6.2,
            "legend.fontsize": 5.8,
            "axes.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.major.width": 0.55,
            "ytick.major.width": 0.55,
            "xtick.major.size": 2.7,
            "ytick.major.size": 2.7,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "text.usetex": False,
        }
    )

def _style_axis(axis: plt.Axes) -> None:
    axis.spines["left"].set_linewidth(0.6)
    axis.spines["bottom"].set_linewidth(0.6)
    axis.tick_params(axis="both", width=0.55, length=2.7, pad=2)

def marker_size_from_q(q_value: float) -> float:
    q_value = float(q_value) if np.isfinite(q_value) else 1.0
    magnitude = min(3.5, max(0.0, -np.log10(q_value + EPS)))
    return float(np.clip(55.0 + 55.0 * magnitude, 55.0, 250.0))

def format_probability(value: float, symbol: str) -> str:
    if value < 0.001:
        return f"{symbol}<0.001"
    return f"{symbol}={value:.3f}"

def add_panel_letter(figure: plt.Figure, axis: plt.Axes, letter: str) -> None:
    box = axis.get_position()
    figure.text(
        box.x0 - 0.028,
        box.y1 + 0.008,
        letter,
        fontsize=10.0,
        fontweight="bold",
        ha="left",
        va="bottom",
        color="#111111",
    )

def plot_landscape(
    axis: plt.Axes,
    data: pd.DataFrame,
    outcome_order: Sequence[str],
    outcome_labels: Mapping[str, str],
    title: str,
    vmax: float,
    domain_groups: Sequence[tuple[str, int, int, str]] = (),
) -> mpl.cm.ScalarMappable:
    x_positions = {outcome: index for index, outcome in enumerate(outcome_order)}
    y_positions = {hsdi: index for index, hsdi in enumerate(HSDI_ORDER)}
    for hsdi in HSDI_ORDER:
        for outcome in outcome_order:
            x = x_positions[outcome]
            y = y_positions[hsdi]
            axis.add_patch(
                Rectangle(
                    (x - 0.44, y - 0.36),
                    0.88,
                    0.72,
                    facecolor="white",
                    edgecolor="#E4E4E4",
                    linewidth=0.55,
                    zorder=0,
                )
            )
            selected = data[data["outcome"].eq(outcome) & data["hsdi"].eq(hsdi)]
            if len(selected) != 1:
                raise AssertionError(f"Expected one landscape row for {outcome}, {hsdi}")
            row = selected.iloc[0]
            q_value = float(row["q_fdr_family"])
            p_value = float(row["p_value"])
            if q_value < ALPHA:
                edgecolor, linewidth = "#1F1F1F", 0.95
            elif p_value < ALPHA:
                edgecolor, linewidth = "#777777", 0.75
            else:
                edgecolor, linewidth = "white", 0.6
            axis.scatter(
                x,
                y,
                s=marker_size_from_q(q_value),
                c=[float(row["beta_std"])],
                cmap=BETA_CMAP,
                vmin=-vmax,
                vmax=vmax,
                edgecolor=edgecolor,
                linewidth=linewidth,
                zorder=3,
            )
            if q_value < ALPHA:
                axis.text(x + 0.25, y - 0.22, "*", fontsize=7, fontweight="bold")
    for label, start, end, color in domain_groups:
        axis.add_patch(
            Rectangle(
                (start - 0.45, -0.93),
                end - start + 0.90,
                0.28,
                facecolor=color,
                edgecolor="none",
                clip_on=False,
            )
        )
        axis.text(
            (start + end) / 2.0,
            -0.79,
            label,
            ha="center",
            va="center",
            fontsize=5.8,
            color="#333333",
            clip_on=False,
        )
    axis.set_xticks(range(len(outcome_order)))
    axis.set_xticklabels([outcome_labels[outcome] for outcome in outcome_order])
    axis.set_yticks(range(len(HSDI_ORDER)))
    axis.set_yticklabels([HSDI_LANDSCAPE_LABELS[hsdi] for hsdi in HSDI_ORDER])
    for label in axis.get_xticklabels():
        label.set_linespacing(1.0)
    for label in axis.get_yticklabels():
        label.set_linespacing(0.95)
        label.set_multialignment("right")
    axis.invert_yaxis()
    axis.set_xlim(-0.55, len(outcome_order) - 0.45)
    axis.set_ylim(len(HSDI_ORDER) - 0.45, -1.05 if domain_groups else -0.55)
    axis.set_title(title, loc="left", pad=4)
    axis.tick_params(length=0, pad=3)
    for spine in axis.spines.values():
        spine.set_visible(False)
    return mpl.cm.ScalarMappable(
        norm=mpl.colors.Normalize(vmin=-vmax, vmax=vmax),
        cmap=BETA_CMAP,
    )

def plot_partial_panel(
    axis: plt.Axes,
    bins: pd.DataFrame,
    line: pd.DataFrame,
    summary: pd.Series,
    color: str,
    title: str,
    y_label: str,
    show_y_label: bool,
) -> None:
    axis.errorbar(
        bins["x_mean"],
        bins["y_mean"],
        yerr=bins["y_ci95_descriptive"],
        fmt="o",
        ms=3.5,
        mfc=color,
        mec="white",
        mew=0.4,
        ecolor=color,
        elinewidth=0.6,
        capsize=1.2,
        alpha=0.95,
        zorder=4,
    )
    axis.fill_between(
        line["x_partial"],
        line["ci95_low"],
        line["ci95_high"],
        color=color,
        alpha=0.13,
        linewidth=0,
        zorder=1,
    )
    axis.plot(line["x_partial"], line["fitted_y"], color=color, lw=1.15, zorder=3)
    axis.axhline(0, color="#D6D6D6", lw=0.5, zorder=0)
    axis.axvline(0, color="#D6D6D6", lw=0.5, zorder=0)
    axis.set_title(title, loc="left", pad=3)
    axis.set_xlabel("Residualized SA high-expansion\nhuman-specific deviation (z)")
    axis.set_ylabel(y_label if show_y_label else "")
    annotation = (
        f"β={summary['beta_std']:.3f} "
        f"[{summary['ci95_low']:.3f}, {summary['ci95_high']:.3f}]\n"
        f"{format_probability(float(summary['p_value']), 'p')}; "
        f"{format_probability(float(summary['q_fdr_family']), 'q')}"
    )
    axis.text(
        0.03,
        0.97,
        annotation,
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=5.7,
        color="#222222",
    )
    _style_axis(axis)

def plot_joint_panel(axis: plt.Axes, joint: pd.DataFrame) -> None:
    y_positions: list[float] = []
    labels: list[str] = []
    cursor = 0.0
    for relationship_order in range(len(REPRESENTATIVE_SPECS)):
        subset = joint[joint["relationship_order"].eq(relationship_order)].sort_values("hsdi_order")
        for _, row in subset.iterrows():
            y_positions.append(cursor)
            labels.append(f"{row['relationship_short_label']} | {HSDI_LABELS[row['hsdi']].replace('-expansion', '')}")
            color = SAHIGH_COLOR if row["hsdi"] == HSDI_ORDER[0] else NEUTRAL_COLOR
            filled = bool(
                row["hsdi"] == HSDI_ORDER[0]
                and np.isfinite(row["holm_p_sahigh4"])
                and row["holm_p_sahigh4"] < ALPHA
            )
            axis.errorbar(
                row["beta_std"],
                cursor,
                xerr=[[row["beta_std"] - row["ci95_low"]], [row["ci95_high"] - row["beta_std"]]],
                fmt="o",
                ms=3.5,
                mfc=color if filled else "white",
                mec=color,
                mew=0.8,
                ecolor=color,
                elinewidth=0.8,
                capsize=1.3,
            )
            cursor += 0.74
        cursor += 0.35
    axis.axvline(0, color="#888888", lw=0.6)
    axis.set_yticks(y_positions)
    axis.set_yticklabels(labels, fontsize=5.0)
    axis.invert_yaxis()
    axis.set_xlabel("Conditional standardized β (95% CI)")
    axis.set_title("Joint HSDI conditional estimates", loc="left", pad=3)
    _style_axis(axis)

def plot_sensitivity_panel(axis: plt.Axes, sensitivity: pd.DataFrame) -> None:
    labels = [spec["short_label"] for spec in REPRESENTATIVE_SPECS]
    base_y = np.arange(len(labels))[::-1].astype(float)
    styles = {
        "PRIMARY": ("o", "#222222", -0.24, "Primary"),
        "PRIMARY_OOD": ("s", "#4C78A8", -0.08, "+ OOD"),
        "PRIMARY_OOD_TRIM1P": ("^", "#8E6C9E", 0.08, "+ OOD + trim 1%"),
        "PRIMARY_OOD_TRIM5P": ("D", "#B2795C", 0.24, "+ OOD + trim 5%"),
    }
    for relationship_order, y in enumerate(base_y):
        subset = sensitivity[sensitivity["relationship_order"].eq(relationship_order)]
        for variant, (marker, color, offset, _) in styles.items():
            row = subset[subset["model_variant"].eq(variant)].iloc[0]
            axis.errorbar(
                row["beta_std"],
                y + offset,
                xerr=[[row["beta_std"] - row["ci95_low"]], [row["ci95_high"] - row["beta_std"]]],
                fmt=marker,
                ms=3.2,
                mfc="white",
                mec=color,
                mew=0.8,
                ecolor=color,
                elinewidth=0.75,
                capsize=1.2,
            )
    axis.axvline(0, color="#888888", lw=0.6)
    axis.set_yticks(base_y)
    axis.set_yticklabels(labels)
    axis.set_xlabel("Standardized β (95% CI)")
    axis.set_title("OOD and tail sensitivity", loc="left", pad=3)
    _style_axis(axis)
    axis.legend(
        handles=[
            Line2D([0], [0], marker=marker, color=color, mfc="white", lw=0, label=label)
            for marker, color, _, label in styles.values()
        ],
        loc="upper left",
        bbox_to_anchor=(0.01, 0.99),
        fontsize=5.0,
        ncol=1,
        columnspacing=0.6,
        handletextpad=0.3,
    )

def plot_site_panel(axis: plt.Axes, loso: pd.DataFrame) -> None:
    labels = [spec["short_label"] for spec in REPRESENTATIVE_SPECS]
    base_y = np.arange(len(labels))[::-1].astype(float)
    colors = [COGNITION_COLOR, COGNITION_COLOR, CBCL_COLOR, CBCL_COLOR]
    rng = np.random.default_rng(20260828)
    for relationship_order, (y, color) in enumerate(zip(base_y, colors)):
        subset = loso[loso["relationship_order"].eq(relationship_order)].sort_values("site_order")
        beta = subset["beta_std"].to_numpy(dtype=float)
        jitter = rng.uniform(-0.11, 0.11, size=len(beta))
        axis.hlines(y, float(np.min(beta)), float(np.max(beta)), color=color, lw=1.0, alpha=0.75)
        axis.scatter(beta, y + jitter, s=8, facecolor="white", edgecolor=color, linewidth=0.55, alpha=0.8)
        row = subset.iloc[0]
        axis.errorbar(
            row["full_beta_std"],
            y,
            xerr=[
                [row["full_beta_std"] - row["full_ci95_low"]],
                [row["full_ci95_high"] - row["full_beta_std"]],
            ],
            fmt="D",
            ms=3.8,
            mfc=color,
            mec=color,
            ecolor=color,
            elinewidth=0.9,
            capsize=1.4,
            zorder=5,
        )
    axis.axvline(0, color="#888888", lw=0.6)
    axis.set_yticks(base_y)
    axis.set_yticklabels(labels)
    axis.set_xlabel("Leave-one-site-out standardized β")
    axis.set_title("Site stability", loc="left", pad=3)
    _style_axis(axis)

def build_figure(
    cognition: pd.DataFrame,
    cbcl: pd.DataFrame,
    partial_bins: pd.DataFrame,
    partial_lines: pd.DataFrame,
    partial_summary: pd.DataFrame,
    joint: pd.DataFrame,
    sensitivity: pd.DataFrame,
    loso: pd.DataFrame,
):
    configure_publication_style()
    width_mm, height_mm = 180.0, 210.0
    figure = plt.figure(figsize=(width_mm / 25.4, height_mm / 25.4), dpi=300)
    outer = figure.add_gridspec(
        3,
        1,
        left=0.115,
        right=0.94,
        bottom=0.07,
        top=0.96,
        height_ratios=[1.05, 1.18, 1.35],
        hspace=0.48,
    )
    row1 = outer[0].subgridspec(1, 2, width_ratios=[8, 5], wspace=0.28)
    row2 = outer[1].subgridspec(1, 4, wspace=0.42)
    row3 = outer[2].subgridspec(1, 3, width_ratios=[1.32, 1.0, 1.0], wspace=0.48)
    axes = {
        "a": figure.add_subplot(row1[0, 0]),
        "b": figure.add_subplot(row1[0, 1]),
        "c": figure.add_subplot(row2[0, 0]),
        "d": figure.add_subplot(row2[0, 1]),
        "e": figure.add_subplot(row2[0, 2]),
        "f": figure.add_subplot(row2[0, 3]),
        "g": figure.add_subplot(row3[0, 0]),
        "h": figure.add_subplot(row3[0, 1]),
        "i": figure.add_subplot(row3[0, 2]),
    }
    displayed_cbcl = cbcl[cbcl["outcome"].isin(CBCL_DISPLAY_ORDER)].copy()
    vmax = max(
        0.10,
        float(np.nanmax(np.abs(cognition["beta_std"]))),
        float(np.nanmax(np.abs(displayed_cbcl["beta_std"]))),
    )
    cognition_order = [str(spec["outcome"]) + "|" + str(spec["event_key"]) for spec in COGNITION_SPECS]
    cognition_plot = cognition.copy()
    cognition_plot["plot_outcome"] = cognition_plot["outcome"] + "|" + cognition_plot["event_key"]
    cognition_labels = {
        str(spec["outcome"]) + "|" + str(spec["event_key"]): str(spec["short_label"])
        for spec in COGNITION_SPECS
    }
    mappable = plot_landscape(
        axes["a"],
        cognition_plot.rename(columns={"plot_outcome": "outcome_display"}).assign(
            outcome=lambda frame: frame["outcome_display"]
        ),
        cognition_order,
        cognition_labels,
        "Cognitive associations",
        vmax,
        (
            ("Reasoning", 0, 0, "#EAF1FB"),
            ("NIH Toolbox", 1, 3, "#F7EFE8"),
            ("Verbal memory", 4, 5, "#F1F1F1"),
            ("Arithmetic", 6, 6, "#EEF5EE"),
        ),
    )
    plot_landscape(
        axes["b"],
        displayed_cbcl,
        CBCL_DISPLAY_ORDER,
        {outcome: CBCL_LABELS[outcome][0].replace("CBCL ", "") for outcome in CBCL_DISPLAY_ORDER},
        "Parent-reported CBCL associations",
        vmax,
        (("Parent-reported symptoms", 0, len(CBCL_DISPLAY_ORDER) - 1, "#F3ECF1"),),
    )
    for label in axes["b"].get_xticklabels():
        label.set_fontsize(5.7)
    color_box = axes["b"].get_position()
    color_axis = figure.add_axes([color_box.x1 + 0.006, color_box.y0 + 0.035, 0.007, color_box.height - 0.065])
    colorbar = figure.colorbar(mappable, cax=color_axis)
    colorbar.set_label("Standardized β", fontsize=6.2)
    colorbar.ax.tick_params(labelsize=5.6, length=2)
    axes["a"].legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                mfc="white",
                mec="#1F1F1F",
                ms=4.2,
                label="FDR q<0.05",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                mfc="white",
                mec="#777777",
                ms=4.2,
                label="Nominal p<0.05",
            ),
        ],
        loc="upper right",
        bbox_to_anchor=(1.0, 1.18),
        fontsize=5.4,
        ncol=2,
        columnspacing=0.8,
        handletextpad=0.4,
    )

    panel_letters = ("c", "d", "e", "f")
    panel_titles = (
        "Matrix reasoning",
        "Crystallized cognition",
        "CBCL total problems",
        "CBCL attention problems",
    )
    panel_colors = (COGNITION_COLOR, COGNITION_COLOR, CBCL_COLOR, CBCL_COLOR)
    for index, (letter, title, color) in enumerate(zip(panel_letters, panel_titles, panel_colors)):
        bins = partial_bins[partial_bins["relationship_order"].eq(index)]
        line = partial_lines[partial_lines["relationship_order"].eq(index)]
        summary = partial_summary[partial_summary["relationship_order"].eq(index)].iloc[0]
        plot_partial_panel(
            axes[letter],
            bins,
            line,
            summary,
            color,
            title,
            "Residualized cognitive score (z)" if index < 2 else "Residualized CBCL score (z)",
            index in (0, 2),
        )
    cognitive_y = [axes["c"].get_ylim(), axes["d"].get_ylim()]
    cbcl_y = [axes["e"].get_ylim(), axes["f"].get_ylim()]
    for pair, limits in ((("c", "d"), cognitive_y), (("e", "f"), cbcl_y)):
        ymin = min(item[0] for item in limits)
        ymax = max(item[1] for item in limits)
        for letter in pair:
            axes[letter].set_ylim(ymin, ymax)
    x_limits = [axes[letter].get_xlim() for letter in panel_letters]
    xmin = min(item[0] for item in x_limits)
    xmax = max(item[1] for item in x_limits)
    for letter in panel_letters:
        axes[letter].set_xlim(xmin, xmax)

    plot_joint_panel(axes["g"], joint)
    plot_sensitivity_panel(axes["h"], sensitivity)
    plot_site_panel(axes["i"], loso)
    for letter, axis in axes.items():
        add_panel_letter(figure, axis, letter)

    return figure, axes, colorbar

def export_corrected_figure(figure, axes, colorbar):
    figure.set_dpi(DPI)
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    original_width, original_height = map(float, figure.get_size_inches())
    colorbar_bbox = colorbar.ax.get_tightbbox(renderer).transformed(
        figure.dpi_scale_trans.inverted()
    )


    corrected_width = max(
        original_width, colorbar_bbox.x1 + RIGHT_MARGIN_MM / 25.4
    )
    export_bbox = Bbox.from_extents(0.0, 0.0, corrected_width, original_height)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        OUT_PNG, dpi=DPI, facecolor="white",
        bbox_inches=export_bbox, pad_inches=0,
    )
    return {"output": str(OUT_PNG)}

def main():
    tables = load_frozen_tables()
    figure, axes, colorbar = build_figure(
        tables["cognition_primary"], tables["cbcl_primary"],
        tables["partial_bins"], tables["partial_lines"], tables["partial_summary"],
        tables["joint"], tables["sensitivity"], tables["loso"],
    )
    qa = export_corrected_figure(figure, axes, colorbar)
    plt.close(figure)
    return qa

if __name__ == "__main__":
    main()
