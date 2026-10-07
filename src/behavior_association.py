"""Inputs: ABCD_CORE_DIR/, CROSS_SPECIES_COGNITION_DIR/, CROSS_SPECIES_DEMOGRAPHICS_FILE. Outputs: CROSS_SPECIES_OUTPUT_DIR/."""

from __future__ import annotations

import os
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = Path(os.environ.get("CROSS_SPECIES_OUTPUT_DIR", str(PACKAGE_ROOT / "outputs")))
COGNITION_DIR = Path(os.environ.get("CROSS_SPECIES_COGNITION_DIR", str(OUTPUT_ROOT / "cognition_inputs")))
DEMOGRAPHICS_FILE = Path(os.environ.get("CROSS_SPECIES_DEMOGRAPHICS_FILE", str(PACKAGE_ROOT / "data" / "private" / "subinfo_event_covariates_with_demographics_grouped.xlsx")))
CBCL_REFERENCE_FILE = PACKAGE_ROOT / "results" / "behavior_association" / "demographic_sensitivity_results.csv"


import gc

import hashlib

import importlib.util

import json

import platform

import shutil

import sys

from dataclasses import dataclass

from datetime import datetime, timezone


from typing import Any, Mapping, Sequence

import matplotlib as mpl

import matplotlib.pyplot as plt

from matplotlib.colors import LinearSegmentedColormap

from matplotlib.lines import Line2D

from matplotlib.patches import Rectangle

import numpy as np

import pandas as pd

import scipy

import statsmodels

import statsmodels.formula.api as smf

from statsmodels.stats.multitest import multipletests

from statsmodels.stats.outliers_influence import variance_inflation_factor

SCRIPT_VERSION = "1.2.1"

RUN_NAME = "fig6_cognition_cbcl_race_income_v1"

FIGURE_STEM = "Fig6_cognition_cbcl_race_income_v1"

ALPHA = 0.05

EPS = 1e-12

MIN_SUBJECTS = 150

MIN_FAMILIES = 100

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

OOD_RAW_COLUMNS: tuple[str, ...] = (
    "OOD_max_abs_z_max",
    "OOD_frac_q_max",
)

OOD_Z_COLUMNS: tuple[str, ...] = (
    "z_OOD_max_abs_z_max",
    "z_OOD_frac_q_max",
)

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

CBCL_VARIABLES: tuple[str, ...] = (
    "cbcl_scr_syn_totprob_r",
    "cbcl_scr_syn_internal_r",
    "cbcl_scr_syn_external_r",
    "cbcl_scr_syn_attention_r",
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

SENSITIVITY_VARIANTS: tuple[tuple[str, bool, float | None], ...] = (
    ("PRIMARY", False, None),
    ("PRIMARY_OOD", True, None),
    ("PRIMARY_OOD_TRIM1P", True, 0.01),
    ("PRIMARY_OOD_TRIM5P", True, 0.05),
)

@dataclass(frozen=True)
class AnalysisPaths:
    repo_root: Path
    fig6_input_dir: Path
    disease_dir: Path
    demographics_xlsx: Path
    cbcl_engine: Path
    cbcl_reference_results: Path
    output_dir: Path
    figure_main_dir: Path

    @classmethod
    def from_root(cls, repo_root: Path, run_name: str = RUN_NAME) -> "AnalysisPaths":
        return cls(
            repo_root=Path(repo_root).resolve(),
            fig6_input_dir=COGNITION_DIR,
            disease_dir=PACKAGE_ROOT / "results" / "behavior_association",
            demographics_xlsx=DEMOGRAPHICS_FILE,
            cbcl_engine=Path(__file__).with_name("cbcl_inputs.py"),
            cbcl_reference_results=CBCL_REFERENCE_FILE,
            output_dir=OUTPUT_ROOT / run_name,
            figure_main_dir=OUTPUT_ROOT / "figures",
        )


@dataclass
class FitBundle:
    metadata: Mapping[str, Any]
    frame: pd.DataFrame
    fit: Any
    formula: str
    outcome_term: str
    predictor_terms: tuple[str, ...]
    covariate_terms: tuple[str, ...]
    cluster_column: str
    omitted_redundant_covariates: tuple[str, ...]
    n_removed_tail_subjects: int
    trim_threshold: float | None

def locate_repo_root(start: Path | None = None) -> Path:
    return PACKAGE_ROOT if start is None else Path(start).resolve()

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def normalize_id(series: pd.Series) -> pd.Series:
    values = series.astype("string").str.strip()
    values = values.str.replace(r"^sub-", "", regex=True)
    invalid = values.isna() | values.eq("") | values.str.lower().isin(("nan", "none", "<na>"))
    return values.mask(invalid)

def clean_category(series: pd.Series) -> pd.Series:
    values = series.astype("string").str.strip()
    invalid = values.isna() | values.eq("") | values.str.lower().isin(("nan", "none", "<na>"))
    return values.mask(invalid).astype(object)

def mode_or_na(series: pd.Series) -> Any:
    values = series.dropna()
    if values.empty:
        return np.nan
    return values.mode(dropna=True).iloc[0]

def zscore(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").astype(float)
    mean = float(np.nanmean(values))
    sd = float(np.nanstd(values, ddof=0))
    if not np.isfinite(sd) or sd <= EPS:
        return pd.Series(np.nan, index=series.index, dtype=float)
    return (values - mean) / sd

def validate_required_paths(paths: AnalysisPaths) -> None:
    required = [
        paths.fig6_input_dir / "HSDI_subjectLevel_fromByTriple_core4.csv",
        paths.demographics_xlsx,
        paths.cbcl_engine,
        paths.cbcl_reference_results,
        *[paths.fig6_input_dir / str(spec["merged_file"]) for spec in COGNITION_SPECS],
    ]
    missing = sorted({str(path) for path in required if not Path(path).is_file()})
    if missing:
        raise FileNotFoundError("Missing required Figure 6 inputs:\n" + "\n".join(missing))

def load_hsdi(paths: AnalysisPaths) -> tuple[pd.DataFrame, dict[str, Any]]:
    source = paths.fig6_input_dir / "HSDI_subjectLevel_fromByTriple_core4.csv"
    frame = pd.read_csv(source, low_memory=False)
    required = ["id", *HSDI_ORDER, *OOD_RAW_COLUMNS]
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"HSDI input is missing columns: {missing}")
    frame = frame[required].copy()
    frame["id"] = normalize_id(frame["id"])
    if frame["id"].isna().any() or frame["id"].duplicated().any():
        raise ValueError("HSDI input must contain unique, nonmissing normalized IDs")
    for column in [*HSDI_ORDER, *OOD_RAW_COLUMNS]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        if frame[column].isna().any():
            raise ValueError(f"HSDI input contains missing values in {column}")
    for column in HSDI_ORDER:
        frame[f"z_{column}"] = zscore(frame[column])
    frame["z_OOD_max_abs_z_max"] = zscore(frame["OOD_max_abs_z_max"])
    frame["z_OOD_frac_q_max"] = zscore(frame["OOD_frac_q_max"])
    qc = {
        "source": "human_specific_deviation",
        "path": str(source.resolve()),
        "rows": int(len(frame)),
        "unique_ids": int(frame["id"].nunique()),
        "duplicate_ids": int(frame["id"].duplicated().sum()),
        "null_ids": int(frame["id"].isna().sum()),
        "sha256": sha256_file(source),
    }
    return frame, qc

def load_demographics(
    paths: AnalysisPaths,
    expected_ids: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    required_sheets = {"Covariates", "Codebook"}
    with pd.ExcelFile(paths.demographics_xlsx, engine="openpyxl") as workbook:
        missing_sheets = sorted(required_sheets - set(workbook.sheet_names))
        if missing_sheets:
            raise ValueError(f"Demographic workbook is missing sheets: {missing_sheets}")
        raw = pd.read_excel(workbook, sheet_name="Covariates")
    required = {"id", "eventname", "race_ethnicity", "family_income_grouped"}
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"Demographic workbook is missing columns: {missing}")
    frame = raw[["id", "eventname", "race_ethnicity", "family_income_grouped"]].copy()
    frame["id"] = normalize_id(frame["id"])
    if frame["id"].isna().any():
        raise ValueError("Demographic workbook contains invalid normalized IDs")
    if frame.duplicated(["id", "eventname"]).any():
        raise ValueError("Demographic workbook contains duplicate subject-event rows")
    contracts = {
        "race_ethnicity": {1, 2, 3, 4, 5},
        "family_income_grouped": {0, 1, 2, 3},
    }
    for column, allowed in contracts.items():
        numeric = pd.to_numeric(frame[column], errors="coerce")
        if (numeric.dropna() % 1 != 0).any():
            raise ValueError(f"{column} contains noninteger codes")
        unexpected = sorted(set(numeric.dropna().astype(int)) - allowed)
        if unexpected:
            raise ValueError(f"{column} contains unexpected codes: {unexpected}")
        frame[column] = numeric
    income_nonresponse_rows = int(frame["family_income_grouped"].eq(0).sum())
    income_nonresponse_subjects = int(
        frame.loc[frame["family_income_grouped"].eq(0), "id"].nunique()
    )
    frame.loc[frame["family_income_grouped"].eq(0), "family_income_grouped"] = np.nan
    for column in ("race_ethnicity", "family_income_grouped"):
        within_id_levels = frame.groupby("id", observed=True)[column].nunique(dropna=True)
        changing = int(within_id_levels.gt(1).sum())
        if changing:
            raise ValueError(f"{column} varies within {changing} subjects")
    static = (
        frame.groupby("id", as_index=False, observed=True)
        .agg(
            race_ethnicity=("race_ethnicity", mode_or_na),
            family_income_grouped=("family_income_grouped", mode_or_na),
        )
    )
    overlap = set(static.loc[static["id"].isin(expected_ids), "id"].astype(str))
    if overlap != expected_ids:
        raise ValueError(
            "Demographic subject coverage is incomplete for the HSDI cohort: "
            f"{len(overlap)}/{len(expected_ids)}"
        )
    static = static[static["id"].isin(expected_ids)].copy()
    level_rows: list[dict[str, Any]] = []
    for column in ("race_ethnicity", "family_income_grouped"):
        counts = static[column].astype("string").fillna("<MISSING>").value_counts().sort_index()
        for level, count in counts.items():
            level_rows.append(
                {
                    "variable": column,
                    "coded_level": str(level),
                    "n_subjects": int(count),
                }
            )
    qc = {
        "source": "grouped_demographics",
        "path": str(paths.demographics_xlsx.resolve()),
        "raw_rows": int(len(raw)),
        "raw_unique_ids": int(frame["id"].nunique()),
        "hsdi_overlap_ids": int(len(overlap)),
        "hsdi_join_coverage_pct": 100.0,
        "race_missing_subjects": int(static["race_ethnicity"].isna().sum()),
        "income_missing_subjects": int(static["family_income_grouped"].isna().sum()),
        "income_nonresponse_rows_recoded_missing": income_nonresponse_rows,
        "income_nonresponse_subjects": income_nonresponse_subjects,
        "sha256": sha256_file(paths.demographics_xlsx),
    }
    return static, pd.DataFrame(level_rows), qc

def load_cognition_frames(
    paths: AnalysisPaths,
    hsdi: pd.DataFrame,
    demographics: pd.DataFrame,
) -> tuple[dict[tuple[str, str], pd.DataFrame], pd.DataFrame]:
    cache: dict[str, pd.DataFrame] = {}
    outputs: dict[tuple[str, str], pd.DataFrame] = {}
    qc_rows: list[dict[str, Any]] = []
    hsdi_columns = [
        "id",
        *HSDI_ORDER,
        *(f"z_{column}" for column in HSDI_ORDER),
        *OOD_RAW_COLUMNS,
        *OOD_Z_COLUMNS,
    ]
    for spec in COGNITION_SPECS:
        merged_file = str(spec["merged_file"])
        if merged_file not in cache:
            cache[merged_file] = pd.read_csv(
                paths.fig6_input_dir / merged_file,
                low_memory=False,
            )
        raw = cache[merged_file]
        required = {
            "id",
            "eventname",
            str(spec["outcome"]),
            "age_subinfo_years",
            "site",
            "sex",
            "scanner_manufacturer",
        }
        missing = sorted(required - set(raw.columns))
        if missing:
            raise ValueError(f"{merged_file} is missing columns: {missing}")
        frame = raw.loc[
            raw["eventname"].astype(str).eq(str(spec["event_key"])),
            list(required),
        ].copy()
        frame["id"] = normalize_id(frame["id"])
        if frame.duplicated(["id", "eventname"]).any():
            raise ValueError(f"Duplicate cognition subject-event rows for {spec}")
        before = int(len(frame))
        frame = frame.merge(hsdi[hsdi_columns], on="id", how="inner", validate="one_to_one")
        hsdi_joined = int(len(frame))
        frame = frame.merge(
            demographics[["id", "race_ethnicity"]],
            on="id",
            how="left",
            validate="one_to_one",
            indicator=True,
        )
        demographic_coverage = float(frame["_merge"].eq("both").mean() * 100.0)
        frame = frame.drop(columns="_merge")
        if demographic_coverage != 100.0:
            raise ValueError(f"Cognition demographic join coverage is {demographic_coverage:.3f}%")
        frame["outcome_raw"] = pd.to_numeric(frame[str(spec["outcome"])], errors="coerce")
        frame["outcome_z"] = zscore(frame["outcome_raw"])
        frame["age_cov"] = zscore(frame["age_subinfo_years"])
        frame["event_key"] = str(spec["event_key"])
        frame["family_cluster"] = frame["id"].astype(str)
        frame["analysis_site"] = clean_category(frame["site"])
        outputs[(str(spec["event_key"]), str(spec["outcome"]))] = frame
        qc_rows.append(
            {
                "source": "cognition",
                "event_key": spec["event_key"],
                "outcome": spec["outcome"],
                "candidate_rows": before,
                "hsdi_joined_rows": hsdi_joined,
                "hsdi_join_coverage_pct": 100.0 * hsdi_joined / max(before, 1),
                "demographic_join_coverage_pct": demographic_coverage,
                "nonmissing_outcome_rows": int(frame["outcome_raw"].notna().sum()),
                "unique_ids": int(frame["id"].nunique()),
            }
        )
    return outputs, pd.DataFrame(qc_rows)

def _import_cbcl_engine(path: Path):
    module_spec = importlib.util.spec_from_file_location(
        "fig6_cbcl_input_engine",
        path,
    )
    if module_spec is None or module_spec.loader is None:
        raise ImportError(f"Could not import CBCL engine from {path}")
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_spec.name] = module
    module_spec.loader.exec_module(module)
    return module

def load_cbcl_long(
    paths: AnalysisPaths,
    demographics: pd.DataFrame,
    expected_hsdi_ids: set[str],
) -> tuple[pd.DataFrame, tuple[Mapping[str, Any], ...], pd.DataFrame]:
    engine = _import_cbcl_engine(paths.cbcl_engine)
    engine_specs = tuple(
        dict(spec)
        for spec in engine.OUTCOME_SPECS
        if str(spec["instrument"]) == "CBCL"
    )
    attention_spec = {
        "instrument": "CBCL",
        "reporter": "Parent",
        "path": engine.CBCL_INPUT,
        "variable": "cbcl_scr_syn_attention_r",
        "label": "CBCL attention problems",
        "short_label": "CBCL: Attention",
        "primary_events": tuple(engine.ANNUAL_EVENTS),
        "sensitivity_events": (),
    }
    cbcl_specs = (*engine_specs, attention_spec)
    if tuple(str(spec["variable"]) for spec in cbcl_specs) != CBCL_VARIABLES:
        raise AssertionError("Unexpected CBCL outcome order in the source engine")
    engine.OUTCOME_SPECS = cbcl_specs
    engine.OUTCOME_ORDER = tuple(spec["variable"] for spec in cbcl_specs)
    engine.OUTCOME_LOOKUP = {str(spec["variable"]): spec for spec in cbcl_specs}

    def selected_source_specs():
        return (
            (
                "CBCL",
                "Parent",
                engine.CBCL_INPUT,
                tuple(str(spec["variable"]) for spec in cbcl_specs),
            ),
        )

    engine.instrument_source_specs = selected_source_specs
    cohort, hsdi_qc = engine.load_hsdi()
    cohort_ids = set(cohort["id"].astype(str))
    if cohort_ids != expected_hsdi_ids:
        raise ValueError("CBCL engine and Figure 6 HSDI cohorts do not match")
    static, static_qc = engine.load_static_covariates(cohort_ids)
    event_covariates, tracking_qc = engine.load_event_covariates(cohort, static, cohort_ids)
    long_data, instrument_qc = engine.load_mental_health_long(event_covariates, cohort_ids)
    long_data = long_data[
        long_data["instrument"].eq("CBCL") & long_data["is_primary_event"]
    ].copy()
    long_data = long_data.merge(
        demographics,
        on="id",
        how="left",
        validate="many_to_one",
        indicator=True,
    )
    coverage = float(long_data["_merge"].eq("both").mean() * 100.0)
    if coverage != 100.0:
        raise ValueError(f"CBCL demographic join coverage is {coverage:.3f}%")
    long_data = long_data.drop(columns="_merge")
    site_by_id = (
        long_data.groupby("id", observed=True)["site"]
        .agg(mode_or_na)
        .rename("analysis_site")
        .reset_index()
    )
    long_data = long_data.merge(site_by_id, on="id", how="left", validate="many_to_one")
    qc_rows = [
        {
            "source": "cbcl",
            "metric": "hsdi_cohort_size",
            "value": len(cohort_ids),
        },
        {
            "source": "cbcl",
            "metric": "annual_primary_rows",
            "value": int(len(long_data)),
        },
        {
            "source": "cbcl",
            "metric": "annual_primary_unique_ids",
            "value": int(long_data["id"].nunique()),
        },
        {
            "source": "cbcl",
            "metric": "demographic_join_coverage_pct",
            "value": coverage,
        },
        {
            "source": "cbcl",
            "metric": "tracking_family_clusters",
            "value": int(long_data["family_cluster"].nunique()),
        },
        {
            "source": "cbcl",
            "metric": "input_sha256",
            "value": sha256_file(Path(engine.CBCL_INPUT)),
        },
        {
            "source": "cbcl",
            "metric": "source_hsdi_join_rows",
            "value": int(hsdi_qc.get("matched_rows_in_hsdi_cohort", len(cohort_ids))),
        },
        {
            "source": "cbcl",
            "metric": "static_covariate_ids",
            "value": int(static_qc.get("matched_unique_ids_in_hsdi_cohort", len(cohort_ids))),
        },
        {
            "source": "cbcl",
            "metric": "tracking_rows",
            "value": int(tracking_qc.get("matched_rows_in_hsdi_cohort", 0)),
        },
        {
            "source": "cbcl",
            "metric": "instrument_join_records",
            "value": int(len(instrument_qc)),
        },
    ]
    return long_data, cbcl_specs, pd.DataFrame(qc_rows)

def cognition_metadata(spec: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "source": "cognition",
        "instrument": "Cognition",
        "reporter": "Participant performance",
        "outcome": str(spec["outcome"]),
        "outcome_label": str(spec["display_label"]),
        "outcome_short_label": str(spec["short_label"]).replace("\n", " "),
        "event_key": str(spec["event_key"]),
        "event_label": str(spec["event_key"]),
        "covariate_set": "age, sex, site, scanner manufacturer, race/ethnicity",
        "cluster_unit": "participant",
    }

def cbcl_metadata(spec: Mapping[str, Any]) -> dict[str, Any]:
    short, label = CBCL_LABELS[str(spec["variable"])]
    return {
        "source": "cbcl",
        "instrument": "CBCL",
        "reporter": "Parent",
        "outcome": str(spec["variable"]),
        "outcome_label": label,
        "outcome_short_label": short,
        "event_key": "POOLED_ANNUAL",
        "event_label": "Pooled annual visits",
        "covariate_set": (
            "within-event age, event, sex, site, scanner manufacturer, "
            "race/ethnicity, grouped family income"
        ),
        "cluster_unit": "baseline family ID; participant fallback",
    }

def prepare_cbcl_outcome_frame(long_data: pd.DataFrame, outcome: str) -> pd.DataFrame:
    frame = long_data[long_data["outcome"].eq(outcome)].copy()
    frame["outcome_z"] = pd.to_numeric(frame["outcome_z_within_event"], errors="coerce")
    frame["age_cov"] = pd.to_numeric(frame["age_within_event_z"], errors="coerce")
    return frame

def _fit_ols_with_rank_guard(
    formula: str,
    frame: pd.DataFrame,
    cluster_column: str,
    scanner_term_present: bool,
) -> tuple[Any, str, tuple[str, ...]]:
    omitted: list[str] = []
    estimable_formula = formula
    preflight = smf.ols(formula=estimable_formula, data=frame)
    rank = int(np.linalg.matrix_rank(preflight.exog))
    columns = int(preflight.exog.shape[1])
    if rank < columns and scanner_term_present:
        estimable_formula = estimable_formula.replace(" + C(scanner_manufacturer)", "")
        omitted.append("scanner_manufacturer collinear with site")
        preflight = smf.ols(formula=estimable_formula, data=frame)
        rank = int(np.linalg.matrix_rank(preflight.exog))
        columns = int(preflight.exog.shape[1])
    if rank < columns:
        raise ValueError(f"Rank-deficient design: rank {rank} of {columns}")
    fit = preflight.fit(
        cov_type="cluster",
        cov_kwds={
            "groups": frame[cluster_column].astype(str).to_numpy(),
            "use_correction": True,
        },
    )
    return fit, estimable_formula, tuple(omitted)

def fit_model(
    frame: pd.DataFrame,
    metadata: Mapping[str, Any],
    hsdi_vars: Sequence[str],
    category_columns: Sequence[str],
    cluster_column: str,
    model_variant: str,
    include_ood: bool = False,
    trim_fraction: float | None = None,
) -> FitBundle:
    work = frame.copy()
    if len(hsdi_vars) == 1:
        predictor_terms = ("hsdi_z",)
        work["hsdi_z"] = pd.to_numeric(
            work[f"z_{hsdi_vars[0]}"], errors="coerce"
        )
    else:
        predictor_terms = tuple(f"z_{hsdi}" for hsdi in hsdi_vars)
        for hsdi, term in zip(hsdi_vars, predictor_terms):
            work[term] = pd.to_numeric(work[f"z_{hsdi}"], errors="coerce")
    work["outcome_z"] = pd.to_numeric(work["outcome_z"], errors="coerce")
    work["age_cov"] = pd.to_numeric(work["age_cov"], errors="coerce")
    for column in category_columns:
        work[column] = clean_category(work[column])
    required = [
        "id",
        "analysis_site",
        cluster_column,
        "outcome_z",
        "age_cov",
        *predictor_terms,
        *category_columns,
    ]
    if include_ood:
        required.extend(OOD_Z_COLUMNS)
        for column in OOD_Z_COLUMNS:
            work[column] = pd.to_numeric(work[column], errors="coerce")
    work = work.dropna(subset=list(dict.fromkeys(required))).copy()
    if work["id"].nunique() < MIN_SUBJECTS:
        raise ValueError(f"Fewer than {MIN_SUBJECTS} complete-case subjects")
    if work[cluster_column].nunique() < MIN_FAMILIES:
        raise ValueError(f"Fewer than {MIN_FAMILIES} complete-case clusters")
    n_removed_tail_subjects = 0
    trim_threshold: float | None = None
    if trim_fraction is not None:
        if not 0.0 < float(trim_fraction) < 0.5:
            raise ValueError("trim_fraction must be between 0 and 0.5")
        subject_values = work[["id", predictor_terms[0]]].drop_duplicates("id")
        trim_threshold = float(
            subject_values[predictor_terms[0]].abs().quantile(1.0 - trim_fraction)
        )
        removed_ids = set(
            subject_values.loc[
                subject_values[predictor_terms[0]].abs() > trim_threshold,
                "id",
            ]
        )
        n_removed_tail_subjects = len(removed_ids)
        work = work[~work["id"].isin(removed_ids)].copy()
    for column in [*category_columns, cluster_column]:
        work[column] = work[column].astype(str)

    included_categories: list[str] = []
    omitted: list[str] = []
    for column in category_columns:
        if column == "scanner_manufacturer":
            continue
        if work[column].nunique(dropna=True) >= 2:
            included_categories.append(column)
    if "scanner_manufacturer" in category_columns and work["scanner_manufacturer"].nunique() >= 2:
        scanner_nested = (
            "site" in included_categories
            and work.groupby("site", observed=True)["scanner_manufacturer"].nunique().max() <= 1
        )
        if scanner_nested:
            omitted.append("scanner_manufacturer nested within site")
        else:
            scanner_index = min(
                len(included_categories),
                list(category_columns).index("scanner_manufacturer"),
            )
            included_categories.insert(scanner_index, "scanner_manufacturer")
    covariate_terms = ["age_cov", *[f"C({column})" for column in included_categories]]
    if include_ood:
        covariate_terms.extend(OOD_Z_COLUMNS)
    formula = "outcome_z ~ " + " + ".join([*predictor_terms, *covariate_terms])
    fit, estimable_formula, rank_omissions = _fit_ols_with_rank_guard(
        formula,
        work,
        cluster_column,
        "C(scanner_manufacturer)" in formula,
    )
    omitted.extend(rank_omissions)
    return FitBundle(
        metadata=dict(metadata),
        frame=work,
        fit=fit,
        formula=estimable_formula,
        outcome_term="outcome_z",
        predictor_terms=predictor_terms,
        covariate_terms=tuple(
            term for term in covariate_terms if term != "C(scanner_manufacturer)" or term in estimable_formula
        ),
        cluster_column=cluster_column,
        omitted_redundant_covariates=tuple(dict.fromkeys(omitted)),
        n_removed_tail_subjects=n_removed_tail_subjects,
        trim_threshold=trim_threshold,
    )

def single_result_record(
    bundle: FitBundle,
    hsdi: str,
    model_variant: str,
    include_ood: bool,
    trim_fraction: float | None,
) -> dict[str, Any]:
    term = bundle.predictor_terms[0]
    confidence = bundle.fit.conf_int().loc[term]
    design = np.asarray(bundle.fit.model.exog, dtype=float)
    return {
        **bundle.metadata,
        "hsdi": hsdi,
        "hsdi_label": HSDI_LABELS[hsdi],
        "model_variant": model_variant,
        "beta_std": float(bundle.fit.params[term]),
        "se_cluster": float(bundle.fit.bse[term]),
        "ci95_low": float(confidence.iloc[0]),
        "ci95_high": float(confidence.iloc[1]),
        "t_value": float(bundle.fit.tvalues[term]),
        "p_value": float(bundle.fit.pvalues[term]),
        "n_observations": int(bundle.fit.nobs),
        "n_subjects": int(bundle.frame["id"].nunique()),
        "n_families": int(bundle.frame[bundle.cluster_column].nunique()),
        "n_events": int(bundle.frame["event_key"].nunique()),
        "n_sites": int(bundle.frame["analysis_site"].nunique()),
        "formula": bundle.formula,
        "covariates": ", ".join(bundle.covariate_terms),
        "omitted_redundant_covariates": "; ".join(bundle.omitted_redundant_covariates),
        "cluster_column": bundle.cluster_column,
        "predictor_standardization": "once across the 853-subject HSDI cohort",
        "outcome_standardization": (
            "within outcome-event" if bundle.metadata["source"] == "cognition" else "within annual visit"
        ),
        "includes_race_ethnicity": True,
        "includes_grouped_family_income": bundle.metadata["source"] == "cbcl",
        "includes_ood_covariates": bool(include_ood),
        "trim_fraction_each_tail_metric": trim_fraction,
        "trim_threshold_abs_hsdi_z": bundle.trim_threshold,
        "n_removed_tail_subjects": int(bundle.n_removed_tail_subjects),
        "condition_number": float(np.linalg.cond(design)),
        "design_rank": int(np.linalg.matrix_rank(design)),
        "n_design_columns": int(design.shape[1]),
    }

def apply_cognition_fdr(results: pd.DataFrame) -> pd.DataFrame:
    output = results.copy()
    if len(output) != len(COGNITION_SPECS) * len(HSDI_ORDER):
        raise AssertionError("Cognition primary family must contain 28 tests")
    output["q_fdr_family"] = multipletests(output["p_value"], method="fdr_bh")[1]
    output["fdr_family"] = "cognition primary: 7 outcomes x 4 HSDI (28 tests)"
    output["n_tests_fdr_family"] = len(output)
    output["passes_fdr"] = output["q_fdr_family"].lt(ALPHA)
    return output

def apply_cbcl_fdr(results: pd.DataFrame) -> pd.DataFrame:
    output = results.copy()
    expected_tests = len(CBCL_VARIABLES) * len(HSDI_ORDER)
    if len(output) != expected_tests:
        raise AssertionError(f"CBCL primary family must contain {expected_tests} tests")
    output["q_fdr_family"] = multipletests(output["p_value"], method="fdr_bh")[1]
    output["n_tests_fdr_family"] = expected_tests
    output["fdr_family"] = "CBCL primary: 4 outcomes x 4 HSDI (16 tests)"
    output["passes_fdr"] = output["q_fdr_family"].lt(ALPHA)
    return output

def build_cbcl_supplementary_table(results: pd.DataFrame) -> pd.DataFrame:
    table = results[
        [
            "outcome_label",
            "hsdi_label",
            "beta_std",
            "se_cluster",
            "ci95_low",
            "ci95_high",
            "q_fdr_family",
            "passes_fdr",
            "n_observations",
            "n_subjects",
            "n_families",
            "covariate_set",
            "cluster_unit",
        ]
    ].copy()
    table = table.rename(columns={"q_fdr_family": "q_fdr_bh_16"})
    table["fdr_family_size"] = len(results)
    return table.sort_values(["outcome_label", "hsdi_label"]).reset_index(drop=True)

def fit_primary_landscapes(
    cognition_frames: Mapping[tuple[str, str], pd.DataFrame],
    cbcl_long: pd.DataFrame,
    cbcl_specs: Sequence[Mapping[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cognition_rows: list[dict[str, Any]] = []
    for spec in COGNITION_SPECS:
        frame = cognition_frames[(str(spec["event_key"]), str(spec["outcome"]))]
        metadata = cognition_metadata(spec)
        for hsdi in HSDI_ORDER:
            bundle = fit_model(
                frame,
                metadata,
                (hsdi,),
                ("sex", "site", "scanner_manufacturer", "race_ethnicity"),
                "family_cluster",
                "COGNITION_RACE_PRIMARY",
            )
            cognition_rows.append(
                single_result_record(
                    bundle,
                    hsdi,
                    "COGNITION_RACE_PRIMARY",
                    False,
                    None,
                )
            )
    cbcl_rows: list[dict[str, Any]] = []
    for spec in cbcl_specs:
        frame = prepare_cbcl_outcome_frame(cbcl_long, str(spec["variable"]))
        metadata = cbcl_metadata(spec)
        for hsdi in HSDI_ORDER:
            bundle = fit_model(
                frame,
                metadata,
                (hsdi,),
                (
                    "event_key",
                    "sex",
                    "site",
                    "scanner_manufacturer",
                    "race_ethnicity",
                    "family_income_grouped",
                ),
                "family_cluster",
                "CBCL_RACE_INCOME_PRIMARY",
            )
            cbcl_rows.append(
                single_result_record(
                    bundle,
                    hsdi,
                    "CBCL_RACE_INCOME_PRIMARY",
                    False,
                    None,
                )
            )
    return apply_cognition_fdr(pd.DataFrame(cognition_rows)), apply_cbcl_fdr(pd.DataFrame(cbcl_rows))

def validate_cbcl_against_reference(
    results: pd.DataFrame,
    reference_path: Path,
) -> dict[str, Any]:
    reference = pd.read_csv(reference_path, low_memory=False)
    reference = reference[
        reference["instrument"].eq("CBCL")
        & reference["model_variant"].eq("OLS_RACE_ETHNICITY_INCOME")
    ].copy()
    if len(reference) != len(CBCL_VARIABLES) * len(HSDI_ORDER):
        raise AssertionError(
            "Validated CBCL reference must contain 16 outcome-HSDI rows"
        )
    reference["q_fdr_bh_16"] = multipletests(
        reference["p_value"], method="fdr_bh"
    )[1]
    merged = results.merge(
        reference[
            [
                "outcome",
                "hsdi",
                "beta_std",
                "se_cluster",
                "p_value",
                "n_observations",
                "n_subjects",
                "n_families",
                "q_fdr_bh_16",
            ]
        ],
        on=["outcome", "hsdi"],
        how="inner",
        suffixes=("_new", "_reference"),
        validate="one_to_one",
    )
    if len(merged) != 16:
        raise AssertionError(f"Expected 16 matched CBCL reference rows, found {len(merged)}")
    numeric_pairs = (
        ("beta_std_new", "beta_std_reference"),
        ("se_cluster_new", "se_cluster_reference"),
        ("p_value_new", "p_value_reference"),
        ("q_fdr_family", "q_fdr_bh_16"),
    )
    differences = {
        left: float(np.max(np.abs(merged[left] - merged[right])))
        for left, right in numeric_pairs
    }
    integer_matches = bool(
        merged["n_observations_new"].eq(merged["n_observations_reference"]).all()
        and merged["n_subjects_new"].eq(merged["n_subjects_reference"]).all()
        and merged["n_families_new"].eq(merged["n_families_reference"]).all()
    )
    tolerance = 1e-10
    passed = all(value <= tolerance for value in differences.values()) and integer_matches
    if not passed:
        raise AssertionError(
            "CBCL primary refit did not reproduce the validated income-only reference: "
            f"differences={differences}, integer_matches={integer_matches}"
        )
    return {
        "status": "passed",
        "matched_rows": int(len(merged)),
        "tolerance": tolerance,
        "max_absolute_differences": differences,
        "sample_counts_match": integer_matches,
        "reference_path": str(reference_path.resolve()),
        "reference_sha256": sha256_file(reference_path),
    }

def _representative_frame_and_metadata(
    spec: Mapping[str, str],
    cognition_frames: Mapping[tuple[str, str], pd.DataFrame],
    cbcl_long: pd.DataFrame,
    cbcl_specs: Sequence[Mapping[str, Any]],
) -> tuple[pd.DataFrame, dict[str, Any], tuple[str, ...], str]:
    if spec["source"] == "cognition":
        cognitive_spec = next(
            item
            for item in COGNITION_SPECS
            if item["outcome"] == spec["outcome"] and item["event_key"] == spec["event_key"]
        )
        frame = cognition_frames[(spec["event_key"], spec["outcome"])]
        metadata = cognition_metadata(cognitive_spec)
        metadata["relationship_short_label"] = spec["short_label"]
        categories = ("sex", "site", "scanner_manufacturer", "race_ethnicity")
        return frame, metadata, categories, "family_cluster"
    cbcl_spec = next(item for item in cbcl_specs if item["variable"] == spec["outcome"])
    frame = prepare_cbcl_outcome_frame(cbcl_long, spec["outcome"])
    metadata = cbcl_metadata(cbcl_spec)
    metadata["relationship_short_label"] = spec["short_label"]
    categories = (
        "event_key",
        "sex",
        "site",
        "scanner_manufacturer",
        "race_ethnicity",
        "family_income_grouped",
    )
    return frame, metadata, categories, "family_cluster"

def summarize_quantile_bins(
    x: np.ndarray,
    y: np.ndarray,
    ids: np.ndarray,
    clusters: np.ndarray,
    n_bins: int = 16,
    min_count: int = 8,
) -> pd.DataFrame:
    edges = np.unique(np.quantile(x, np.linspace(0.0, 1.0, n_bins + 1)))
    if len(edges) < 5:
        raise ValueError("Too few unique quantile-bin edges")
    bin_index = np.searchsorted(edges[1:-1], x, side="right")
    records: list[dict[str, Any]] = []
    for index in range(len(edges) - 1):
        mask = bin_index == index
        count = int(mask.sum())
        if count < min_count:
            continue
        y_values = y[mask]
        y_sd = float(np.std(y_values, ddof=1)) if count > 1 else 0.0
        records.append(
            {
                "bin": index,
                "n_observations": count,
                "n_subjects": int(pd.Series(ids[mask]).nunique()),
                "n_clusters": int(pd.Series(clusters[mask]).nunique()),
                "x_mean": float(np.mean(x[mask])),
                "y_mean": float(np.mean(y_values)),
                "y_ci95_descriptive": float(1.96 * y_sd / np.sqrt(max(count, 1))),
            }
        )
    output = pd.DataFrame(records)
    if output.empty:
        raise ValueError("No valid quantile bins were produced")
    return output

def build_partial_display_data(
    cognition_frames: Mapping[tuple[str, str], pd.DataFrame],
    cbcl_long: pd.DataFrame,
    cbcl_specs: Sequence[Mapping[str, Any]],
    primary_results: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    bin_parts: list[pd.DataFrame] = []
    line_parts: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []
    for order, spec in enumerate(REPRESENTATIVE_SPECS):
        frame, metadata, categories, cluster_column = _representative_frame_and_metadata(
            spec,
            cognition_frames,
            cbcl_long,
            cbcl_specs,
        )
        bundle = fit_model(
            frame,
            metadata,
            (HSDI_ORDER[0],),
            categories,
            cluster_column,
            "PRIMARY",
        )
        result_row = primary_results[
            primary_results["source"].eq(spec["source"])
            & primary_results["outcome"].eq(spec["outcome"])
            & primary_results["event_key"].eq(spec["event_key"])
            & primary_results["hsdi"].eq(HSDI_ORDER[0])
        ]
        if len(result_row) != 1:
            raise AssertionError(f"Expected one representative primary result for {spec}")
        result_row = result_row.iloc[0]
        if not np.isclose(
            float(bundle.fit.params[bundle.predictor_terms[0]]),
            float(result_row["beta_std"]),
            atol=1e-10,
            rtol=0.0,
        ):
            raise AssertionError(f"Representative refit mismatch for {spec}")
        nuisance_rhs = " + ".join(bundle.covariate_terms)
        predictor = bundle.predictor_terms[0]
        x_residual = smf.ols(
            f"{predictor} ~ {nuisance_rhs}",
            data=bundle.frame,
        ).fit().resid.to_numpy(dtype=float)
        y_residual = smf.ols(
            f"{bundle.outcome_term} ~ {nuisance_rhs}",
            data=bundle.frame,
        ).fit().resid.to_numpy(dtype=float)
        partial = pd.DataFrame(
            {
                "x_partial": x_residual,
                "y_partial": y_residual,
                "id": bundle.frame["id"].astype(str).to_numpy(),
                "cluster": bundle.frame[cluster_column].astype(str).to_numpy(),
            }
        )
        partial_fit = smf.ols("y_partial ~ x_partial", data=partial).fit(
            cov_type="cluster",
            cov_kwds={
                "groups": partial["cluster"].to_numpy(),
                "use_correction": True,
            },
        )
        partial_slope = float(partial_fit.params["x_partial"])
        if not np.isclose(partial_slope, float(result_row["beta_std"]), atol=1e-10, rtol=0.0):
            raise AssertionError(f"Partial-regression slope mismatch for {spec}")
        bins = summarize_quantile_bins(
            partial["x_partial"].to_numpy(dtype=float),
            partial["y_partial"].to_numpy(dtype=float),
            partial["id"].to_numpy(),
            partial["cluster"].to_numpy(),
        )
        x_grid = np.linspace(
            float(np.quantile(partial["x_partial"], 0.005)),
            float(np.quantile(partial["x_partial"], 0.995)),
            200,
        )
        prediction = partial_fit.get_prediction(
            pd.DataFrame({"x_partial": x_grid})
        ).summary_frame(alpha=0.05)
        line = pd.DataFrame(
            {
                "x_partial": x_grid,
                "fitted_y": prediction["mean"].to_numpy(dtype=float),
                "ci95_low": prediction["mean_ci_lower"].to_numpy(dtype=float),
                "ci95_high": prediction["mean_ci_upper"].to_numpy(dtype=float),
            }
        )
        for output in (bins, line):
            output.insert(0, "relationship_order", order)
            output.insert(1, "source", spec["source"])
            output.insert(2, "outcome", spec["outcome"])
            output.insert(3, "event_key", spec["event_key"])
            output.insert(4, "relationship_short_label", spec["short_label"])
        bin_parts.append(bins)
        line_parts.append(line)
        summaries.append(
            {
                "relationship_order": order,
                "source": spec["source"],
                "outcome": spec["outcome"],
                "event_key": spec["event_key"],
                "relationship_short_label": spec["short_label"],
                "display_label": spec["display_label"],
                "beta_std": float(result_row["beta_std"]),
                "se_cluster": float(result_row["se_cluster"]),
                "ci95_low": float(result_row["ci95_low"]),
                "ci95_high": float(result_row["ci95_high"]),
                "p_value": float(result_row["p_value"]),
                "q_fdr_family": float(result_row["q_fdr_family"]),
                "fdr_family": str(result_row["fdr_family"]),
                "n_observations": int(result_row["n_observations"]),
                "n_subjects": int(result_row["n_subjects"]),
                "n_families": int(result_row["n_families"]),
                "partial_slope": partial_slope,
                "partial_slope_difference": partial_slope - float(result_row["beta_std"]),
                "n_bins": int(len(bins)),
            }
        )
    return (
        pd.concat(bin_parts, ignore_index=True),
        pd.concat(line_parts, ignore_index=True),
        pd.DataFrame(summaries),
    )

def build_joint_models(
    cognition_frames: Mapping[tuple[str, str], pd.DataFrame],
    cbcl_long: pd.DataFrame,
    cbcl_specs: Sequence[Mapping[str, Any]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for relationship_order, spec in enumerate(REPRESENTATIVE_SPECS):
        frame, metadata, categories, cluster_column = _representative_frame_and_metadata(
            spec,
            cognition_frames,
            cbcl_long,
            cbcl_specs,
        )
        bundle = fit_model(
            frame,
            metadata,
            HSDI_ORDER,
            categories,
            cluster_column,
            "JOINT_HSDI_PRIMARY",
        )
        hsdi_matrix = bundle.frame[list(bundle.predictor_terms)].to_numpy(dtype=float)
        corr_condition = float(np.linalg.cond(np.corrcoef(hsdi_matrix, rowvar=False)))
        vif_values = {
            hsdi: float(variance_inflation_factor(hsdi_matrix, index))
            for index, hsdi in enumerate(HSDI_ORDER)
        }
        for hsdi_order, (hsdi, term) in enumerate(zip(HSDI_ORDER, bundle.predictor_terms)):
            confidence = bundle.fit.conf_int().loc[term]
            rows.append(
                {
                    **bundle.metadata,
                    "relationship_order": relationship_order,
                    "relationship_short_label": spec["short_label"],
                    "hsdi_order": hsdi_order,
                    "hsdi": hsdi,
                    "hsdi_label": HSDI_LABELS[hsdi],
                    "model_variant": "JOINT_HSDI_PRIMARY",
                    "beta_std": float(bundle.fit.params[term]),
                    "se_cluster": float(bundle.fit.bse[term]),
                    "ci95_low": float(confidence.iloc[0]),
                    "ci95_high": float(confidence.iloc[1]),
                    "t_value": float(bundle.fit.tvalues[term]),
                    "p_value": float(bundle.fit.pvalues[term]),
                    "n_observations": int(bundle.fit.nobs),
                    "n_subjects": int(bundle.frame["id"].nunique()),
                    "n_families": int(bundle.frame[cluster_column].nunique()),
                    "formula": bundle.formula,
                    "hsdi_corr_condition_number": corr_condition,
                    "hsdi_vif": vif_values[hsdi],
                    "includes_race_ethnicity": True,
                    "includes_grouped_family_income": spec["source"] == "cbcl",
                }
            )
    output = pd.DataFrame(rows)
    output["q_fdr_joint16"] = multipletests(output["p_value"], method="fdr_bh")[1]
    output["holm_p_sahigh4"] = np.nan
    sahigh = output["hsdi"].eq(HSDI_ORDER[0])
    output.loc[sahigh, "holm_p_sahigh4"] = multipletests(
        output.loc[sahigh, "p_value"],
        method="holm",
    )[1]
    return output

def build_sensitivity_models(
    cognition_frames: Mapping[tuple[str, str], pd.DataFrame],
    cbcl_long: pd.DataFrame,
    cbcl_specs: Sequence[Mapping[str, Any]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for relationship_order, spec in enumerate(REPRESENTATIVE_SPECS):
        frame, metadata, categories, cluster_column = _representative_frame_and_metadata(
            spec,
            cognition_frames,
            cbcl_long,
            cbcl_specs,
        )
        for variant_order, (variant, include_ood, trim_fraction) in enumerate(SENSITIVITY_VARIANTS):
            bundle = fit_model(
                frame,
                metadata,
                (HSDI_ORDER[0],),
                categories,
                cluster_column,
                variant,
                include_ood=include_ood,
                trim_fraction=trim_fraction,
            )
            record = single_result_record(
                bundle,
                HSDI_ORDER[0],
                variant,
                include_ood,
                trim_fraction,
            )
            record["relationship_order"] = relationship_order
            record["relationship_short_label"] = spec["short_label"]
            record["variant_order"] = variant_order
            rows.append(record)
    return pd.DataFrame(rows)

def build_leave_one_site_out(
    cognition_frames: Mapping[tuple[str, str], pd.DataFrame],
    cbcl_long: pd.DataFrame,
    cbcl_specs: Sequence[Mapping[str, Any]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for relationship_order, spec in enumerate(REPRESENTATIVE_SPECS):
        frame, metadata, categories, cluster_column = _representative_frame_and_metadata(
            spec,
            cognition_frames,
            cbcl_long,
            cbcl_specs,
        )
        full = fit_model(
            frame,
            metadata,
            (HSDI_ORDER[0],),
            categories,
            cluster_column,
            "PRIMARY",
        )
        term = full.predictor_terms[0]
        full_beta = float(full.fit.params[term])
        full_se = float(full.fit.bse[term])
        sites = sorted(full.frame["analysis_site"].astype(str).unique())
        if len(sites) < 10:
            raise AssertionError(f"Unexpectedly few sites for {spec}: {len(sites)}")
        for site_order, omitted_site in enumerate(sites):
            omitted_ids = set(
                full.frame.loc[
                    full.frame["analysis_site"].astype(str).eq(omitted_site),
                    "id",
                ]
            )
            retained = full.frame[~full.frame["id"].isin(omitted_ids)].copy()
            deletion = fit_model(
                retained,
                metadata,
                (HSDI_ORDER[0],),
                categories,
                cluster_column,
                "LEAVE_ONE_SITE_OUT",
            )
            deletion_term = deletion.predictor_terms[0]
            confidence = deletion.fit.conf_int().loc[deletion_term]
            beta = float(deletion.fit.params[deletion_term])
            rows.append(
                {
                    **metadata,
                    "relationship_order": relationship_order,
                    "relationship_short_label": spec["short_label"],
                    "site_order": site_order,
                    "omitted_site": omitted_site,
                    "omitted_subjects": int(len(omitted_ids)),
                    "retained_observations": int(deletion.fit.nobs),
                    "retained_subjects": int(deletion.frame["id"].nunique()),
                    "retained_families": int(deletion.frame[cluster_column].nunique()),
                    "beta_std": beta,
                    "se_cluster": float(deletion.fit.bse[deletion_term]),
                    "ci95_low": float(confidence.iloc[0]),
                    "ci95_high": float(confidence.iloc[1]),
                    "p_value": float(deletion.fit.pvalues[deletion_term]),
                    "full_beta_std": full_beta,
                    "full_se_cluster": full_se,
                    "full_ci95_low": full_beta - 1.959963984540054 * full_se,
                    "full_ci95_high": full_beta + 1.959963984540054 * full_se,
                    "delta_from_full": beta - full_beta,
                    "same_direction_as_full": bool(beta * full_beta > 0.0),
                    "sign_reversal": bool(beta * full_beta < 0.0),
                    "includes_race_ethnicity": True,
                    "includes_grouped_family_income": spec["source"] == "cbcl",
                }
            )
    return pd.DataFrame(rows)

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

BETA_CMAP = LinearSegmentedColormap.from_list(
    "fig6_beta",
    ["#4C78A8", "#F7F7F7", "#B2795C"],
)

COGNITION_COLOR = "#58799C"

CBCL_COLOR = "#8B5E78"

SAHIGH_COLOR = "#3B8D6D"

NEUTRAL_COLOR = "#777777"

def marker_size_from_q(q_value: float) -> float:
    q_value = float(q_value) if np.isfinite(q_value) else 1.0
    magnitude = min(3.5, max(0.0, -np.log10(q_value + EPS)))
    return float(np.clip(55.0 + 55.0 * magnitude, 55.0, 250.0))

def _style_axis(axis: plt.Axes) -> None:
    axis.spines["left"].set_linewidth(0.6)
    axis.spines["bottom"].set_linewidth(0.6)
    axis.tick_params(axis="both", width=0.55, length=2.7, pad=2)

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

def format_probability(value: float, symbol: str) -> str:
    if value < 0.001:
        return f"{symbol}<0.001"
    return f"{symbol}={value:.3f}"

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

def create_figure(
    cognition: pd.DataFrame,
    cbcl: pd.DataFrame,
    partial_bins: pd.DataFrame,
    partial_lines: pd.DataFrame,
    partial_summary: pd.DataFrame,
    joint: pd.DataFrame,
    sensitivity: pd.DataFrame,
    loso: pd.DataFrame,
    output_dir: Path,
    figure_main_dir: Path,
) -> dict[str, Path]:
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

    output_dir.mkdir(parents=True, exist_ok=True)
    figure_main_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "png": output_dir / f"{FIGURE_STEM}.png",
        "pdf": output_dir / f"{FIGURE_STEM}.pdf",
        "svg": output_dir / f"{FIGURE_STEM}.svg",
    }
    figure.savefig(outputs["png"], dpi=600, facecolor="white", bbox_inches=None)
    figure.savefig(outputs["pdf"], facecolor="white", bbox_inches=None)
    figure.savefig(outputs["svg"], facecolor="white", bbox_inches=None)
    plt.close(figure)
    for source in outputs.values():
        shutil.copy2(source, figure_main_dir / source.name)
    return outputs

def build_run_summary(
    cognition: pd.DataFrame,
    cbcl: pd.DataFrame,
    joint: pd.DataFrame,
    sensitivity: pd.DataFrame,
    loso: pd.DataFrame,
) -> str:
    primary = pd.concat([cognition, cbcl], ignore_index=True, sort=False)
    representative_keys = {
        ("cognition", "pea_wiscv_tss", "baselineYear1Arm1"),
        ("cognition", "nihtbx_cryst_fc", "2YearFollowUpYArm1"),
        ("cbcl", "cbcl_scr_syn_totprob_r", "POOLED_ANNUAL"),
        ("cbcl", "cbcl_scr_syn_attention_r", "POOLED_ANNUAL"),
    }
    representative = primary[
        primary["hsdi"].eq(HSDI_ORDER[0])
        & primary.apply(
            lambda row: (row["source"], row["outcome"], row["event_key"])
            in representative_keys,
            axis=1,
        )
    ].copy()
    lines = [
        "# Integrated Figure 6 run summary",
        "",
        "- Main models all include race/ethnicity.",
        "- CBCL main models additionally include grouped family income.",
        "- No WLS, PQ-BC, trajectory association, or no-race main model was run.",
        f"- Cognition primary tests: {len(cognition)} (one 28-test BH family).",
        f"- CBCL primary tests: {len(cbcl)} (one 16-test BH family).",
        f"- Joint conditional coefficients: {len(joint)}.",
        f"- OOD/tail sensitivity rows: {len(sensitivity)}.",
        f"- Leave-one-site-out rows: {len(loso)}.",
        "",
        "## SA-high representative results",
        "",
    ]
    for _, row in representative.sort_values(["source", "outcome", "event_key"]).iterrows():
        lines.append(
            f"- {row['outcome_label']}: beta={row['beta_std']:.6f}, "
            f"95% CI [{row['ci95_low']:.6f}, {row['ci95_high']:.6f}], "
            f"p={row['p_value']:.6g}, q={row['q_fdr_family']:.6g}, "
            f"N={int(row['n_subjects'])} subjects."
        )
    lines.extend(
        [
            "",
            "The results are associational. A stable signed coefficient does not establish a causal or protective mechanism.",
            "",
        ]
    )
    return "\n".join(lines)

def run_analysis(
    repo_root: Path | None = None,
    run_name: str = RUN_NAME,
) -> dict[str, Any]:
    root = locate_repo_root(repo_root)
    paths = AnalysisPaths.from_root(root, run_name=run_name)
    validate_required_paths(paths)
    paths.output_dir.mkdir(parents=True, exist_ok=True)

    hsdi, hsdi_qc = load_hsdi(paths)
    hsdi_ids = set(hsdi["id"].astype(str))
    demographics, demographic_levels, demographic_qc = load_demographics(paths, hsdi_ids)
    cognition_frames, cognition_qc = load_cognition_frames(paths, hsdi, demographics)
    cbcl_long, cbcl_specs, cbcl_qc = load_cbcl_long(paths, demographics, hsdi_ids)

    cognition_results, cbcl_results = fit_primary_landscapes(
        cognition_frames,
        cbcl_long,
        cbcl_specs,
    )
    cbcl_supplementary_table = build_cbcl_supplementary_table(cbcl_results)
    reference_validation = validate_cbcl_against_reference(
        cbcl_results,
        paths.cbcl_reference_results,
    )
    primary_results = pd.concat(
        [cognition_results, cbcl_results],
        ignore_index=True,
        sort=False,
    )
    partial_bins, partial_lines, partial_summary = build_partial_display_data(
        cognition_frames,
        cbcl_long,
        cbcl_specs,
        primary_results,
    )
    joint_results = build_joint_models(cognition_frames, cbcl_long, cbcl_specs)
    sensitivity_results = build_sensitivity_models(cognition_frames, cbcl_long, cbcl_specs)
    loso_results = build_leave_one_site_out(cognition_frames, cbcl_long, cbcl_specs)

    output_tables = {
        "cognition_primary": paths.output_dir / "fig6_cognition_primary_associations.csv",
        "cbcl_primary": paths.output_dir / "fig6_cbcl_primary_associations.csv",
        "cbcl_supplementary": paths.output_dir / "fig6_cbcl_supplementary_table_fdr16.csv",
        "partial_bins": paths.output_dir / "fig6_primary_partial_bins.csv",
        "partial_lines": paths.output_dir / "fig6_primary_partial_lines.csv",
        "partial_summary": paths.output_dir / "fig6_primary_partial_summary.csv",
        "joint": paths.output_dir / "fig6_joint_hsdi_conditional.csv",
        "sensitivity": paths.output_dir / "fig6_ood_tail_sensitivity.csv",
        "loso": paths.output_dir / "fig6_leave_one_site_out.csv",
        "cognition_qc": paths.output_dir / "qc_cognition_join_summary.csv",
        "cbcl_qc": paths.output_dir / "qc_cbcl_join_summary.csv",
        "demographic_levels": paths.output_dir / "qc_demographic_level_counts.csv",
    }
    table_frames = {
        "cognition_primary": cognition_results,
        "cbcl_primary": cbcl_results,
        "cbcl_supplementary": cbcl_supplementary_table,
        "partial_bins": partial_bins,
        "partial_lines": partial_lines,
        "partial_summary": partial_summary,
        "joint": joint_results,
        "sensitivity": sensitivity_results,
        "loso": loso_results,
        "cognition_qc": cognition_qc,
        "cbcl_qc": cbcl_qc,
        "demographic_levels": demographic_levels,
    }
    for name, frame in table_frames.items():
        frame.to_csv(output_tables[name], index=False)

    figure_outputs = create_figure(
        cognition_results,
        cbcl_results,
        partial_bins,
        partial_lines,
        partial_summary,
        joint_results,
        sensitivity_results,
        loso_results,
        paths.output_dir,
        paths.figure_main_dir,
    )
    run_summary = build_run_summary(
        cognition_results,
        cbcl_results,
        joint_results,
        sensitivity_results,
        loso_results,
    )
    summary_path = paths.output_dir / "RUN_SUMMARY.md"
    summary_path.write_text(run_summary, encoding="utf-8")
    environment_path = paths.output_dir / "environment_versions.txt"
    environment_path.write_text(
        "\n".join(
            [
                f"python={platform.python_version()}",
                f"numpy={np.__version__}",
                f"pandas={pd.__version__}",
                f"scipy={scipy.__version__}",
                f"statsmodels={statsmodels.__version__}",
                f"matplotlib={mpl.__version__}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    representative_cbcl_outcomes = {
        str(spec["outcome"])
        for spec in REPRESENTATIVE_SPECS
        if str(spec["source"]) == "cbcl"
    }
    representative_primary = primary_results[
        primary_results["hsdi"].eq(HSDI_ORDER[0])
        & (
            (primary_results["source"].eq("cognition") & primary_results.apply(
                lambda row: (row["outcome"], row["event_key"])
                in {("pea_wiscv_tss", "baselineYear1Arm1"), ("nihtbx_cryst_fc", "2YearFollowUpYArm1")},
                axis=1,
            ))
            | (
                primary_results["source"].eq("cbcl")
                & primary_results["outcome"].isin(representative_cbcl_outcomes)
            )
        )
    ].copy()
    loso_summary = (
        loso_results.groupby(
            ["relationship_order", "relationship_short_label"],
            observed=True,
        )
        .agg(
            n_site_deletions=("omitted_site", "size"),
            same_direction_fraction=("same_direction_as_full", "mean"),
            minimum_beta=("beta_std", "min"),
            maximum_beta=("beta_std", "max"),
            maximum_absolute_delta=("delta_from_full", lambda values: float(np.max(np.abs(values)))),
        )
        .reset_index()
    )
    audit = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "script_version": SCRIPT_VERSION,
        "run_name": run_name,
        "analysis_objective": "Integrated cognition and CBCL associations with human-specific deviation",
        "primary_covariates": {
            "cognition": "age, sex, site, scanner manufacturer when estimable, race/ethnicity",
            "cbcl": (
                "within-event age, event, sex, site, scanner manufacturer when estimable, "
                "race/ethnicity, grouped family income"
            ),
        },
        "dependence": {
            "cognition": "participant-clustered standard errors",
            "cbcl": "baseline-family-clustered standard errors with participant fallback",
        },
        "multiple_testing": {
            "cognition": "BH across 28 primary outcome-HSDI tests",
            "cbcl": "BH across 16 primary outcome-HSDI tests (4 outcomes x 4 HSDI)",
            "joint_sahigh": "Holm across four representative SA-high conditional coefficients",
        },
        "excluded_analyses": [
            "WLS",
            "PQ-BC",
            "trajectory association",
            "no-race main model",
        ],
        "input_qc": {
            "hsdi": hsdi_qc,
            "demographics": demographic_qc,
        },
        "cbcl_reference_validation": reference_validation,
        "output_validation": {
            "cognition_primary_rows": int(len(cognition_results)),
            "cbcl_primary_rows": int(len(cbcl_results)),
            "partial_relationships": int(partial_summary["relationship_order"].nunique()),
            "partial_slopes_match_primary": bool(
                partial_summary["partial_slope_difference"].abs().max() <= 1e-10
            ),
            "joint_rows": int(len(joint_results)),
            "sensitivity_rows": int(len(sensitivity_results)),
            "loso_rows": int(len(loso_results)),
            "all_main_models_include_race": bool(primary_results["includes_race_ethnicity"].all()),
            "all_cbcl_main_models_include_income": bool(
                cbcl_results["includes_grouped_family_income"].all()
            ),
            "cbcl_fdr_family_sizes": sorted(
                cbcl_results["n_tests_fdr_family"].astype(int).unique().tolist()
            ),
            "cbcl_fdr_significant_rows": int(cbcl_results["passes_fdr"].sum()),
            "cbcl_supplementary_table_omits_raw_p": bool(
                "p_value" not in cbcl_supplementary_table.columns
            ),
            "no_wls_rows": bool(
                ~primary_results["model_variant"].astype(str).str.contains("WLS").any()
                and ~sensitivity_results["model_variant"].astype(str).str.contains("WLS").any()
            ),
            "participant_level_merged_output_saved": False,
        },
        "representative_primary_results": representative_primary[
            [
                "source",
                "outcome",
                "event_key",
                "beta_std",
                "se_cluster",
                "ci95_low",
                "ci95_high",
                "p_value",
                "q_fdr_family",
                "n_observations",
                "n_subjects",
                "n_families",
            ]
        ].to_dict(orient="records"),
        "site_stability_summary": loso_summary.to_dict(orient="records"),
        "outputs": {
            **{
                name: {
                    "path": str(path.resolve()),
                    "bytes": int(path.stat().st_size),
                    "sha256": sha256_file(path),
                }
                for name, path in output_tables.items()
            },
            **{
                f"figure_{extension}": {
                    "path": str(path.resolve()),
                    "bytes": int(path.stat().st_size),
                    "sha256": sha256_file(path),
                }
                for extension, path in figure_outputs.items()
            },
        },
    }
    audit_path = paths.output_dir / "analysis_audit.json"
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    del cognition_frames, cbcl_long, hsdi, demographics
    gc.collect()
    return {
        "paths": paths,
        "cognition_results": cognition_results,
        "cbcl_results": cbcl_results,
        "cbcl_supplementary_table": cbcl_supplementary_table,
        "partial_summary": partial_summary,
        "joint_results": joint_results,
        "sensitivity_results": sensitivity_results,
        "loso_results": loso_results,
        "reference_validation": reference_validation,
        "audit": audit,
        "audit_path": audit_path,
        "summary_path": summary_path,
        "figure_outputs": figure_outputs,
    }

if __name__ == "__main__":
    run_analysis()
