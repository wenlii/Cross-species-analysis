"""Inputs: ABCD_CORE_DIR/ and FIG56_COGNITION_DIR/. No participant-level outputs."""

from __future__ import annotations

import os
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = Path(os.environ.get("FIG56_OUTPUT_DIR", str(PACKAGE_ROOT / "outputs")))
COGNITION_DIR = Path(os.environ.get("FIG56_COGNITION_DIR", str(OUTPUT_ROOT / "cognition_inputs")))


from typing import Any, Dict, List, Mapping, Sequence, Tuple


import numpy as np

import pandas as pd


REPO_ROOT = PACKAGE_ROOT

HSDI_INPUT = COGNITION_DIR / 'HSDI_subjectLevel_fromByTriple_core4.csv'

FIG6_COVARIATE_INPUT = COGNITION_DIR / 'subinfo_event_covariates_used.csv'

ABCD_RELEASE_ROOT = Path(os.environ['ABCD_CORE_DIR'])

ABCD_LONGITUDINAL_INPUT = ABCD_RELEASE_ROOT / "abcd-general" / "abcd_y_lt.csv"

MENTAL_HEALTH_DIR = ABCD_RELEASE_ROOT / "mental-health"

CBCL_INPUT = MENTAL_HEALTH_DIR / "mh_p_cbcl.csv"

BPM_INPUT = MENTAL_HEALTH_DIR / "mh_y_bpm.csv"

PPS_INPUT = MENTAL_HEALTH_DIR / "mh_y_pps.csv"

HSDI_ORDER: Sequence[str] = (
    "HSdevZ_SA_high_expansion",
    "HSdevZ_SA_low_expansion",
    "HSdevZ_CT_high_expansion",
    "HSdevZ_CT_low_expansion",
)

OOD_RAW_COLUMNS: Sequence[str] = (
    "OOD_max_abs_z_max",
    "OOD_frac_q_max",
)

EPS = 1e-12

EVENT_DEFINITIONS: Mapping[str, Mapping[str, Any]] = {
    "baselineyear1arm1": {"label": "Baseline", "scheduled_year": 0.0, "order": 0},
    "6monthfollowuparm1": {"label": "6 months", "scheduled_year": 0.5, "order": 1},
    "1yearfollowupyarm1": {"label": "1 year", "scheduled_year": 1.0, "order": 2},
    "18monthfollowuparm1": {"label": "18 months", "scheduled_year": 1.5, "order": 3},
    "2yearfollowupyarm1": {"label": "2 years", "scheduled_year": 2.0, "order": 4},
    "30monthfollowuparm1": {"label": "30 months", "scheduled_year": 2.5, "order": 5},
    "3yearfollowupyarm1": {"label": "3 years", "scheduled_year": 3.0, "order": 6},
    "42monthfollowuparm1": {"label": "42 months", "scheduled_year": 3.5, "order": 7},
    "4yearfollowupyarm1": {"label": "4 years", "scheduled_year": 4.0, "order": 8},
}

ANNUAL_EVENTS: Sequence[str] = (
    "baselineyear1arm1",
    "1yearfollowupyarm1",
    "2yearfollowupyarm1",
    "3yearfollowupyarm1",
    "4yearfollowupyarm1",
)

BPM_PRIMARY_EVENTS: Sequence[str] = (
    "1yearfollowupyarm1",
    "2yearfollowupyarm1",
    "3yearfollowupyarm1",
    "4yearfollowupyarm1",
)

BPM_INTERIM_EVENTS: Sequence[str] = (
    "6monthfollowuparm1",
    "18monthfollowuparm1",
    "30monthfollowuparm1",
    "42monthfollowuparm1",
)

OUTCOME_SPECS: Sequence[Mapping[str, Any]] = (
    {
        "instrument": "CBCL",
        "reporter": "Parent",
        "path": CBCL_INPUT,
        "variable": "cbcl_scr_syn_totprob_r",
        "label": "CBCL total problems",
        "short_label": "CBCL: Total",
        "primary_events": ANNUAL_EVENTS,
        "sensitivity_events": (),
    },
    {
        "instrument": "CBCL",
        "reporter": "Parent",
        "path": CBCL_INPUT,
        "variable": "cbcl_scr_syn_internal_r",
        "label": "CBCL internalizing problems",
        "short_label": "CBCL: Internalizing",
        "primary_events": ANNUAL_EVENTS,
        "sensitivity_events": (),
    },
    {
        "instrument": "CBCL",
        "reporter": "Parent",
        "path": CBCL_INPUT,
        "variable": "cbcl_scr_syn_external_r",
        "label": "CBCL externalizing problems",
        "short_label": "CBCL: Externalizing",
        "primary_events": ANNUAL_EVENTS,
        "sensitivity_events": (),
    },
    {
        "instrument": "BPM",
        "reporter": "Youth",
        "path": BPM_INPUT,
        "variable": "bpm_y_scr_totalprob_r",
        "label": "BPM total problems",
        "short_label": "BPM: Total",
        "primary_events": BPM_PRIMARY_EVENTS,
        "sensitivity_events": BPM_INTERIM_EVENTS,
    },
    {
        "instrument": "BPM",
        "reporter": "Youth",
        "path": BPM_INPUT,
        "variable": "bpm_y_scr_internal_r",
        "label": "BPM internalizing problems",
        "short_label": "BPM: Internalizing",
        "primary_events": BPM_PRIMARY_EVENTS,
        "sensitivity_events": BPM_INTERIM_EVENTS,
    },
    {
        "instrument": "BPM",
        "reporter": "Youth",
        "path": BPM_INPUT,
        "variable": "bpm_y_scr_external_r",
        "label": "BPM externalizing problems",
        "short_label": "BPM: Externalizing",
        "primary_events": BPM_PRIMARY_EVENTS,
        "sensitivity_events": BPM_INTERIM_EVENTS,
    },
    {
        "instrument": "BPM",
        "reporter": "Youth",
        "path": BPM_INPUT,
        "variable": "bpm_y_scr_attention_r",
        "label": "BPM attention problems",
        "short_label": "BPM: Attention",
        "primary_events": BPM_PRIMARY_EVENTS,
        "sensitivity_events": BPM_INTERIM_EVENTS,
    },
    {
        "instrument": "PQ-BC",
        "reporter": "Youth",
        "path": PPS_INPUT,
        "variable": "pps_y_ss_number",
        "label": "PQ-BC endorsed experiences",
        "short_label": "PQ-BC: Count",
        "primary_events": ANNUAL_EVENTS,
        "sensitivity_events": (),
    },
    {
        "instrument": "PQ-BC",
        "reporter": "Youth",
        "path": PPS_INPUT,
        "variable": "pps_y_ss_severity_score",
        "label": "PQ-BC distress/severity",
        "short_label": "PQ-BC: Severity",
        "primary_events": ANNUAL_EVENTS,
        "sensitivity_events": (),
    },
)

OUTCOME_LOOKUP: Mapping[str, Mapping[str, Any]] = {
    str(spec["variable"]): spec for spec in OUTCOME_SPECS
}

def normalize_id(series: pd.Series) -> pd.Series:
    values = series.astype("string").str.strip().str.upper()
    values = values.str.replace(r"^SUB-", "", regex=True)
    values = values.str.replace("_", "", regex=False)
    return values.mask(values.isin(["", "<NA>"]))

def normalize_event(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .str.strip()
        .str.lower()
        .str.replace(r"[^a-z0-9]", "", regex=True)
    )

def clean_category(series: pd.Series) -> pd.Series:
    values = series.astype("string").str.strip()
    return values.mask(values.str.lower().isin(["", "nan", "none", "<na>"]))

def mode_or_na(series: pd.Series) -> Any:
    values = clean_category(series).dropna()
    if values.empty:
        return pd.NA
    return values.value_counts().index[0]

def zscore(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").astype(float)
    mean = float(values.mean(skipna=True))
    sd = float(values.std(skipna=True, ddof=0))
    if not np.isfinite(sd) or sd <= EPS:
        return pd.Series(np.nan, index=series.index, dtype=float)
    return (values - mean) / sd

def normalization_collision_count(raw_ids: pd.Series, normalized_ids: pd.Series) -> int:
    table = pd.DataFrame({"raw": raw_ids.astype("string"), "normalized": normalized_ids}).dropna()
    if table.empty:
        return 0
    counts = table.groupby("normalized", observed=True)["raw"].nunique()
    return int((counts > 1).sum())

def source_qc_record(
    source: str,
    path: Path,
    frame: pd.DataFrame,
    raw_id_col: str,
    cohort_ids: set,
    raw_event_col: str | None = None,
) -> Dict[str, Any]:
    normalized_ids = normalize_id(frame[raw_id_col])
    record: Dict[str, Any] = {
        "source": source,
        "source_path": str(path),
        "join_keys": "normalized subject ID" + (" + normalized event" if raw_event_col else ""),
        "raw_rows": int(len(frame)),
        "raw_unique_ids": int(frame[raw_id_col].astype("string").nunique(dropna=True)),
        "normalized_unique_ids": int(normalized_ids.nunique(dropna=True)),
        "normalization_collision_count": normalization_collision_count(frame[raw_id_col], normalized_ids),
        "normalized_ids_overlapping_hsdi": int(len(set(normalized_ids.dropna()) & cohort_ids)),
        "hsdi_cohort_size": int(len(cohort_ids)),
    }
    record["hsdi_subject_join_coverage_pct"] = (
        100.0 * record["normalized_ids_overlapping_hsdi"] / max(len(cohort_ids), 1)
    )
    if raw_event_col is not None:
        keys = pd.DataFrame(
            {"id": normalized_ids, "event": normalize_event(frame[raw_event_col])}
        ).dropna()
        record["normalized_unique_subject_event_keys"] = int(keys.drop_duplicates().shape[0])
        record["duplicate_subject_event_rows"] = int(keys.duplicated(["id", "event"], keep=False).sum())
    else:
        record["normalized_unique_subject_event_keys"] = np.nan
        record["duplicate_subject_event_rows"] = np.nan
    return record

def load_hsdi() -> Tuple[pd.DataFrame, Dict[str, Any]]:
    raw = pd.read_csv(HSDI_INPUT, low_memory=False)
    required = ["id", *HSDI_ORDER, *OOD_RAW_COLUMNS]
    missing = [column for column in required if column not in raw.columns]
    if missing:
        raise ValueError(f"HSDI input is missing columns: {missing}")

    cohort = raw[required].copy()
    cohort["id"] = normalize_id(cohort["id"])
    if cohort["id"].isna().any():
        raise ValueError("HSDI input contains missing normalized subject IDs")
    if cohort["id"].duplicated().any():
        raise ValueError("HSDI input contains duplicate normalized subject IDs")
    for column in [*HSDI_ORDER, *OOD_RAW_COLUMNS]:
        cohort[column] = pd.to_numeric(cohort[column], errors="coerce")
        if cohort[column].isna().any():
            raise ValueError(f"HSDI input contains missing values in {column}")
    for column in HSDI_ORDER:
        cohort[f"z_{column}"] = zscore(cohort[column])
    cohort["z_OOD_max_abs_z_max"] = zscore(cohort["OOD_max_abs_z_max"])
    cohort["z_OOD_frac_q_max"] = zscore(cohort["OOD_frac_q_max"])

    qc = {
        "source": "human_specific_deviation",
        "source_path": str(HSDI_INPUT),
        "join_keys": "normalized subject ID",
        "raw_rows": int(len(raw)),
        "raw_unique_ids": int(raw["id"].nunique(dropna=True)),
        "normalized_unique_ids": int(cohort["id"].nunique()),
        "normalization_collision_count": normalization_collision_count(raw["id"], normalize_id(raw["id"])),
        "normalized_ids_overlapping_hsdi": int(len(cohort)),
        "hsdi_cohort_size": int(len(cohort)),
        "hsdi_subject_join_coverage_pct": 100.0,
        "normalized_unique_subject_event_keys": np.nan,
        "duplicate_subject_event_rows": np.nan,
    }
    return cohort, qc

def load_static_covariates(cohort_ids: set) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    raw = pd.read_csv(FIG6_COVARIATE_INPUT, low_memory=False)
    required = ["id", "eventname", "site", "sex", "scanner_manufacturer"]
    missing = [column for column in required if column not in raw.columns]
    if missing:
        raise ValueError(f"Figure 6 covariate input is missing columns: {missing}")
    qc = source_qc_record(
        "fig6_covariates",
        FIG6_COVARIATE_INPUT,
        raw,
        "id",
        cohort_ids,
        "eventname",
    )
    frame = raw[required].copy()
    frame["id"] = normalize_id(frame["id"])
    frame = frame[frame["id"].isin(cohort_ids)].copy()
    for column in ["site", "sex", "scanner_manufacturer"]:
        frame[column] = clean_category(frame[column])
    static = (
        frame.groupby("id", as_index=False, observed=True)
        .agg(
            site_static=("site", mode_or_na),
            sex=("sex", mode_or_na),
            scanner_manufacturer=("scanner_manufacturer", mode_or_na),
        )
    )
    qc["matched_rows_in_hsdi_cohort"] = int(len(frame))
    qc["matched_unique_ids_in_hsdi_cohort"] = int(static["id"].nunique())
    return static, qc

def load_event_covariates(
    cohort: pd.DataFrame,
    static: pd.DataFrame,
    cohort_ids: set,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    usecols = [
        "src_subject_id",
        "eventname",
        "site_id_l",
        "rel_family_id",
        "interview_age",
    ]
    raw = pd.read_csv(
        ABCD_LONGITUDINAL_INPUT,
        usecols=usecols,
        dtype={"src_subject_id": "string", "eventname": "string", "rel_family_id": "string"},
        low_memory=False,
    )
    qc = source_qc_record(
        "abcd_longitudinal_tracking",
        ABCD_LONGITUDINAL_INPUT,
        raw,
        "src_subject_id",
        cohort_ids,
        "eventname",
    )
    frame = raw.copy()
    frame["id"] = normalize_id(frame["src_subject_id"])
    frame["event_key"] = normalize_event(frame["eventname"])
    frame = frame[
        frame["id"].isin(cohort_ids) & frame["event_key"].isin(EVENT_DEFINITIONS)
    ].copy()
    duplicate_count = int(frame.duplicated(["id", "event_key"], keep=False).sum())
    if duplicate_count:
        raise ValueError(f"ABCD tracking table has {duplicate_count} duplicate normalized subject-event rows")

    frame["age_event_years"] = pd.to_numeric(frame["interview_age"], errors="coerce") / 12.0
    frame["site_event"] = clean_category(frame["site_id_l"])
    frame["rel_family_id"] = clean_category(frame["rel_family_id"])
    frame["event_label"] = frame["event_key"].map(
        lambda value: EVENT_DEFINITIONS[str(value)]["label"]
    )
    frame["scheduled_year"] = frame["event_key"].map(
        lambda value: EVENT_DEFINITIONS[str(value)]["scheduled_year"]
    ).astype(float)
    frame["event_order"] = frame["event_key"].map(
        lambda value: EVENT_DEFINITIONS[str(value)]["order"]
    ).astype(int)

    baseline = frame[frame["event_key"] == "baselineyear1arm1"][
        ["id", "rel_family_id", "age_event_years"]
    ].rename(
        columns={
            "rel_family_id": "baseline_family_id",
            "age_event_years": "baseline_age_years",
        }
    )
    if baseline["id"].duplicated().any():
        raise ValueError("Baseline tracking rows are not unique by subject")

    event_covariates = frame[
        [
            "id",
            "event_key",
            "event_label",
            "scheduled_year",
            "event_order",
            "age_event_years",
            "site_event",
        ]
    ].merge(baseline, on="id", how="left", validate="many_to_one")
    event_covariates = event_covariates.merge(static, on="id", how="left", validate="many_to_one")
    event_covariates["site"] = event_covariates["site_event"].combine_first(
        event_covariates["site_static"]
    )
    event_covariates["family_cluster"] = event_covariates["baseline_family_id"].astype("string")
    missing_family = event_covariates["family_cluster"].isna()
    event_covariates.loc[missing_family, "family_cluster"] = (
        "SUBJECT_" + event_covariates.loc[missing_family, "id"].astype(str)
    )
    event_covariates["time_from_baseline_years"] = (
        event_covariates["age_event_years"] - event_covariates["baseline_age_years"]
    )
    event_covariates["age_within_event_z"] = event_covariates.groupby(
        "event_key", observed=True
    )["age_event_years"].transform(zscore)

    subject_baseline_age = (
        event_covariates[["id", "baseline_age_years"]]
        .drop_duplicates("id")
        .set_index("id")["baseline_age_years"]
    )
    baseline_age_z = zscore(subject_baseline_age).rename("baseline_age_z")
    event_covariates = event_covariates.merge(
        baseline_age_z.reset_index(), on="id", how="left", validate="many_to_one"
    )
    event_covariates = event_covariates.merge(cohort, on="id", how="left", validate="many_to_one")

    qc["matched_rows_in_hsdi_cohort"] = int(len(event_covariates))
    qc["matched_unique_ids_in_hsdi_cohort"] = int(event_covariates["id"].nunique())
    qc["baseline_family_id_nonmissing"] = int(baseline["baseline_family_id"].notna().sum())
    qc["unique_family_clusters"] = int(event_covariates["family_cluster"].nunique())
    qc["missing_baseline_age_rows"] = int(event_covariates["baseline_age_years"].isna().sum())
    return event_covariates, qc

def instrument_source_specs() -> Sequence[Tuple[str, str, Path, Sequence[str]]]:
    return (
        (
            "CBCL",
            "Parent",
            CBCL_INPUT,
            tuple(spec["variable"] for spec in OUTCOME_SPECS if spec["instrument"] == "CBCL"),
        ),
        (
            "BPM",
            "Youth",
            BPM_INPUT,
            tuple(spec["variable"] for spec in OUTCOME_SPECS if spec["instrument"] == "BPM"),
        ),
        (
            "PQ-BC",
            "Youth",
            PPS_INPUT,
            tuple(spec["variable"] for spec in OUTCOME_SPECS if spec["instrument"] == "PQ-BC"),
        ),
    )

def load_mental_health_long(
    event_covariates: pd.DataFrame,
    cohort_ids: set,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    long_frames: List[pd.DataFrame] = []
    qc_records: List[Dict[str, Any]] = []
    event_merge_columns = [
        "id",
        "event_key",
        "event_label",
        "scheduled_year",
        "event_order",
        "age_event_years",
        "baseline_age_years",
        "time_from_baseline_years",
        "age_within_event_z",
        "baseline_age_z",
        "site",
        "sex",
        "scanner_manufacturer",
        "family_cluster",
        *HSDI_ORDER,
        *(f"z_{column}" for column in HSDI_ORDER),
        *OOD_RAW_COLUMNS,
        "z_OOD_max_abs_z_max",
        "z_OOD_frac_q_max",
    ]
    event_table = event_covariates[event_merge_columns].copy()

    for instrument, reporter, path, variables in instrument_source_specs():
        usecols = ["src_subject_id", "eventname", *variables]
        raw = pd.read_csv(path, usecols=usecols, low_memory=False)
        qc = source_qc_record(
            f"{instrument.lower()}_{reporter.lower()}",
            path,
            raw,
            "src_subject_id",
            cohort_ids,
            "eventname",
        )
        frame = raw.copy()
        frame["id"] = normalize_id(frame["src_subject_id"])
        frame["event_key"] = normalize_event(frame["eventname"])
        frame = frame[
            frame["id"].isin(cohort_ids) & frame["event_key"].isin(EVENT_DEFINITIONS)
        ].copy()
        duplicate_count = int(frame.duplicated(["id", "event_key"], keep=False).sum())
        if duplicate_count:
            raise ValueError(
                f"{instrument} has {duplicate_count} duplicate normalized subject-event rows"
            )
        for variable in variables:
            frame[variable] = pd.to_numeric(frame[variable], errors="coerce")

        joined = frame[["id", "event_key", *variables]].merge(
            event_table,
            on=["id", "event_key"],
            how="left",
            validate="one_to_one",
            indicator=True,
        )
        unmatched = int((joined["_merge"] != "both").sum())
        joined = joined.drop(columns="_merge")
        joined["event_label"] = joined["event_label"].fillna(
            joined["event_key"].map(
                lambda value: EVENT_DEFINITIONS[str(value)]["label"]
            )
        )
        joined["scheduled_year"] = joined["scheduled_year"].fillna(
            joined["event_key"].map(
                lambda value: EVENT_DEFINITIONS[str(value)]["scheduled_year"]
            )
        )
        joined["event_order"] = joined["event_order"].fillna(
            joined["event_key"].map(
                lambda value: EVENT_DEFINITIONS[str(value)]["order"]
            )
        )
        qc["matched_rows_in_hsdi_cohort"] = int(len(frame))
        qc["matched_unique_ids_in_hsdi_cohort"] = int(frame["id"].nunique())
        qc["unmatched_subject_event_covariate_rows"] = unmatched
        qc_records.append(qc)

        id_columns = [column for column in joined.columns if column not in variables]
        melted = joined.melt(
            id_vars=id_columns,
            value_vars=list(variables),
            var_name="outcome",
            value_name="outcome_raw",
        )
        melted["instrument"] = instrument
        melted["reporter"] = reporter
        melted["outcome_label"] = melted["outcome"].map(
            lambda value: OUTCOME_LOOKUP[str(value)]["label"]
        )
        melted["outcome_short_label"] = melted["outcome"].map(
            lambda value: OUTCOME_LOOKUP[str(value)]["short_label"]
        )
        melted["is_primary_event"] = melted.apply(
            lambda row: row["event_key"]
            in OUTCOME_LOOKUP[str(row["outcome"])]["primary_events"],
            axis=1,
        )
        melted["is_sensitivity_event"] = melted.apply(
            lambda row: row["event_key"]
            in OUTCOME_LOOKUP[str(row["outcome"])]["sensitivity_events"],
            axis=1,
        )
        melted = melted[
            melted["is_primary_event"] | melted["is_sensitivity_event"]
        ].copy()
        melted["event_set"] = np.where(
            melted["is_primary_event"],
            "annual_primary",
            "bpm_interim_sensitivity",
        )
        long_frames.append(melted)

    long_data = pd.concat(long_frames, ignore_index=True)
    long_data["outcome_z_within_event"] = long_data.groupby(
        ["instrument", "outcome", "event_key"], observed=True
    )["outcome_raw"].transform(zscore)
    long_data["outcome_z_global_primary"] = np.nan
    primary_mask = long_data["is_primary_event"]
    long_data.loc[primary_mask, "outcome_z_global_primary"] = long_data.loc[
        primary_mask
    ].groupby(["instrument", "outcome"], observed=True)["outcome_raw"].transform(zscore)
    return long_data, qc_records
