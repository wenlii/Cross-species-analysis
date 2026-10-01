"""Inputs: FIG56_BEHAVIOR_DIR/ and FIG56_HSDI_DIR/. Outputs: FIG56_OUTPUT_DIR/cognition_inputs/."""

from __future__ import annotations

import os
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = Path(os.environ.get("FIG56_OUTPUT_DIR", str(PACKAGE_ROOT / "outputs")))
BEHAVIOR_DIR = Path(os.environ.get("FIG56_BEHAVIOR_DIR", str(PACKAGE_ROOT / "data" / "private" / "behavior")))
HSDI_DIR = Path(os.environ.get("FIG56_HSDI_DIR", str(OUTPUT_ROOT / "hsdi")))
COGNITION_DIR = OUTPUT_ROOT / "cognition_inputs"
COGNITION_DIR.mkdir(parents=True, exist_ok=True)

import numpy as np

import pandas as pd


from typing import Dict, List, Optional, Tuple


FINALFIG_DIR = HSDI_DIR

HSDI_VARIANT = "noClip"

HSDI_BYTRIPLE = FINALFIG_DIR / f"HSDI_wide_byTriple_{HSDI_VARIANT}.csv"

SUBINFO_CSV = BEHAVIOR_DIR / 'subinfo_new_with_3year_complete_covariates.csv'

NEUROCOG_DIR = BEHAVIOR_DIR

OUT_BEH_DIR = COGNITION_DIR

BEH_SPECS: Dict[str, Dict[str, List[str]]] = {
    "nc_y_nihtb.csv": {
        "events": ["baselineYear1Arm1", "2YearFollowUpYArm1", "4YearFollowUpYArm1"],
        "vars": ["nihtbx_cryst_fc", "nihtbx_fluidcomp_fc"],
    },
    "nc_y_ravlt.csv": {
        "events": ["baselineYear1Arm1", "2YearFollowUpYArm1"],
        "vars": ["pea_ravlt_ld_trial_vii_tc"],
    },
    "nc_y_wisc.csv": {
        "events": ["baselineYear1Arm1"],
        "vars": ["pea_wiscv_tss"],
    },
    "nc_y_smarte.csv": {
        "events": ["3YearFollowUpYArm1"],
        "vars": ["smarte_ss_all_total_corr"],
    },
}

CELL_COLS = [
    "HSdevZ_CT_high_expansion",
    "HSdevZ_CT_low_expansion",
    "HSdevZ_SA_high_expansion",
    "HSdevZ_SA_low_expansion",
]

BASE_COVARS = ["sex", "site", "scanner_manufacturer"]

OOD_CANDIDATES = [
    "OOD_frac_q_mean",
    "OOD_frac_q_max",
    "OOD_max_abs_z_mean",
    "OOD_max_abs_z_max",
    "OOD_n_feat_absz_ge_thresh_mean",
    "OOD_n_feat_absz_ge_thresh_max",
]

def _normalize_id(x: pd.Series) -> pd.Series:
    s = x.astype("string").str.strip()
    s = s.str.replace(r"^sub-", "", regex=True, case=False)
    return s.str.upper()

def _clean_required_category(x: pd.Series) -> pd.Series:
    s = x.astype("string").str.strip()
    invalid = s.isna() | s.eq("") | s.str.lower().isin(["nan", "none", "<na>"])
    return s.mask(invalid).astype(object)

def _find_col_case_insensitive(df: pd.DataFrame, name: str) -> Optional[str]:
    mp = {str(c).lower(): c for c in df.columns}
    return mp.get(str(name).lower(), None)

def _find_first_existing(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        hit = _find_col_case_insensitive(df, c)
        if hit is not None:
            return hit
    return None

def _infer_id_event_cols(df: pd.DataFrame) -> Tuple[str, str]:
    id_col = _find_first_existing(df, ["subjectkey", "src_subject_id", "subject_id", "id"])
    ev_col = _find_first_existing(df, ["eventname", "event_name", "event", "visit"])
    if id_col is None or ev_col is None:
        raise ValueError("Cannot infer ID/event columns.")
    return id_col, ev_col

def load_subinfo_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}")

    df = pd.read_csv(path, low_memory=False)
    source_columns = {
        "id": _find_first_existing(
            df, ["src_subject_id", "subjectkey", "subject_id", "id"]
        ),
        "eventname": _find_first_existing(
            df, ["eventname", "event_name", "event", "visit"]
        ),
        "age_years": _find_first_existing(
            df, ["age_years", "interview_age_yrs", "age", "age_years_bl"]
        ),
        "site": _find_first_existing(df, ["site_id_l", "site_id", "site"]),
        "sex": _find_first_existing(df, ["demo_sex_v2", "sex", "sex_at_birth"]),
        "scanner_manufacturer": _find_first_existing(
            df, ["mri_info_manufacturer", "scanner_manufacturer"]
        ),
    }
    missing_roles = [
        role for role, column in source_columns.items() if column is None
    ]
    if missing_roles:
        raise ValueError(
            f"Covariate table is missing required roles: {missing_roles}"
        )

    age_years = pd.to_numeric(
        df[source_columns["age_years"]], errors="coerce"
    ).astype(float)
    age_months_column = _find_first_existing(df, ["interview_age"])
    if age_months_column is not None:
        age_months = pd.to_numeric(
            df[age_months_column], errors="coerce"
        ).astype(float)
        paired = age_years.notna() & age_months.notna()
        if not np.allclose(
            age_years.loc[paired].to_numpy(),
            (age_months.loc[paired] / 12.0).to_numpy(),
            rtol=0.0,
            atol=1e-12,
        ):
            raise AssertionError(
                "age_years is not exactly interview_age / 12"
            )

    sex = _clean_required_category(df[source_columns["sex"]])
    sex = sex.replace({"1": "M", "2": "F", "M": "M", "F": "F"})
    out = pd.DataFrame(
        {
            "id": _normalize_id(df[source_columns["id"]]),
            "eventname": _clean_required_category(
                df[source_columns["eventname"]]
            ),
            "age_subinfo_years": age_years,
            "site": _clean_required_category(df[source_columns["site"]]),
            "sex": sex,
            "scanner_manufacturer": _clean_required_category(
                df[source_columns["scanner_manufacturer"]]
            ),
        }
    )
    if out[["id", "eventname"]].isna().any().any():
        raise AssertionError("Covariate keys contain missing values")
    duplicate_keys = out.duplicated(["id", "eventname"], keep=False)
    if duplicate_keys.any():
        raise AssertionError(
            "Covariate table contains duplicate participant-event keys"
        )
    return out.sort_values(["id", "eventname"]).reset_index(drop=True)

def build_hsdi_subject_table(hsdi_bytriple: pd.DataFrame) -> pd.DataFrame:
    df = hsdi_bytriple.copy()

    if "id" not in df.columns:
        raise ValueError("HSDI_wide_byTriple file must contain 'id'.")

    df["id"] = _normalize_id(df["id"])

    for c in CELL_COLS + ["age_T2"] + OOD_CANDIDATES:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)

    mean_cols = CELL_COLS + ["age_T2"] + OOD_CANDIDATES
    out = df.groupby("id", as_index=False)[mean_cols].mean()
    return out

def main() -> None:
    required_behavior_inputs = [
        HSDI_BYTRIPLE,
        SUBINFO_CSV,
        *[NEUROCOG_DIR / filename for filename in BEH_SPECS],
    ]
    missing_behavior_inputs = [p for p in required_behavior_inputs if not p.exists()]
    if missing_behavior_inputs:
        missing_text = "\n".join(f"  - {p}" for p in missing_behavior_inputs)
        raise FileNotFoundError(
            "Missing required Figure 6 behavior inputs:\n" + missing_text
        )


    hsdi_bytriple = pd.read_csv(HSDI_BYTRIPLE)
    hsdi_bytriple["id"] = _normalize_id(hsdi_bytriple["id"])

    hsdi_subj = build_hsdi_subject_table(hsdi_bytriple)
    hsdi_subj.to_csv(OUT_BEH_DIR / "HSDI_subjectLevel_fromByTriple_core4.csv", index=False)

    subinfo = load_subinfo_table(SUBINFO_CSV)
    subinfo.to_csv(OUT_BEH_DIR / "subinfo_event_covariates_used.csv", index=False)


    for beh_file, spec in BEH_SPECS.items():
        fp = NEUROCOG_DIR / beh_file

        df = pd.read_csv(fp, low_memory=False)
        id_col, ev_col = _infer_id_event_cols(df)

        df = df.rename(columns={id_col: "id", ev_col: "eventname"}).copy()
        df["id"] = _normalize_id(df["id"])
        df["eventname"] = df["eventname"].astype(str)

        want_vars = spec["vars"]
        keep_vars = [v for v in want_vars if v in df.columns]
        if len(keep_vars) == 0:
            continue

        keep_cols = ["id", "eventname"] + keep_vars
        df_small = df[keep_cols].copy()

        merged = (
            df_small
            .merge(
                subinfo,
                on=["id", "eventname"],
                how="left",
                validate="many_to_one",
            )
            .merge(
                hsdi_subj,
                on="id",
                how="inner",
                validate="many_to_one",
            )
        )

        if beh_file == "nc_y_smarte.csv":
            smarte_rows = merged.loc[
                merged["eventname"].astype(str).eq(
                    "3YearFollowUpYArm1"
                )
            ].copy()
            if len(smarte_rows) != 527:
                raise AssertionError(
                    f"Expected 527 SMARTE rows, found {len(smarte_rows)}"
                )
            if smarte_rows["id"].nunique() != 527:
                raise AssertionError(
                    "SMARTE participant-event merge is not one row per ID"
                )
            if smarte_rows[
                ["age_subinfo_years"] + BASE_COVARS
            ].isna().any().any():
                raise AssertionError(
                    "SMARTE contains missing required covariates after join"
                )

        merged.to_csv(OUT_BEH_DIR / f"MERGED_{Path(beh_file).stem}_withHSDI_subinfo_tail.csv", index=False)

if __name__ == "__main__":
    main()
