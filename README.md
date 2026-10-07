# Developmental Change Operator and Human-Specific Deviation Modeling

This repository documents the modeling workflow used in *Evolutionary expansion organizes adolescent cortical remodeling rates and behavior-associated human-specific developmental deviation*. It learns a developmental change operator from human longitudinal structural MRI, evaluates cross-species predictions in macaques, estimates human-specific deviation in a separate macaque-referenced stage, and tests behavior associations.

## Modeling workflow

| Stage | Implementation | Main output |
| --- | --- | --- |
| Human longitudinal reference and cross-species prediction | `src/developmental_change_operator.py` | `prediction_outputs/` |
| Macaque-referenced human-specific deviation | `src/human_specific_deviation.py` | `hsdi/` |
| Cognition and covariate preparation | `src/prepare_cognition_inputs.py` | `cognition_inputs/` |
| Cognition and parent-reported CBCL behavior association | `src/behavior_association.py`, using `src/cbcl_inputs.py` | `fig6_cognition_cbcl_race_income_v1/` |

The output directories above are created under `CROSS_SPECIES_OUTPUT_DIR` (default: `outputs/`). The release begins with prepared expansion-group feature tables and cohort metadata.

## Model specification

- The human developmental change operator uses the `baselineT2_noCratio` variant, subject-grouped five-fold validation, fold-wise confound residualization, feature screening, a spline basis and the preserved parameter grid. The human-to-macaque age conversion factor is 2.83. Prediction-space choices and optional calibration calculations remain as implemented.
- The separate macaque-referenced stage retains the original age-matching rule, complete scan-triple keys, out-of-distribution diagnostics and unclipped human-specific deviation definitions. It does not refit the human operator as a macaque model.
- Cognition associations adjust for age, sex, site, estimable scanner manufacturer and race/ethnicity, with participant-clustered standard errors. Pooled annual CBCL associations adjust for within-event age, event, sex, site, estimable scanner manufacturer, race/ethnicity and grouped family income, with the original family-clustering rules.
- The primary Benjamini–Hochberg families contain 28 cognition tests and 16 CBCL tests. Joint models, out-of-distribution adjustment, 1% and 5% tail exclusions, and leave-one-site-out analyses retain their original specifications. These are associations, not causal estimates.

## Inputs and execution

Use Python 3.11 and install the pinned packages:

```sh
python -m pip install -r requirements.txt
```

Set the following environment variables to authorized inputs outside this repository before recomputing participant-level results:

| Variable | Required contents | Stage |
| --- | --- | --- |
| `CROSS_SPECIES_FEATURE_DIR` | `allSubjects_human_features_expansion.csv`, `allSubjects_macaque_features_expansion_completed.csv` | Developmental change operator |
| `CROSS_SPECIES_BEHAVIOR_DIR` | `subinfo_new_with_3year_complete_covariates.csv`, `nc_y_nihtb.csv`, `nc_y_ravlt.csv`, `nc_y_wisc.csv`, `nc_y_smarte.csv` | Cognition preparation |
| `CROSS_SPECIES_DEMOGRAPHICS_FILE` | `subinfo_event_covariates_with_demographics_grouped.xlsx`, with `Covariates` and `Codebook` sheets | Behavior association |
| `ABCD_CORE_DIR` | ABCD release 5.1 `core`, including `abcd-general/abcd_y_lt.csv` and `mental-health/mh_p_cbcl.csv` | Behavior association |
| `CROSS_SPECIES_OUTPUT_DIR` | Writable output directory; defaults to `outputs/` | All stages |

Run the modeling stages in order:

```sh
python src/developmental_change_operator.py
python src/human_specific_deviation.py
python src/prepare_cognition_inputs.py
python src/behavior_association.py
```

`CROSS_SPECIES_PREDICTION_DIR`, `CROSS_SPECIES_HSDI_DIR` and `CROSS_SPECIES_COGNITION_DIR` can point to existing authorized intermediate results. `inputs.json` records required filenames, schemas and checksums; it contains no participant records.

## Public results and figure regeneration

The public release includes two aggregate developmental-operator performance tables under `modeling_outputs/developmental_change_operator/`. Its eight aggregate behavior-association plotting tables and one CBCL reference table are under `modeling_outputs/behavior_association/`. Participant-level feature, prediction and behavioral inputs are not included. In particular, the two public performance tables cannot regenerate the participant-level plots on their own.

From this directory, the supplied behavior-association tables can be checked and used to regenerate the reported Figure 6 without controlled inputs:

```sh
python verify.py
python src/plot_behavior_associations.py
```
