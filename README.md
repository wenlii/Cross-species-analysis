# Figures 5 and 6

Code and aggregate source tables for *Evolutionary expansion organizes adolescent cortical remodeling rates and behavior-associated human-specific developmental deviation*.

This release starts from the study's expansion-group feature tables and cohort metadata. Image preprocessing and atlas construction are outside its scope. Participant-level inputs are excluded; their schemas and checksums are listed in `inputs.json`.

## Install and reproduce Figure 6

Use Python 3.11. From this directory:

```sh
python -m pip install -r requirements.txt
python verify.py
python src/plot_figure6.py
```

The eight included Figure 6 plotting tables reproduce `outputs/figure6/Fig6_colorbar_v2.png` without participant data. The ninth Figure 6 table is the reference used by the analysis validation. The two Figure 5 tables contain reference model-performance results.

## Recompute with authorized inputs

Set environment variables to directories outside this public repository. The same variables work on Windows, Linux and macOS.

| Variable | Contents | Required for |
| --- | --- | --- |
| `FIG56_FEATURE_DIR` | `allSubjects_human_features_expansion.csv`, `allSubjects_macaque_features_expansion_completed.csv` | Figure 5 model fitting |
| `FIG56_BEHAVIOR_DIR` | `subinfo_new_with_3year_complete_covariates.csv`, `nc_y_nihtb.csv`, `nc_y_ravlt.csv`, `nc_y_wisc.csv`, `nc_y_smarte.csv` | Cognition input preparation |
| `FIG56_DEMOGRAPHICS_FILE` | `subinfo_event_covariates_with_demographics_grouped.xlsx`, with `Covariates` and `Codebook` sheets | Figure 6 analysis |
| `ABCD_CORE_DIR` | ABCD release 5.1 `core` directory, containing `abcd-general/abcd_y_lt.csv` and `mental-health/mh_p_cbcl.csv` | Figure 6 analysis |
| `FIG56_OUTPUT_DIR` | Writable output directory; defaults to `outputs/` | All scripts |

Run in this order after configuring the inputs:

```sh
python src/figure5_analysis.py
python src/plot_figure5.py
python src/derive_hsdi.py
python src/prepare_cognition.py
python src/figure6_analysis.py
```

The Figure 5 model writes `prediction_outputs/`; the HSDI step writes `hsdi/`; cognition preparation writes `cognition_inputs/`. Figure 6 analysis writes `fig6_cognition_cbcl_race_income_v1/`. All are under the output directory. To draw the final Figure 6 layout from recomputed tables, set `FIG56_FIG6_TABLE_DIR` to that last directory and run `python src/plot_figure6.py`. Checksums are enforced for the supplied frozen tables; explicitly configured external tables retain the row-count checks. Use the supplied original tables for exact submitted-figure reproduction.

For existing authorized intermediate data, `FIG56_PREDICTION_DIR`, `FIG56_HSDI_DIR` and `FIG56_COGNITION_DIR` override the corresponding input directories. `inputs.json` lists the exact files needed for these entry points. Figure 5 requires participant-level prediction tables even when model fitting is skipped; it cannot be reconstructed from the two public performance tables alone.

## Analysis specification

- Figure 5 retains the `baselineT2_noCratio` developmental change operator, human subject-grouped five-fold validation, original confound residualization, feature screening, spline specification, model grids, and human-to-macaque transfer. The age conversion factor remains 2.83. Prediction-space choices and optional calibration calculations are preserved.
- Figure 5 plotting retains the four expansion-by-metric groups, Fisher exact tests, Spearman correlations, 24-month bins, 100 bootstrap draws per bin and random seed 0.
- HSDI generation retains the macaque reference fitting, age-matching rules, complete triple keys, out-of-distribution diagnostics and unclipped deviation definitions.
- Figure 6 cognition models retain age, sex, site, estimable scanner manufacturer and race/ethnicity adjustment with participant-clustered standard errors. Pooled annual CBCL models retain within-event age, event, sex, site, estimable scanner manufacturer, race/ethnicity and grouped family income, with the original family-clustering rules.
- Primary BH families remain 28 cognition tests and 16 CBCL tests. The joint, OOD-adjusted, 1%/5% trimming and leave-one-site-out analyses retain their original specifications. Associations do not establish causality.

## Source correspondence

| Release script | Original source |
| --- | --- |
| `src/figure5_analysis.py` | `analysis/notebooks/fig5/fig5_new_v2.ipynb`, code cell 2 |
| `src/plot_figure5.py` | `analysis/scripts/fig5_finalfig_sp_0921.ipynb`, final figure and required helpers |
| `src/derive_hsdi.py` | `analysis/scripts/fig6_finalfig_sp.ipynb`, code cell 2 |
| `src/prepare_cognition.py` | Same notebook, code cell 4, preparation through merged-table export |
| `src/cbcl_inputs.py` | Required loaders from `analysis/notebooks/fig11_disease/fig11_disease_association.py` |
| `src/figure6_analysis.py` | `src/behavior_association/fig6_cognition_cbcl_race_income.py` |
| `src/plot_figure6.py` | `src/plotting/fig6_colorbar_v2.py` |

Cell indices are zero-based. `validation.json` records the release checks. `checksums.sha256` covers every released file except itself. Original source files, manuscript documents, figures and data were preserved.

Validation reproduced both submitted figures pixel for pixel. The Figure 5 model's 57 CSV tables matched the saved numerical values; the Figure 6 analysis from frozen prepared inputs reproduced 12 CSV files byte for byte. Original and cleaned HSDI/cognition preparation produced identical tables with identical inputs. A complete refit showed maximum differences of 2.10e-9 in historical HSDI intermediates and 1.09e-12 in Figure 6 outputs. Four archived per-phenotype metadata files predate the current source's four added diagnostic fields; their schema mismatch is also present in the unmodified source. These historical differences were retained and documented, not corrected during cleanup.

The public package contains aggregate data only. Obtain controlled ABCD data through its authorized access process and the study-specific derived inputs through the study authors. Generated participant-level outputs belong in a private location; `outputs/` and `data/private/` are ignored by Git.
