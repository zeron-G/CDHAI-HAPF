# Data Governance

- No patient data, source archives, Parquet files, checkpoints, or generated
  predictions are committed to this public repository.
- Local data are addressed by command-line path and remain in the governed
  workspace.
- Experiments use deidentified `subject_key` values internally and emit random
  evaluation aliases in shareable reports.
- Train, adaptation, calibration, and test boundaries are materialized in a
  split manifest before model fitting.
- Static attributes are used only when their provenance and missingness are
  documented.
- Protected attributes may be used for fairness evaluation. Their use as model
  inputs requires a separate scientific and governance justification.
- Genetic data, if added later, require explicit consent, access controls,
  ancestry-aware validation, and a prohibition on public row-level release.
- Outputs are research forecasts and uncertainty estimates. They are not
  diagnoses, dosing recommendations, or medical advice.

