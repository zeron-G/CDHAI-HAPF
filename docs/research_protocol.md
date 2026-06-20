# Research Protocol

## Primary Question

Does a shared causal forecasting model with a low-dimensional, regularized
patient adapter improve held-out subject forecasts over the same population
model without adaptation?

## Primary Hypothesis

Across held-out subjects, HAPF reduces 30- and 60-minute RMSE relative to the
population model. The patient, not the window, is the unit of inference.

## Secondary Questions

- How many days of support data are needed before adaptation helps?
- Which subjects are harmed by adaptation, and can a validation gate detect
  those cases?
- Does patient adaptation improve interval calibration or only point error?
- Are gains retained across CGMacros, OhioT1DM, and WellDoc source domains?
- Do event channels add value after glucose history is accounted for?

## Falsification Criteria

The main idea is not supported by this sample if:

- Median patient-level improvement is non-positive.
- Improvement disappears against output-bias or last-layer tuning.
- Gains depend on leakage through PatientID, normalization, or future events.
- Personalized intervals are narrower but under-cover their nominal level.
- Benefits occur only for subjects represented during population training.

## Data Split

1. Group all records by stable human `subject_key`.
2. Hold out complete subjects from population and adapter pretraining.
3. For each held-out subject, sort windows by target time.
4. Assign the earliest segment to adaptation, the next segment to calibration,
   and the final segment to test.
5. Fit normalization only on population-training subjects.

Nested subject-level cross-validation is the target protocol. The initial smoke
experiment may run a single held-out subject and is not confirmatory evidence.

## Baselines

- Persistence.
- Population causal TCN.
- Population model with output bias tuned on support.
- Last-layer tuning.
- Full fine-tuning with weight decay and early stopping.
- Patient-only TCN.
- Shared multitask model with subject-specific head.
- MAML-style fast adaptation.
- HAPF patient code with factorized low-rank adapter.

## Metrics

Per subject and horizon:

- MAE and RMSE in mg/dL.
- Mean signed error and lag diagnostics.
- Nominal 90% interval coverage and mean interval width.
- Hypoglycemia and hyperglycemia event sensitivity when event counts permit.
- Clarke or consensus error-grid proportions when method assumptions are met.

Across subjects:

- Median paired improvement and bootstrap confidence interval.
- Wilcoxon signed-rank test as a secondary summary.
- Proportion of subjects improved.
- Worst-subject degradation.
- Holm correction for multiple horizons and primary model comparisons.

## Ablations

- Patient code removed.
- Variable code removed.
- Adapter rank in `{1, 2, 4, 8, 16}`.
- Static conditioner removed.
- Event channels removed by group.
- Context length and support duration.
- Adapter regularization and fallback gate.
- Cohort/device adapter removed.

## Reporting Rules

- Report all held-out subjects, not only successful examples.
- Separate exploratory single-split results from nested cross-validation.
- Publish split manifests and seeds without patient identifiers.
- Do not infer causality from predictive feature importance.
- Do not translate model deviations into diagnoses or treatment advice.

