# A-User-Store Sample Fitness Audit

Audit date: 2026-06-20.

## Inventory

The local composite produced by CDHAI_June contains:

| Asset | Rows | Human coverage |
| --- | ---: | ---: |
| Patient table | 20 | 15 subjects |
| CGM5Min | 138,802 | 12 subjects, 17 PatientIDs |
| Diet5Min | 926 | 12 subjects |
| Exercise5Min | 591 | 6 subjects |
| Med5Min | 1,976 | 8 subjects |

The CGM sources are CGMacros, OhioT1DM, and WellDoc2022CGM. Every CGM series
has a median five-minute interval and at least seven days of observations.
Seven PatientID series span at least 30 days. Fourteen of 17 PatientID series
have at least 80% coverage of a complete five-minute grid; one long WellDoc
series has much lower coverage and requires explicit missingness handling.

## Available Personal Context

- Gender: available in all 20 patient rows.
- Year of birth: available in 10 of 20 rows.
- Disease type: available in all rows.
- Height: available in 17 of 20 human-summary rows.
- Weight: available in 18 of 20 rows.
- Timezone/source cohort: available.

Race, ancestry, genotype, and genomic variant data are not present. The sample
therefore cannot test claims about genetic personalization.

## Identity Caveat

There are 15 stable subjects but 20 PatientIDs. OhioT1DM subjects can have two
PatientIDs. The machine-learning identity and outer split key must be
`subject_key`. `PatientID + DT_s` remains a valid record alignment key, but
PatientID alone is not a safe human-level split key.

## What This Sample Can Test

- End-to-end patient adaptation mechanics.
- Glucose-only 30/60-minute forecasting.
- Support-duration sensitivity.
- Cross-source robustness as an exploratory analysis.
- Incremental value of partially available diet, medication, and exercise
  streams.

## What It Cannot Establish

- A large medical foundation model.
- Robust demographic, racial, or genomic personalization.
- Clinical effectiveness or treatment benefit.
- Generalization to broad disease populations.
- Stable variable-specific adapters for many output variables.

## Data Quality Work Before Confirmatory Experiments

- Reconcile multiple PatientIDs to one stable subject timeline.
- Verify source-specific date transformations and timezone semantics.
- Quantify long gaps and sensor changes per subject.
- Define whether missing diet/exercise/medication means unobserved or absent.
- Validate units for glucose, carbohydrate, medication dose, height, and weight.
- Create a governed subject-level split manifest.

