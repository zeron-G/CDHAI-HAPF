# Exploratory Proof of Concept

Run date: 2026-06-20.

This document records the first executable HAPF experiment on the local
A-User-Store CGM composite. It is engineering evidence, not a clinical result
or a preregistered confirmatory study.

## Protocol

- 12 unique subjects with usable CGM windows.
- Leave one complete `subject_key` out at a time.
- Refit the population model independently in every fold.
- Use the held-out subject's earliest 30% of windows for adaptation, the next
  15% for calibration and gate selection, and the remainder for testing.
- Use 24 five-minute context steps and predict 30 and 60 minutes ahead.
- Cap population-training windows at 5,000 per subject.
- Fit normalization only on population-training subjects.
- Patient state contains 12 code parameters plus two horizon biases.
- Accept personalization only if calibration RMSE improves by at least 1% and
  no horizon degrades; otherwise deploy the population model.

## Initial Results

| Model | Mean RMSE 30 min | Mean RMSE 60 min |
| --- | ---: | ---: |
| Persistence | 20.893 | 32.872 |
| Population TCN | 18.724 | 30.674 |
| Raw personalized HAPF | 18.507 | 29.721 |
| Gated deployment | 18.492 | 29.696 |

Raw personalization improved RMSE for 10 of 12 subjects at 30 minutes and 8
of 12 at 60 minutes. The calibration gate accepted personalization in 6 of 12
folds.

The patient-level median personalized-minus-population RMSE was -0.076 mg/dL
at 30 minutes and -0.333 mg/dL at 60 minutes. Exploratory bootstrap 95%
intervals were [-0.624, -0.017] and [-1.669, 0.023], respectively. The
60-minute interval crosses zero.

## Interpretation

The experiment supports continued investigation, not a strong claim. The
population model captures most of the gain over persistence. Personalization
adds a small average benefit, is heterogeneous across subjects, and requires a
fallback gate. The 30-minute result is more consistent across subjects; the
60-minute estimate remains uncertain.

## Required Next Steps

1. Add output-bias-only, last-layer, full-fine-tune, multitask, MAML, and
   patient-only baselines.
2. Separate hyperparameter development subjects from final evaluation subjects.
3. Add diet, medication/insulin, and exercise channels with observation masks.
4. Add source-cohort and device adapters to avoid treating domain shift as
   personal biology.
5. Repeat with larger governed WellDoc/AIDataStore cohorts.
6. Evaluate event sensitivity, clinical error grids, calibration by subject,
   and worst-subject harm.

