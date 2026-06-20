# HAPF Architecture

## 1. Problem Definition

For patient `i`, variable set `V`, and time `t`, let `x_(i,<=t)` be all
observations available by time `t`, `s_i` be static context, and `y_(i,t+h,v)`
be a future target. The forecasting objective is the conditional distribution:

```text
p(y_(i,t+h,v) | x_(i,<=t), s_i, D_i_support)
```

`D_i_support` contains only the personal observations explicitly assigned to
adaptation. It never includes calibration or test targets.

The normative objective is a separate output derived from forecast residuals:

```text
r_(i,t,h,v) = y_(i,t+h,v) - median_hat_(i,t,h,v)
```

A calibrated residual distribution produces a personal deviation score. A
large deviation means "unexpected under this model and context," not "disease."

## 2. Shared Backbone

Version 0 uses a causal temporal convolutional network because it is compact,
stable on small data, and directly comparable with glucose forecasting work
such as GluNet. A future larger-data version can replace the backbone with a
patch transformer without changing the adaptation contract.

Input channels in the first experiment are:

- Globally standardized glucose.
- First difference of standardized glucose.
- Sine of time of day.
- Cosine of time of day.

Diet, insulin/medication, exercise, missingness, and static covariates are the
next input groups. Each group must retain an observation mask and time since
last observation.

## 3. Factorized Low-Rank Adaptation

At hidden layer `l`:

```text
base_(i,l) = F_l(h_(i,l); W_l)
gate_(i,v,l) = tanh(P_l a_i) * sigmoid(b_(v,l))
delta_(i,v,l) = U_l [gate_(i,v,l) * (V_l base_(i,l))]
h_(i,l+1) = base_(i,l) + alpha_l delta_(i,v,l) / r
```

where:

- `a_i` is a low-dimensional patient code.
- `b_v` is a variable or modality code.
- `U_l` and `V_l` are shared rank-`r` matrices.
- `alpha_l` controls adapter scale.

The parameter count grows approximately with `O(N_patient * d_code +
N_variable * r)` rather than storing a complete network or full LoRA matrices
for every patient-variable pair.

For the initial single-target glucose experiment, `v` denotes the glucose
output channel. The variable factorization becomes testable when additional
targets are introduced.

## 4. Cold Start and Warm Adaptation

The current prototype initializes an unseen patient code at zero and optimizes
it on a chronological support window while freezing the backbone. The full
design adds a context encoder:

```text
a_i = H_psi(s_i, Summary(D_i_support)) + delta_i
```

`H_psi` provides a cold-start code from static context and early history.
`delta_i` is a small residual optimized for the patient. This separates an
inference path for unseen patients from a memorized patient-ID embedding.

## 5. Hierarchical Shrinkage

Personalization is useful only when supported by data. The adaptation objective
is:

```text
L_i = L_forecast(D_i_support) + lambda_i ||a_i||_2^2 + gamma ||bias_i||_2^2
```

`lambda_i` should increase when support data are sparse, irregular, or shifted.
If adaptation does not improve a support-validation split, the deployed code is
reset to zero and the population forecast is used.

The prototype gate requires at least 1% pooled calibration-RMSE improvement and
non-inferiority at every forecast horizon. This threshold is a configurable
engineering guardrail and must be preregistered before confirmatory evaluation.

Future versions should learn cohort and device codes above the patient level:

```text
a_i = a_global + a_cohort(i) + a_patient(i)
```

This prevents device or source-domain artifacts from being mistaken for human
biology.

## 6. Probabilistic Head and Calibration

The prototype predicts Gaussian location and scale for 30- and 60-minute
horizons. A production research version should compare Gaussian, Student-t,
quantile, and evidential heads.

Regardless of head choice, uncertainty is calibrated on a time segment that is
not used to fit the patient adapter. Split conformal residual intervals are the
initial method. Adaptive conformal inference is a later option for temporal
distribution shift.

## 7. Missing Modalities

The model must not assume every patient has every channel. Training should use
modality dropout and explicit masks. A missing event stream means "unobserved,"
not "no event," unless the source data contract guarantees that interpretation.

## 8. Extension Beyond Glucose

Variables need distribution-appropriate heads:

- Continuous: Gaussian, Student-t, or quantile.
- Binary event: Bernoulli hazard.
- Count: negative binomial.
- Time-to-event: survival head.

The shared backbone and factorized adapter can remain common, but evaluation,
calibration, and clinical interpretation must remain target-specific.
