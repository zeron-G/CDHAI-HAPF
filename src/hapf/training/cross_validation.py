from __future__ import annotations

import gc
import json
from pathlib import Path

import numpy as np
import torch

from hapf.config import ExperimentConfig
from hapf.data.cgm import build_cgm_windows, load_cgm_parquet
from hapf.training.experiment import run_sample_experiment


def run_cross_validation(
    data_path: str | Path,
    output_dir: str | Path,
    config: ExperimentConfig,
    device_name: str | None = None,
) -> dict[str, object]:
    collection = build_cgm_windows(load_cgm_parquet(data_path), config.data)
    subjects = sorted(set(collection.subject_keys.tolist()))
    root = Path(output_dir)
    folds: list[dict[str, object]] = []
    for index, subject in enumerate(subjects, start=1):
        result = run_sample_experiment(
            data_path=data_path,
            output_dir=root / f"fold_{index:02d}",
            config=config,
            heldout_subject=subject,
            device_name=device_name,
        )
        folds.append(
            {
                "fold": f"fold_{index:02d}",
                "metrics": result["metrics"],
                "personalization_gate": result["personalization_gate"],
                "data": {
                    "windows_adaptation": result["data"]["windows_adaptation"],
                    "windows_calibration": result["data"]["windows_calibration"],
                    "windows_test": result["data"]["windows_test"],
                },
            }
        )
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    aggregate = summarize_folds(folds, config.seed)
    payload: dict[str, object] = {
        "status": "exploratory_leave_one_subject_out",
        "source_file": Path(data_path).name,
        "subject_count": len(subjects),
        "horizon_minutes": [
            horizon * config.data.expected_interval_minutes for horizon in config.data.horizons
        ],
        "aggregate": aggregate,
        "folds": folds,
        "limitations": [
            "Exploratory protocol and hyperparameters were not externally preregistered.",
            "The sample contains only 12 CGM subjects across heterogeneous source cohorts.",
            "Only glucose history and clock features are used in this first model.",
            "No race or genomic variables are available.",
        ],
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "cross_validation.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (root / "cross_validation.md").write_text(_report(payload), encoding="utf-8")
    return payload


def summarize_folds(folds: list[dict[str, object]], seed: int) -> dict[str, object]:
    models = ("persistence", "population", "personalized", "deployed")
    rmse = {
        model: np.asarray([fold["metrics"][model]["rmse"] for fold in folds], dtype=float)
        for model in models
    }
    personalized_delta = rmse["personalized"] - rmse["population"]
    deployed_delta = rmse["deployed"] - rmse["population"]
    population_denominator = np.maximum(rmse["population"], 1e-8)
    accepted = np.asarray([bool(fold["personalization_gate"]["accepted"]) for fold in folds])
    interval_coverage = np.asarray(
        [fold["metrics"]["deployed_interval"]["coverage"] for fold in folds],
        dtype=float,
    )
    interval_width = np.asarray(
        [fold["metrics"]["deployed_interval"]["mean_width"] for fold in folds],
        dtype=float,
    )
    return {
        "rmse_mean": {model: values.mean(axis=0).tolist() for model, values in rmse.items()},
        "rmse_median": {model: np.median(values, axis=0).tolist() for model, values in rmse.items()},
        "personalized_minus_population_median": np.median(personalized_delta, axis=0).tolist(),
        "personalized_minus_population_median_ci95": _bootstrap_median_ci(personalized_delta, seed),
        "deployed_minus_population_median": np.median(deployed_delta, axis=0).tolist(),
        "personalized_relative_rmse_improvement_mean_percent": (
            100.0 * (rmse["population"] - rmse["personalized"]) / population_denominator
        ).mean(axis=0).tolist(),
        "subjects_personalized_improved": (personalized_delta < 0).sum(axis=0).astype(int).tolist(),
        "subjects_deployed_improved": (deployed_delta < 0).sum(axis=0).astype(int).tolist(),
        "personalized_worst_degradation": np.max(personalized_delta, axis=0).tolist(),
        "deployed_worst_degradation": np.max(deployed_delta, axis=0).tolist(),
        "subjects_gate_accepted": int(accepted.sum()),
        "accepted_subjects_improved_all_horizons_on_test": int(
            np.sum(accepted & np.all(personalized_delta < 0, axis=1))
        ),
        "rejected_subjects_that_improved_all_horizons_on_test": int(
            np.sum(~accepted & np.all(personalized_delta < 0, axis=1))
        ),
        "deployed_interval_coverage_mean": interval_coverage.mean(axis=0).tolist(),
        "deployed_interval_width_mean": interval_width.mean(axis=0).tolist(),
    }


def _bootstrap_median_ci(values: np.ndarray, seed: int, repetitions: int = 5_000) -> list[list[float]]:
    random = np.random.default_rng(seed)
    sample_indices = random.integers(0, len(values), size=(repetitions, len(values)))
    estimates = np.median(values[sample_indices], axis=1)
    lower = np.quantile(estimates, 0.025, axis=0)
    upper = np.quantile(estimates, 0.975, axis=0)
    return np.stack([lower, upper], axis=1).astype(float).tolist()


def _report(payload: dict[str, object]) -> str:
    aggregate = payload["aggregate"]
    horizons = payload["horizon_minutes"]
    median_ci = [
        [round(bound, 3) for bound in interval]
        for interval in aggregate["personalized_minus_population_median_ci95"]
    ]
    lines = [
        "# HAPF Exploratory Leave-One-Subject-Out Report",
        "",
        "> Engineering evidence only. This run is not a clinical validation or a preregistered confirmatory study.",
        "",
        f"Subjects: {payload['subject_count']}",
        "",
        "## Mean RMSE",
        "",
        "| Model | " + " | ".join(f"{horizon} min" for horizon in horizons) + " |",
        "| --- | " + " | ".join("---:" for _ in horizons) + " |",
    ]
    for model, values in aggregate["rmse_mean"].items():
        lines.append(f"| {model} | " + " | ".join(f"{value:.3f}" for value in values) + " |")
    lines.extend(
        [
            "",
            "## Personalization Summary",
            "",
            f"- Subjects improved by raw personalization: {aggregate['subjects_personalized_improved']}",
            f"- Calibration gates accepting personalization: {aggregate['subjects_gate_accepted']}",
            "- Mean relative RMSE improvement (%): "
            f"{[round(value, 3) for value in aggregate['personalized_relative_rmse_improvement_mean_percent']]}",
            "- Median personalized minus population RMSE: "
            f"{[round(value, 3) for value in aggregate['personalized_minus_population_median']]}",
            f"- Bootstrap 95% CI for median difference: {median_ci}",
            "- Mean deployed interval coverage: "
            f"{[round(value, 3) for value in aggregate['deployed_interval_coverage_mean']]}",
            "- A negative personalized-minus-population value favors personalization.",
            "",
        ]
    )
    return "\n".join(lines)
