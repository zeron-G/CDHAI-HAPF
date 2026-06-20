from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from hapf.config import ExperimentConfig
from hapf.data.cgm import (
    WindowDataset,
    build_cgm_windows,
    chronological_subject_split,
    fit_standardizer,
    load_cgm_parquet,
    normalize_collection,
    select_heldout_subject,
)
from hapf.evaluation.metrics import conformal_radius, forecast_metrics, interval_metrics, personalization_gate
from hapf.models.adapters import PatientCodebook
from hapf.models.forecaster import PatientAdaptiveForecaster
from hapf.training.engine import adapt_patient, predict, pretrain_adapters, set_seed, train_population


def run_sample_experiment(
    data_path: str | Path,
    output_dir: str | Path,
    config: ExperimentConfig,
    heldout_subject: str | None = None,
    device_name: str | None = None,
) -> dict[str, object]:
    set_seed(config.seed)
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    raw = build_cgm_windows(load_cgm_parquet(data_path), config.data)
    heldout = select_heldout_subject(raw, requested=heldout_subject)
    split = chronological_subject_split(
        raw,
        heldout,
        adaptation_fraction=config.data.adaptation_fraction,
        calibration_fraction=config.data.calibration_fraction,
    )
    population_indices = np.flatnonzero(raw.subject_keys != heldout)
    population_indices = _balanced_indices(
        raw.subject_keys,
        population_indices,
        config.data.max_windows_per_subject,
        config.seed,
    )
    standardizer = fit_standardizer(raw, population_indices)
    normalized = normalize_collection(raw, standardizer)

    train_subjects = sorted(set(normalized.subject_keys[population_indices].tolist()))
    subject_index = {subject: index for index, subject in enumerate(train_subjects)}
    train_dataset = WindowDataset(normalized, population_indices, subject_index=subject_index)
    adaptation_dataset = WindowDataset(normalized, split.adaptation)
    calibration_dataset = WindowDataset(normalized, split.calibration)
    test_dataset = WindowDataset(normalized, split.test)
    train_loader = _loader(train_dataset, config.training.batch_size, shuffle=True)
    adaptation_loader = _loader(adaptation_dataset, config.training.batch_size, shuffle=True)
    calibration_loader = _loader(calibration_dataset, config.training.batch_size, shuffle=False)
    test_loader = _loader(test_dataset, config.training.batch_size, shuffle=False)

    model = PatientAdaptiveForecaster(config.model, horizons=len(config.data.horizons))
    population_loss = train_population(
        model,
        train_loader,
        epochs=config.training.population_epochs,
        learning_rate=config.training.learning_rate,
        device=device,
    )
    codebook = PatientCodebook(len(train_subjects), config.model.patient_code_dim)
    adapter_loss = pretrain_adapters(
        model,
        codebook,
        train_loader,
        epochs=config.training.adapter_pretrain_epochs,
        learning_rate=config.training.learning_rate,
        code_l2=config.training.patient_code_l2,
        device=device,
    )

    population_calibration_normalized, _ = predict(model, calibration_loader, device=device)
    population_test_normalized, _ = predict(model, test_loader, device=device)
    patient_state = adapt_patient(
        model,
        adaptation_loader,
        epochs=config.training.patient_adaptation_epochs,
        learning_rate=config.training.adaptation_learning_rate,
        code_l2=config.training.patient_code_l2,
        device=device,
    )
    personalized_calibration_normalized, _ = predict(
        model,
        calibration_loader,
        device=device,
        patient_code=patient_state.patient_code,
        output_bias=patient_state.output_bias,
    )
    personalized_test_normalized, _ = predict(
        model,
        test_loader,
        device=device,
        patient_code=patient_state.patient_code,
        output_bias=patient_state.output_bias,
    )

    observed_test = raw.targets[split.test]
    observed_calibration = raw.targets[split.calibration]
    population_calibration = standardizer.inverse_value(population_calibration_normalized)
    population_test = standardizer.inverse_value(population_test_normalized)
    personalized_test = standardizer.inverse_value(personalized_test_normalized)
    personalized_calibration = standardizer.inverse_value(personalized_calibration_normalized)
    persistence_test = np.repeat(raw.last_values[split.test, None], len(config.data.horizons), axis=1)
    gate = personalization_gate(
        observed_calibration,
        population_calibration,
        personalized_calibration,
        minimum_relative_improvement=config.training.personalization_gate_min_relative_improvement,
    )
    if gate["accepted"]:
        deployed_calibration = personalized_calibration
        deployed_test = personalized_test
        deployed_model = "personalized"
    else:
        deployed_calibration = population_calibration
        deployed_test = population_test
        deployed_model = "population_fallback"
    radius = conformal_radius(
        observed_calibration,
        deployed_calibration,
        alpha=config.training.interval_alpha,
    )

    results: dict[str, object] = {
        "status": "exploratory_single_holdout",
        "heldout_alias": "heldout_subject_001",
        "device": str(device),
        "config": asdict(config),
        "data": {
            "source_file": Path(data_path).name,
            "subjects_total": len(set(raw.subject_keys.tolist())),
            "population_subjects": len(train_subjects),
            "windows_population": len(population_indices),
            "windows_adaptation": len(split.adaptation),
            "windows_calibration": len(split.calibration),
            "windows_test": len(split.test),
            "normalization_mean": standardizer.mean,
            "normalization_std": standardizer.std,
        },
        "model": {
            "parameters_total": sum(parameter.numel() for parameter in model.parameters()),
            "patient_state_parameters": int(patient_state.patient_code.numel() + patient_state.output_bias.numel()),
            "horizon_minutes": [
                horizon * config.data.expected_interval_minutes for horizon in config.data.horizons
            ],
        },
        "metrics": {
            "persistence": forecast_metrics(observed_test, persistence_test),
            "population": forecast_metrics(observed_test, population_test),
            "personalized": forecast_metrics(observed_test, personalized_test),
            "deployed": forecast_metrics(observed_test, deployed_test),
            "deployed_interval": interval_metrics(observed_test, deployed_test, radius),
            "conformal_radius": radius.astype(float).tolist(),
        },
        "personalization_gate": {**gate, "deployed_model": deployed_model},
        "training": {
            "population_loss": population_loss,
            "adapter_pretrain_loss": adapter_loss,
            "patient_adaptation_loss": patient_state.losses,
        },
        "limitations": [
            "Single held-out subject; not confirmatory evidence.",
            "Glucose history and clock features only in the first executable prototype.",
            "No race or genomic variables are available in the sample.",
            "Nested subject-level cross-validation and baseline expansion remain required.",
        ],
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (output / "report.md").write_text(_report(results), encoding="utf-8")
    return results


def _balanced_indices(
    subject_keys: np.ndarray,
    indices: np.ndarray,
    maximum: int,
    seed: int,
) -> np.ndarray:
    random = np.random.default_rng(seed)
    selected: list[np.ndarray] = []
    for subject in sorted(set(subject_keys[indices].tolist())):
        subject_indices = indices[subject_keys[indices] == subject]
        if len(subject_indices) > maximum:
            subject_indices = np.sort(random.choice(subject_indices, size=maximum, replace=False))
        selected.append(subject_indices)
    return np.concatenate(selected)


def _loader(dataset: WindowDataset, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def _report(results: dict[str, object]) -> str:
    data = results["data"]
    model = results["model"]
    metrics = results["metrics"]
    gate = results["personalization_gate"]
    horizons = model["horizon_minutes"]
    lines = [
        "# HAPF Exploratory Sample Report",
        "",
        "> This is a single-holdout engineering result, not clinical or confirmatory evidence.",
        "",
        "## Split",
        "",
        f"- Population subjects: {data['population_subjects']}",
        f"- Population windows: {data['windows_population']}",
        f"- Adaptation windows: {data['windows_adaptation']}",
        f"- Calibration windows: {data['windows_calibration']}",
        f"- Test windows: {data['windows_test']}",
        f"- Personalization gate: {gate['status']}",
        "",
        "## RMSE by Horizon",
        "",
        "| Model | " + " | ".join(f"{horizon} min" for horizon in horizons) + " |",
        "| --- | " + " | ".join("---:" for _ in horizons) + " |",
    ]
    for name in ("persistence", "population", "personalized", "deployed"):
        values = metrics[name]["rmse"]
        lines.append(f"| {name} | " + " | ".join(f"{value:.3f}" for value in values) + " |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This run only checks that the population-to-patient adaptation path is executable. "
            "A conclusion requires nested leave-one-subject-out evaluation and all preregistered baselines.",
            "",
        ]
    )
    return "\n".join(lines)
