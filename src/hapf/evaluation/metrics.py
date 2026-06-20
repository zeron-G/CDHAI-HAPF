from __future__ import annotations

import math

import numpy as np


def forecast_metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, list[float]]:
    error = predicted - observed
    return {
        "mae": np.mean(np.abs(error), axis=0).astype(float).tolist(),
        "rmse": np.sqrt(np.mean(np.square(error), axis=0)).astype(float).tolist(),
        "mean_error": np.mean(error, axis=0).astype(float).tolist(),
    }


def conformal_radius(observed: np.ndarray, predicted: np.ndarray, alpha: float) -> np.ndarray:
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be between zero and one.")
    residual = np.abs(observed - predicted)
    n = residual.shape[0]
    quantile = min(1.0, math.ceil((n + 1) * (1.0 - alpha)) / n)
    return np.quantile(residual, quantile, axis=0, method="higher")


def interval_metrics(observed: np.ndarray, predicted: np.ndarray, radius: np.ndarray) -> dict[str, list[float]]:
    lower = predicted - radius
    upper = predicted + radius
    return {
        "coverage": np.mean((observed >= lower) & (observed <= upper), axis=0).astype(float).tolist(),
        "mean_width": np.broadcast_to(2.0 * radius, observed.shape).mean(axis=0).astype(float).tolist(),
    }


def personalization_gate(
    observed: np.ndarray,
    population: np.ndarray,
    personalized: np.ndarray,
    minimum_relative_improvement: float = 0.0,
) -> dict[str, object]:
    population_by_horizon = np.sqrt(np.mean(np.square(population - observed), axis=0))
    personalized_by_horizon = np.sqrt(np.mean(np.square(personalized - observed), axis=0))
    population_rmse = float(np.sqrt(np.mean(np.square(population - observed))))
    personalized_rmse = float(np.sqrt(np.mean(np.square(personalized - observed))))
    relative_improvement = (population_rmse - personalized_rmse) / max(population_rmse, 1e-8)
    all_horizons_noninferior = bool(np.all(personalized_by_horizon <= population_by_horizon))
    accepted = relative_improvement >= minimum_relative_improvement and all_horizons_noninferior
    return {
        "status": "accepted" if accepted else "rejected_population_fallback",
        "accepted": accepted,
        "population_calibration_rmse": population_rmse,
        "personalized_calibration_rmse": personalized_rmse,
        "relative_improvement": relative_improvement,
        "all_horizons_noninferior": all_horizons_noninferior,
        "population_rmse_by_horizon": population_by_horizon.astype(float).tolist(),
        "personalized_rmse_by_horizon": personalized_by_horizon.astype(float).tolist(),
    }
