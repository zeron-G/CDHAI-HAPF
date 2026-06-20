import numpy as np

from hapf.evaluation.metrics import conformal_radius, forecast_metrics, interval_metrics, personalization_gate


def test_forecast_and_conformal_metrics() -> None:
    observed = np.asarray([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])
    predicted = observed + 1.0
    metrics = forecast_metrics(observed, predicted)
    assert metrics["mae"] == [1.0, 1.0]
    radius = conformal_radius(observed, predicted, alpha=0.2)
    assert radius.tolist() == [1.0, 1.0]
    intervals = interval_metrics(observed, predicted, radius)
    assert intervals["coverage"] == [1.0, 1.0]


def test_personalization_gate_rejects_calibration_harm() -> None:
    observed = np.asarray([[1.0], [2.0], [3.0]])
    population = observed + 0.5
    personalized = observed + 1.0
    gate = personalization_gate(observed, population, personalized)
    assert gate["accepted"] is False
    assert gate["status"] == "rejected_population_fallback"


def test_personalization_gate_requires_every_horizon_to_be_noninferior() -> None:
    observed = np.asarray([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    population = observed + np.asarray([1.0, 1.0])
    personalized = observed + np.asarray([0.1, 1.1])
    gate = personalization_gate(observed, population, personalized)
    assert gate["accepted"] is False
    assert gate["all_horizons_noninferior"] is False
