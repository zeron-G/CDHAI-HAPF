from hapf.training.cross_validation import summarize_folds


def _fold(population: list[float], personalized: list[float], deployed: list[float], accepted: bool) -> dict:
    return {
        "metrics": {
            "persistence": {"rmse": [5.0, 6.0]},
            "population": {"rmse": population},
            "personalized": {"rmse": personalized},
            "deployed": {"rmse": deployed},
            "deployed_interval": {"coverage": [0.9, 0.9], "mean_width": [2.0, 3.0]},
        },
        "personalization_gate": {"accepted": accepted},
    }


def test_cross_validation_summary_is_patient_level() -> None:
    folds = [
        _fold([4.0, 5.0], [3.0, 5.5], [3.0, 5.0], True),
        _fold([6.0, 7.0], [7.0, 6.0], [6.0, 6.0], False),
    ]
    summary = summarize_folds(folds, seed=1)
    assert summary["subjects_personalized_improved"] == [1, 1]
    assert summary["subjects_gate_accepted"] == 1
    assert summary["rmse_mean"]["population"] == [5.0, 6.0]
    assert summary["deployed_interval_coverage_mean"] == [0.9, 0.9]
