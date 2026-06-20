import numpy as np
import pandas as pd

from hapf.config import DataConfig
from hapf.data.cgm import build_cgm_windows, chronological_subject_split, fit_standardizer


def _frame() -> pd.DataFrame:
    rows = []
    for subject, offset in (("human-a", 0.0), ("human-b", 30.0)):
        times = pd.date_range("2026-01-01", periods=120, freq="5min")
        values = np.linspace(90.0 + offset, 150.0 + offset, len(times))
        rows.extend(
            {"subject_key": subject, "DT_s": time, "BGValue": value}
            for time, value in zip(times, values, strict=True)
        )
    return pd.DataFrame(rows)


def test_windows_keep_subjects_separate_and_targets_in_future() -> None:
    config = DataConfig(context_steps=12, horizons=(2, 4), stride=2)
    collection = build_cgm_windows(_frame(), config)
    assert set(collection.subject_keys) == {"human-a", "human-b"}
    assert collection.features.shape[1:] == (4, 12)
    assert np.all(collection.targets[:, 0] > collection.last_values)
    assert np.all(collection.targets[:, 1] > collection.targets[:, 0])


def test_gap_crossing_windows_are_rejected() -> None:
    frame = _frame()
    frame = frame.drop(frame[(frame.subject_key == "human-a") & (frame.DT_s == "2026-01-01 03:00")].index)
    config = DataConfig(context_steps=12, horizons=(2, 4), stride=1)
    collection = build_cgm_windows(frame, config)
    subject_indices = np.flatnonzero(collection.subject_keys == "human-a")
    for index in subject_indices:
        values = collection.features[index, 0]
        assert np.all(np.diff(values) > 0)


def test_chronological_split_and_train_only_standardizer() -> None:
    config = DataConfig(context_steps=12, horizons=(2, 4), stride=2)
    collection = build_cgm_windows(_frame(), config)
    split = chronological_subject_split(collection, "human-b", 0.3, 0.2)
    assert collection.target_times[split.adaptation].max() < collection.target_times[split.calibration].min()
    assert collection.target_times[split.calibration].max() < collection.target_times[split.test].min()
    training = np.flatnonzero(collection.subject_keys == "human-a")
    standardizer = fit_standardizer(collection, training)
    assert standardizer.mean < 150.0

