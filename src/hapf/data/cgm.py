from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from hapf.config import DataConfig


@dataclass(frozen=True)
class Standardizer:
    mean: float
    std: float

    def transform_value(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean) / self.std

    def inverse_value(self, values: np.ndarray) -> np.ndarray:
        return values * self.std + self.mean


@dataclass(frozen=True)
class WindowCollection:
    features: np.ndarray
    targets: np.ndarray
    last_values: np.ndarray
    subject_keys: np.ndarray
    target_times: np.ndarray

    def __len__(self) -> int:
        return int(self.features.shape[0])


@dataclass(frozen=True)
class SubjectTimeSplit:
    adaptation: np.ndarray
    calibration: np.ndarray
    test: np.ndarray


class WindowDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        collection: WindowCollection,
        indices: Sequence[int] | np.ndarray,
        subject_index: dict[str, int] | None = None,
    ) -> None:
        self.collection = collection
        self.indices = np.asarray(indices, dtype=np.int64)
        mapping = subject_index or {}
        self.subject_indices = np.asarray(
            [mapping.get(str(collection.subject_keys[index]), -1) for index in self.indices],
            dtype=np.int64,
        )

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        index = self.indices[item]
        return (
            torch.from_numpy(self.collection.features[index]).float(),
            torch.from_numpy(self.collection.targets[index]).float(),
            torch.tensor(self.subject_indices[item], dtype=torch.long),
        )


def load_cgm_parquet(path: str | Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    required = {"subject_key", "DT_s", "BGValue"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"CGM parquet is missing required columns: {sorted(missing)}")
    frame = frame.loc[:, ["subject_key", "DT_s", "BGValue"]].copy()
    frame["DT_s"] = pd.to_datetime(frame["DT_s"], errors="coerce")
    frame["BGValue"] = pd.to_numeric(frame["BGValue"], errors="coerce")
    return frame.dropna(subset=["subject_key", "DT_s", "BGValue"])


def build_cgm_windows(frame: pd.DataFrame, config: DataConfig) -> WindowCollection:
    features: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    last_values: list[float] = []
    subject_keys: list[str] = []
    target_times: list[np.datetime64] = []
    expected_ns = int(pd.Timedelta(minutes=config.expected_interval_minutes).value)
    max_horizon = max(config.horizons)

    for subject_key, subject_frame in frame.groupby("subject_key", sort=True):
        series = (
            subject_frame.groupby("DT_s", as_index=False)["BGValue"]
            .mean()
            .sort_values("DT_s")
            .reset_index(drop=True)
        )
        values = series["BGValue"].to_numpy(dtype=np.float32)
        times = series["DT_s"].to_numpy(dtype="datetime64[ns]")
        if len(values) < config.context_steps + max_horizon:
            continue

        time_ns = times.astype(np.int64)
        bad_step = np.diff(time_ns) != expected_ns
        bad_prefix = np.concatenate(([0], np.cumsum(bad_step.astype(np.int64))))
        deltas = np.diff(values, prepend=values[0]).astype(np.float32)
        minute_of_day = (
            pd.DatetimeIndex(times).hour.to_numpy() * 60 + pd.DatetimeIndex(times).minute.to_numpy()
        )
        phase = 2.0 * np.pi * minute_of_day / 1440.0

        last_start = len(values) - config.context_steps - max_horizon
        for start in range(0, last_start + 1, config.stride):
            final_target = start + config.context_steps - 1 + max_horizon
            if bad_prefix[final_target] - bad_prefix[start] != 0:
                continue
            context_end = start + config.context_steps
            target_indices = [start + config.context_steps - 1 + horizon for horizon in config.horizons]
            feature_window = np.stack(
                [
                    values[start:context_end],
                    deltas[start:context_end],
                    np.sin(phase[start:context_end]).astype(np.float32),
                    np.cos(phase[start:context_end]).astype(np.float32),
                ],
                axis=0,
            )
            features.append(feature_window)
            targets.append(values[target_indices])
            last_values.append(float(values[context_end - 1]))
            subject_keys.append(str(subject_key))
            target_times.append(times[target_indices[-1]])

    if not features:
        raise ValueError("No contiguous CGM windows could be constructed.")
    return WindowCollection(
        features=np.stack(features).astype(np.float32),
        targets=np.stack(targets).astype(np.float32),
        last_values=np.asarray(last_values, dtype=np.float32),
        subject_keys=np.asarray(subject_keys, dtype=str),
        target_times=np.asarray(target_times, dtype="datetime64[ns]"),
    )


def select_heldout_subject(collection: WindowCollection, requested: str | None = None) -> str:
    subjects, counts = np.unique(collection.subject_keys, return_counts=True)
    if requested is not None:
        if requested not in set(subjects):
            raise ValueError("Requested held-out subject is not present in the window collection.")
        return requested
    eligible = sorted(zip(subjects.tolist(), counts.tolist(), strict=True), key=lambda item: (item[1], item[0]))
    return str(eligible[len(eligible) // 2][0])


def chronological_subject_split(
    collection: WindowCollection,
    subject_key: str,
    adaptation_fraction: float,
    calibration_fraction: float,
) -> SubjectTimeSplit:
    indices = np.flatnonzero(collection.subject_keys == subject_key)
    if len(indices) < 30:
        raise ValueError("Held-out subject needs at least 30 windows for adaptation, calibration, and testing.")
    ordered = indices[np.argsort(collection.target_times[indices])]
    adaptation_end = max(1, int(len(ordered) * adaptation_fraction))
    calibration_end = adaptation_end + max(1, int(len(ordered) * calibration_fraction))
    calibration_end = min(calibration_end, len(ordered) - 1)
    return SubjectTimeSplit(
        adaptation=ordered[:adaptation_end],
        calibration=ordered[adaptation_end:calibration_end],
        test=ordered[calibration_end:],
    )


def fit_standardizer(collection: WindowCollection, indices: np.ndarray) -> Standardizer:
    values = collection.features[indices, 0, :].reshape(-1).astype(np.float64)
    mean = float(np.mean(values))
    std = float(np.std(values))
    if not np.isfinite(std) or std < 1e-6:
        raise ValueError("Population training glucose has zero or invalid variance.")
    return Standardizer(mean=mean, std=std)


def normalize_collection(collection: WindowCollection, standardizer: Standardizer) -> WindowCollection:
    features = collection.features.copy()
    features[:, 0, :] = standardizer.transform_value(features[:, 0, :])
    features[:, 1, :] = features[:, 1, :] / standardizer.std
    targets = standardizer.transform_value(collection.targets)
    return WindowCollection(
        features=features.astype(np.float32),
        targets=targets.astype(np.float32),
        last_values=collection.last_values.copy(),
        subject_keys=collection.subject_keys.copy(),
        target_times=collection.target_times.copy(),
    )
