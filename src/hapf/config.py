from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DataConfig:
    context_steps: int = 24
    horizons: tuple[int, ...] = (6, 12)
    stride: int = 3
    expected_interval_minutes: int = 5
    adaptation_fraction: float = 0.30
    calibration_fraction: float = 0.15
    max_windows_per_subject: int = 5_000


@dataclass(frozen=True)
class ModelConfig:
    input_channels: int = 4
    hidden_dim: int = 48
    levels: int = 3
    kernel_size: int = 3
    dropout: float = 0.10
    patient_code_dim: int = 12
    adapter_rank: int = 8


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 512
    population_epochs: int = 8
    adapter_pretrain_epochs: int = 4
    patient_adaptation_epochs: int = 40
    learning_rate: float = 0.001
    adaptation_learning_rate: float = 0.03
    patient_code_l2: float = 0.01
    interval_alpha: float = 0.10
    personalization_gate_min_relative_improvement: float = 0.01


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = 41
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


def load_config(path: str | Path) -> ExperimentConfig:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return ExperimentConfig(
        seed=int(payload.get("seed", 41)),
        data=DataConfig(**_normalize_tuple(payload.get("data", {}), "horizons")),
        model=ModelConfig(**payload.get("model", {})),
        training=TrainingConfig(**payload.get("training", {})),
    )


def _normalize_tuple(payload: dict[str, Any], key: str) -> dict[str, Any]:
    normalized = dict(payload)
    if key in normalized:
        normalized[key] = tuple(int(value) for value in normalized[key])
    return normalized
