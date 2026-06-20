from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from hapf.models.adapters import PatientCodebook
from hapf.models.forecaster import PatientAdaptiveForecaster


@dataclass(frozen=True)
class AdaptationState:
    patient_code: torch.Tensor
    output_bias: torch.Tensor
    losses: list[float]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def gaussian_nll(target: torch.Tensor, location: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    standardized = (target - location) / scale
    return (0.5 * standardized.square() + torch.log(scale)).mean()


def train_population(
    model: PatientAdaptiveForecaster,
    loader: DataLoader,
    epochs: int,
    learning_rate: float,
    device: torch.device,
) -> list[float]:
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    losses: list[float] = []
    for _ in range(epochs):
        running = 0.0
        examples = 0
        for features, target, _ in loader:
            features = features.to(device)
            target = target.to(device)
            optimizer.zero_grad(set_to_none=True)
            location, scale = model(features)
            loss = gaussian_nll(target, location, scale)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            running += float(loss.detach()) * len(features)
            examples += len(features)
        losses.append(running / max(examples, 1))
    return losses


def pretrain_adapters(
    model: PatientAdaptiveForecaster,
    codebook: PatientCodebook,
    loader: DataLoader,
    epochs: int,
    learning_rate: float,
    code_l2: float,
    device: torch.device,
) -> list[float]:
    model.freeze_population()
    model.to(device)
    codebook.to(device)
    parameters = [*model.adapter.parameters(), *codebook.parameters()]
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=1e-4)
    losses: list[float] = []
    model.train()
    for _ in range(epochs):
        running = 0.0
        examples = 0
        for features, target, subject_index in loader:
            if torch.any(subject_index < 0):
                raise ValueError("Adapter pretraining received a subject without a codebook index.")
            features = features.to(device)
            target = target.to(device)
            subject_index = subject_index.to(device)
            optimizer.zero_grad(set_to_none=True)
            code = codebook(subject_index)
            location, scale = model(features, patient_code=code)
            loss = gaussian_nll(target, location, scale) + code_l2 * code.square().mean()
            loss.backward()
            nn.utils.clip_grad_norm_(parameters, max_norm=5.0)
            optimizer.step()
            running += float(loss.detach()) * len(features)
            examples += len(features)
        losses.append(running / max(examples, 1))
    return losses


def adapt_patient(
    model: PatientAdaptiveForecaster,
    loader: DataLoader,
    epochs: int,
    learning_rate: float,
    code_l2: float,
    device: torch.device,
) -> AdaptationState:
    model.freeze_all()
    model.to(device)
    model.eval()
    patient_code = nn.Parameter(torch.zeros(model.patient_code_dim, device=device))
    output_bias = nn.Parameter(torch.zeros(model.horizons, device=device))
    optimizer = torch.optim.Adam([patient_code, output_bias], lr=learning_rate)
    losses: list[float] = []
    for _ in range(epochs):
        running = 0.0
        examples = 0
        for features, target, _ in loader:
            features = features.to(device)
            target = target.to(device)
            optimizer.zero_grad(set_to_none=True)
            code = patient_code.unsqueeze(0).expand(len(features), -1)
            location, scale = model(features, patient_code=code, output_bias=output_bias)
            loss = (
                gaussian_nll(target, location, scale)
                + code_l2 * patient_code.square().mean()
                + 0.01 * output_bias.square().mean()
            )
            loss.backward()
            nn.utils.clip_grad_norm_([patient_code, output_bias], max_norm=5.0)
            optimizer.step()
            running += float(loss.detach()) * len(features)
            examples += len(features)
        losses.append(running / max(examples, 1))
    return AdaptationState(
        patient_code=patient_code.detach(),
        output_bias=output_bias.detach(),
        losses=losses,
    )


@torch.no_grad()
def predict(
    model: PatientAdaptiveForecaster,
    loader: DataLoader,
    device: torch.device,
    patient_code: torch.Tensor | None = None,
    output_bias: torch.Tensor | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    model.to(device)
    model.eval()
    locations: list[np.ndarray] = []
    scales: list[np.ndarray] = []
    for features, _, _ in loader:
        features = features.to(device)
        code = None
        if patient_code is not None:
            code = patient_code.to(device).unsqueeze(0).expand(len(features), -1)
        location, scale = model(
            features,
            patient_code=code,
            output_bias=output_bias.to(device) if output_bias is not None else None,
        )
        locations.append(location.cpu().numpy())
        scales.append(scale.cpu().numpy())
    return np.concatenate(locations), np.concatenate(scales)

