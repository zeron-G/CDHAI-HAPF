from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from hapf.config import ModelConfig
from hapf.models.adapters import FactorizedPatientVariableAdapter
from hapf.models.backbone import CausalTCNBackbone


class PatientAdaptiveForecaster(nn.Module):
    def __init__(self, config: ModelConfig, horizons: int) -> None:
        super().__init__()
        self.patient_code_dim = config.patient_code_dim
        self.horizons = horizons
        self.backbone = CausalTCNBackbone(
            input_channels=config.input_channels,
            hidden_dim=config.hidden_dim,
            levels=config.levels,
            kernel_size=config.kernel_size,
            dropout=config.dropout,
        )
        self.adapter = FactorizedPatientVariableAdapter(
            hidden_dim=config.hidden_dim,
            patient_code_dim=config.patient_code_dim,
            rank=config.adapter_rank,
        )
        self.head = nn.Linear(config.hidden_dim, horizons * 2)

    def forward(
        self,
        inputs: torch.Tensor,
        patient_code: torch.Tensor | None = None,
        output_bias: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.backbone(inputs)
        if patient_code is None:
            patient_code = hidden.new_zeros((hidden.shape[0], self.patient_code_dim))
        variable_id = torch.zeros(hidden.shape[0], dtype=torch.long, device=hidden.device)
        hidden = self.adapter(hidden, patient_code, variable_id)
        statistics = self.head(hidden).view(hidden.shape[0], self.horizons, 2)
        location = statistics[..., 0]
        if output_bias is not None:
            location = location + output_bias
        scale = F.softplus(statistics[..., 1]) + 0.05
        return location, scale

    def freeze_population(self) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        for parameter in self.head.parameters():
            parameter.requires_grad_(False)

    def freeze_all(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad_(False)

