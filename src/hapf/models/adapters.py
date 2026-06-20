from __future__ import annotations

import torch
from torch import nn


class FactorizedPatientVariableAdapter(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        patient_code_dim: int,
        rank: int,
        n_variables: int = 1,
        scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.rank = rank
        self.scale = scale
        self.down = nn.Linear(hidden_dim, rank, bias=False)
        self.up = nn.Linear(rank, hidden_dim, bias=False)
        self.patient_gate = nn.Linear(patient_code_dim, rank, bias=False)
        self.variable_gate = nn.Embedding(n_variables, rank)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.variable_gate.weight)

    def forward(
        self,
        hidden: torch.Tensor,
        patient_code: torch.Tensor,
        variable_id: torch.Tensor,
    ) -> torch.Tensor:
        patient_factor = torch.tanh(self.patient_gate(patient_code))
        variable_factor = torch.sigmoid(self.variable_gate(variable_id))
        low_rank = self.down(hidden) * patient_factor * variable_factor
        return hidden + self.up(low_rank) * (self.scale / self.rank)


class PatientCodebook(nn.Module):
    def __init__(self, n_patients: int, code_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(n_patients, code_dim)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

    def forward(self, patient_index: torch.Tensor) -> torch.Tensor:
        return self.embedding(patient_index)

