from __future__ import annotations

import torch
from torch import nn


class CausalConv1d(nn.Conv1d):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int = 1) -> None:
        self.causal_padding = (kernel_size - 1) * dilation
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            padding=self.causal_padding,
            dilation=dilation,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = super().forward(inputs)
        return output[..., : -self.causal_padding] if self.causal_padding else output


class TemporalResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            CausalConv1d(hidden_dim, hidden_dim, kernel_size, dilation),
            nn.GELU(),
            nn.Dropout(dropout),
            CausalConv1d(hidden_dim, hidden_dim, kernel_size, dilation),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = inputs + self.network(inputs)
        return self.norm(hidden.transpose(1, 2)).transpose(1, 2)


class CausalTCNBackbone(nn.Module):
    def __init__(
        self,
        input_channels: int,
        hidden_dim: int,
        levels: int,
        kernel_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Conv1d(input_channels, hidden_dim, kernel_size=1)
        self.blocks = nn.Sequential(
            *[
                TemporalResidualBlock(hidden_dim, kernel_size, dilation=2**level, dropout=dropout)
                for level in range(levels)
            ]
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.blocks(self.input_projection(inputs))
        return hidden[..., -1]
