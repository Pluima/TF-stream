"""Reusable neural blocks for lightweight separator variants."""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualFFNBlock(nn.Module):
    """Residual FFN block used after sequence models."""

    def __init__(self, hidden_size: int, expansion: int = 4, dropout: float = 0.0):
        super().__init__()
        inner_size = int(hidden_size) * max(1, int(expansion))
        self.norm = nn.LayerNorm(hidden_size)
        self.fc1 = nn.Linear(hidden_size, inner_size)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(inner_size, hidden_size)
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return residual + x


class CausalTCNBlock(nn.Module):
    """Conv-TasNet style causal TCN block with residual and skip outputs."""

    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        if kernel_size < 2:
            raise ValueError("kernel_size must be >= 2")
        self.channels = int(channels)
        self.hidden_channels = int(hidden_channels)
        self.kernel_size = int(kernel_size)
        self.dilation = int(dilation)
        self.cache_len = self.dilation * (self.kernel_size - 1)

        self.in_conv = nn.Conv1d(self.channels, self.hidden_channels * 2, kernel_size=1)
        self.dw_conv = nn.Conv1d(
            self.hidden_channels,
            self.hidden_channels,
            kernel_size=self.kernel_size,
            dilation=self.dilation,
            groups=self.hidden_channels,
            padding=0,
        )
        self.norm = nn.GroupNorm(1, self.hidden_channels)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(float(dropout))

        self.res_out = nn.Conv1d(self.hidden_channels, self.channels, kernel_size=1)
        self.skip_out = nn.Conv1d(self.hidden_channels, self.channels, kernel_size=1)

    def _gated_in(self, x: torch.Tensor) -> torch.Tensor:
        h = self.in_conv(x)
        a, b = torch.chunk(h, 2, dim=1)
        return torch.tanh(a) * torch.sigmoid(b)

    def forward_sequence(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self._gated_in(x)
        if self.cache_len > 0:
            h = F.pad(h, (self.cache_len, 0))
        h = self.dw_conv(h)
        h = self.norm(h)
        h = self.act(h)
        h = self.dropout(h)

        residual = self.res_out(h)
        skip = self.skip_out(h)
        return x + residual, skip

    @torch.no_grad()
    def forward_step(
        self,
        x_step: torch.Tensor,
        cache: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self._gated_in(x_step)

        if self.cache_len > 0:
            if cache is None:
                cache = torch.zeros(
                    h.shape[0],
                    self.hidden_channels,
                    self.cache_len,
                    device=h.device,
                    dtype=h.dtype,
                )
            h_cat = torch.cat([cache, h], dim=-1)
        else:
            h_cat = h

        h_new = self.dw_conv(h_cat)
        h_new = self.norm(h_new)
        h_new = self.act(h_new)
        h_new = self.dropout(h_new)

        residual = self.res_out(h_new)
        skip = self.skip_out(h_new)

        if self.cache_len > 0:
            new_cache = h_cat[:, :, -self.cache_len :]
        else:
            new_cache = h_cat[:, :, :0]

        return x_step + residual, skip, new_cache
