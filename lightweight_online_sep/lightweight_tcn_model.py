from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualFFNBlock(nn.Module):
    """Residual FFN block used after recurrent layers."""

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

        # 1x1 bottleneck with gated linear unit.
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
        """
        Args:
            x: [B, C, T]
        Returns:
            residual_out: [B, C, T]
            skip_out: [B, C, T]
        """
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
        """
        Args:
            x_step: [B, C, 1]
            cache: [B, H, cache_len] or None
        Returns:
            residual_out: [B, C, 1]
            skip_out: [B, C, 1]
            new_cache: [B, H, cache_len]
        """
        h = self._gated_in(x_step)  # [B, H, 1]

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


class LightweightCausalSeparator(nn.Module):
    """
    Streaming-capable separator.

    - architecture='tcn': advanced causal Conv-TasNet-style temporal modeling.
    - architecture='gru': legacy GRU baseline (kept for checkpoint compatibility).
    """

    def __init__(
        self,
        num_speakers: int = 2,
        n_fft: int = 256,
        hop_length: int = 128,
        win_length: int = 256,
        # Shared/general
        architecture: str = "tcn",
        dropout: float = 0.0,
        mask_hidden_size: int = 0,
        mask_head_layers: int = 1,
        # GRU branch
        hidden_size: int = 128,
        num_layers: int = 2,
        post_ffn_blocks: int = 1,
        ffn_expansion: int = 2,
        # TCN branch
        bottleneck_size: int = 256,
        tcn_hidden_size: int = 512,
        tcn_kernel_size: int = 3,
        tcn_blocks: int = 8,
        tcn_repeats: int = 2,
    ):
        super().__init__()
        if num_speakers < 2:
            raise ValueError("num_speakers must be >= 2")
        if hop_length > win_length:
            raise ValueError("hop_length must be <= win_length")
        if architecture not in {"gru", "tcn"}:
            raise ValueError(f"Unsupported architecture: {architecture}")

        self.num_speakers = int(num_speakers)
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.win_length = int(win_length)
        self.architecture = str(architecture)

        self.dropout = float(dropout)
        self.mask_hidden_size = int(mask_hidden_size)
        self.mask_head_layers = max(1, int(mask_head_layers))

        # GRU parameters
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.post_ffn_blocks = int(post_ffn_blocks)
        self.ffn_expansion = int(ffn_expansion)

        # TCN parameters
        self.bottleneck_size = int(bottleneck_size)
        self.tcn_hidden_size = int(tcn_hidden_size)
        self.tcn_kernel_size = int(tcn_kernel_size)
        self.tcn_blocks = int(tcn_blocks)
        self.tcn_repeats = int(tcn_repeats)

        self.freq_bins = self.n_fft // 2 + 1

        # Use Hamming window (non-zero boundary) to satisfy center=False iSTFT overlap-add checks.
        self.register_buffer("analysis_window", torch.hamming_window(self.win_length), persistent=False)

        if self.architecture == "gru":
            self._init_gru_modules()
        else:
            self._init_tcn_modules()

    def _build_mask_head(self, in_dim: int) -> nn.Module:
        out_dim = self.num_speakers * self.freq_bins
        hidden_dim = self.mask_hidden_size if self.mask_hidden_size > 0 else in_dim
        if self.mask_head_layers <= 1:
            return nn.Linear(in_dim, out_dim)

        layers: List[nn.Module] = []
        dim = in_dim
        for _ in range(self.mask_head_layers - 1):
            layers.append(nn.Linear(dim, hidden_dim))
            layers.append(nn.SiLU())
            if self.dropout > 0.0:
                layers.append(nn.Dropout(self.dropout))
            dim = hidden_dim
        layers.append(nn.Linear(dim, out_dim))
        return nn.Sequential(*layers)

    def _init_gru_modules(self):
        in_proj_layers = [
            nn.LayerNorm(self.freq_bins),
            nn.Linear(self.freq_bins, self.hidden_size),
            nn.SiLU(),
        ]
        if self.dropout > 0.0:
            in_proj_layers.append(nn.Dropout(self.dropout))
        self.in_proj = nn.Sequential(*in_proj_layers)

        self.rnn = nn.GRU(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0.0,
        )
        self.post_blocks = nn.ModuleList(
            [
                ResidualFFNBlock(
                    hidden_size=self.hidden_size,
                    expansion=self.ffn_expansion,
                    dropout=self.dropout,
                )
                for _ in range(max(0, self.post_ffn_blocks))
            ]
        )
        self.mask_head = self._build_mask_head(in_dim=self.hidden_size)

    def _init_tcn_modules(self):
        self.in_norm = nn.LayerNorm(self.freq_bins)
        self.in_proj = nn.Linear(self.freq_bins, self.bottleneck_size)

        blocks: List[CausalTCNBlock] = []
        for _ in range(max(1, self.tcn_repeats)):
            for b in range(max(1, self.tcn_blocks)):
                dilation = 2 ** b
                blocks.append(
                    CausalTCNBlock(
                        channels=self.bottleneck_size,
                        hidden_channels=self.tcn_hidden_size,
                        kernel_size=self.tcn_kernel_size,
                        dilation=dilation,
                        dropout=self.dropout,
                    )
                )
        self.tcn_stack = nn.ModuleList(blocks)
        self.tcn_out = nn.Sequential(
            nn.PReLU(),
            nn.Conv1d(self.bottleneck_size, self.bottleneck_size, kernel_size=1),
        )
        self.mask_head = self._build_mask_head(in_dim=self.bottleneck_size)

    def _stft(self, mix: torch.Tensor) -> torch.Tensor:
        window = self.analysis_window.to(device=mix.device, dtype=mix.dtype)
        return torch.stft(
            mix,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            center=False,
            return_complex=True,
        )

    def _istft(self, est_spec: torch.Tensor, length: int) -> torch.Tensor:
        # est_spec: [B, S, F, T]
        b, s, f, t = est_spec.shape
        flat = est_spec.reshape(b * s, f, t)
        window = self.analysis_window.to(device=flat.device, dtype=flat.real.dtype)
        wav = torch.istft(
            flat,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            center=False,
            length=int(length),
        )
        return wav.reshape(b, s, -1)

    def _run_gru(self, feat: torch.Tensor, hidden: Optional[torch.Tensor]):
        """
        Run GRU in FP32 when CUDA autocast is enabled.
        This avoids backend errors like:
        _thnn_fused_gru_cell_cuda not implemented for BFloat16.
        """
        if feat.device.type == "cuda" and torch.is_autocast_enabled():
            hidden_fp32 = None if hidden is None else hidden.float()
            with torch.cuda.amp.autocast(enabled=False):
                out_fp32, hidden_out = self.rnn(feat.float(), hidden_fp32)
            return out_fp32.to(feat.dtype), hidden_out
        return self.rnn(feat, hidden)

    def _apply_post_blocks(self, feat: torch.Tensor) -> torch.Tensor:
        for block in self.post_blocks:
            feat = block(feat)
        return feat

    def _estimate_masks_gru(
        self,
        mix_spec: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # mix_spec: [B, F, T]
        mag = torch.log1p(mix_spec.abs()).transpose(1, 2)  # [B, T, F]
        feat = self.in_proj(mag)
        feat, hidden = self._run_gru(feat, hidden)
        feat = self._apply_post_blocks(feat)

        masks = torch.sigmoid(self.mask_head(feat))
        b, t, _ = masks.shape
        masks = masks.view(b, t, self.num_speakers, self.freq_bins).permute(0, 2, 3, 1).contiguous()
        est_spec = masks * mix_spec.unsqueeze(1)
        return est_spec, hidden, masks

    def _estimate_masks_tcn(
        self,
        mix_spec: torch.Tensor,
    ) -> Tuple[torch.Tensor, None, torch.Tensor]:
        # mix_spec: [B, F, T]
        mag = torch.log1p(mix_spec.abs()).transpose(1, 2)  # [B, T, F]
        feat = self.in_norm(mag)
        feat = self.in_proj(feat)  # [B, T, C]

        x = feat.transpose(1, 2)  # [B, C, T]
        skip_sum = None
        for block in self.tcn_stack:
            x, skip = block.forward_sequence(x)
            skip_sum = skip if skip_sum is None else skip_sum + skip

        if skip_sum is None:
            skip_sum = x
        # Use both accumulated skip path and residual state so every block's
        # residual branch (including the last one) participates in the loss graph.
        out = self.tcn_out(skip_sum + x).transpose(1, 2)  # [B, T, C]

        masks = torch.sigmoid(self.mask_head(out))
        b, t, _ = masks.shape
        masks = masks.view(b, t, self.num_speakers, self.freq_bins).permute(0, 2, 3, 1).contiguous()
        est_spec = masks * mix_spec.unsqueeze(1)
        return est_spec, None, masks

    def _estimate_masks(
        self,
        mix_spec: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        if self.architecture == "gru":
            return self._estimate_masks_gru(mix_spec, hidden)
        return self._estimate_masks_tcn(mix_spec)

    def forward(self, mix: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mix: [B, T]
        Returns:
            est_sources: [B, S, T]
        """
        if mix.ndim != 2:
            raise ValueError(f"Expected mix shape [B, T], got {tuple(mix.shape)}")
        mix_spec = self._stft(mix)
        est_spec, _, _ = self._estimate_masks(mix_spec, hidden=None)
        return self._istft(est_spec, length=mix.shape[-1])

    @torch.no_grad()
    def _forward_step_gru(
        self,
        mix_spec_frame: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mag = torch.log1p(mix_spec_frame.abs()).unsqueeze(1)  # [B, 1, F]
        feat = self.in_proj(mag)
        feat, hidden = self._run_gru(feat, hidden)
        feat = self._apply_post_blocks(feat)

        masks = torch.sigmoid(self.mask_head(feat.squeeze(1)))
        b, _ = masks.shape
        masks = masks.view(b, self.num_speakers, self.freq_bins)
        est_spec = masks * mix_spec_frame.unsqueeze(1)
        return est_spec, hidden, masks

    @torch.no_grad()
    def _forward_step_tcn(
        self,
        mix_spec_frame: torch.Tensor,
        hidden: Optional[Dict[str, List[torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, List[torch.Tensor]], torch.Tensor]:
        mag = torch.log1p(mix_spec_frame.abs()).unsqueeze(1)  # [B, 1, F]
        feat = self.in_norm(mag)
        feat = self.in_proj(feat).transpose(1, 2)  # [B, C, 1]

        prev_caches: List[torch.Tensor] = []
        if isinstance(hidden, dict):
            prev_caches = hidden.get("caches", [])

        new_caches: List[torch.Tensor] = []
        x = feat
        skip_sum = None
        for i, block in enumerate(self.tcn_stack):
            cache_i = prev_caches[i] if i < len(prev_caches) else None
            x, skip, new_cache = block.forward_step(x, cache_i)
            new_caches.append(new_cache)
            skip_sum = skip if skip_sum is None else skip_sum + skip

        if skip_sum is None:
            skip_sum = x
        out = self.tcn_out(skip_sum + x).transpose(1, 2).squeeze(1)  # [B, C]

        masks = torch.sigmoid(self.mask_head(out))
        b, _ = masks.shape
        masks = masks.view(b, self.num_speakers, self.freq_bins)
        est_spec = masks * mix_spec_frame.unsqueeze(1)
        return est_spec, {"caches": new_caches}, masks

    @torch.no_grad()
    def forward_step(
        self,
        mix_spec_frame: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """
        One-frame streaming step.

        Args:
            mix_spec_frame: [B, F] complex
            hidden: model state
        Returns:
            est_spec_frame: [B, S, F] complex
            hidden: updated state
            masks: [B, S, F]
        """
        if mix_spec_frame.ndim != 2:
            raise ValueError(f"Expected mix_spec_frame [B, F], got {tuple(mix_spec_frame.shape)}")

        if self.architecture == "gru":
            return self._forward_step_gru(mix_spec_frame, hidden)
        return self._forward_step_tcn(mix_spec_frame, hidden)

    def model_size_million(self) -> float:
        total = sum(p.numel() for p in self.parameters())
        return float(total) / 1e6

    def export_config(self) -> dict:
        return {
            "num_speakers": self.num_speakers,
            "n_fft": self.n_fft,
            "hop_length": self.hop_length,
            "win_length": self.win_length,
            "architecture": self.architecture,
            "dropout": self.dropout,
            "mask_hidden_size": self.mask_hidden_size,
            "mask_head_layers": self.mask_head_layers,
            # GRU fields
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "post_ffn_blocks": self.post_ffn_blocks,
            "ffn_expansion": self.ffn_expansion,
            # TCN fields
            "bottleneck_size": self.bottleneck_size,
            "tcn_hidden_size": self.tcn_hidden_size,
            "tcn_kernel_size": self.tcn_kernel_size,
            "tcn_blocks": self.tcn_blocks,
            "tcn_repeats": self.tcn_repeats,
        }
