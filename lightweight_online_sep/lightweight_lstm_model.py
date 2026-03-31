from typing import Dict, List, Optional, Tuple

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


class LightweightCausalSeparator(nn.Module):
    """
    Streaming-capable separator.

    - architecture='lstm_hybrid': time-frequency hybrid with causal LSTM.
    - architecture='tcn': causal Conv-TasNet style temporal model.
    - architecture='gru': legacy GRU baseline.
    """

    def __init__(
        self,
        num_speakers: int = 2,
        n_fft: int = 256,
        hop_length: int = 128,
        win_length: int = 256,
        architecture: str = "lstm_hybrid",
        dropout: float = 0.0,
        mask_hidden_size: int = 0,
        mask_head_layers: int = 1,
        # GRU branch
        hidden_size: int = 128,
        num_layers: int = 2,
        post_ffn_blocks: int = 1,
        ffn_expansion: int = 2,
        # LSTM-hybrid branch
        time_encoder_dim: int = 128,
        time_kernel_size: int = 0,
        fusion_hidden_size: int = 0,
        lstm_hidden_size: int = 256,
        lstm_num_layers: int = 3,
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

        architecture = str(architecture).lower()
        if architecture == "lstm":
            architecture = "lstm_hybrid"
        if architecture not in {"gru", "tcn", "lstm_hybrid"}:
            raise ValueError(f"Unsupported architecture: {architecture}")

        self.num_speakers = int(num_speakers)
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.win_length = int(win_length)
        self.architecture = architecture

        self.dropout = float(dropout)
        self.mask_hidden_size = int(mask_hidden_size)
        self.mask_head_layers = max(1, int(mask_head_layers))

        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.post_ffn_blocks = int(post_ffn_blocks)
        self.ffn_expansion = int(ffn_expansion)

        self.time_encoder_dim = int(time_encoder_dim)
        self.time_kernel_size = int(time_kernel_size) if int(time_kernel_size) > 0 else self.win_length
        self.fusion_hidden_size = int(fusion_hidden_size) if int(fusion_hidden_size) > 0 else int(hidden_size) * 2
        self.lstm_hidden_size = int(lstm_hidden_size)
        self.lstm_num_layers = int(lstm_num_layers)

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
        elif self.architecture == "tcn":
            self._init_tcn_modules()
        else:
            self._init_lstm_hybrid_modules()

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

    def _build_freq_encoder(self, out_dim: int) -> nn.Module:
        layers: List[nn.Module] = [
            nn.LayerNorm(self.freq_bins),
            nn.Linear(self.freq_bins, out_dim),
            nn.SiLU(),
        ]
        if self.dropout > 0.0:
            layers.append(nn.Dropout(self.dropout))
        return nn.Sequential(*layers)

    def _init_gru_modules(self):
        self.in_proj = self._build_freq_encoder(self.hidden_size)
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

    def _init_lstm_hybrid_modules(self):
        self.freq_encoder = self._build_freq_encoder(self.hidden_size)

        self.time_encoder = nn.Conv1d(
            in_channels=1,
            out_channels=self.time_encoder_dim,
            kernel_size=self.time_kernel_size,
            stride=self.hop_length,
            padding=0,
            bias=True,
        )
        time_proj_layers: List[nn.Module] = [
            nn.LayerNorm(self.time_encoder_dim),
            nn.Linear(self.time_encoder_dim, self.hidden_size),
            nn.SiLU(),
        ]
        if self.dropout > 0.0:
            time_proj_layers.append(nn.Dropout(self.dropout))
        self.time_proj = nn.Sequential(*time_proj_layers)

        fusion_layers: List[nn.Module] = [
            nn.Linear(self.hidden_size * 2, self.fusion_hidden_size),
            nn.SiLU(),
        ]
        if self.dropout > 0.0:
            fusion_layers.append(nn.Dropout(self.dropout))
        self.fusion_proj = nn.Sequential(*fusion_layers)

        self.lstm = nn.LSTM(
            input_size=self.fusion_hidden_size,
            hidden_size=self.lstm_hidden_size,
            num_layers=self.lstm_num_layers,
            batch_first=True,
            dropout=self.dropout if self.lstm_num_layers > 1 else 0.0,
        )

        self.lstm_post_blocks = nn.ModuleList(
            [
                ResidualFFNBlock(
                    hidden_size=self.lstm_hidden_size,
                    expansion=self.ffn_expansion,
                    dropout=self.dropout,
                )
                for _ in range(max(0, self.post_ffn_blocks))
            ]
        )

        self.mask_head = self._build_mask_head(in_dim=self.lstm_hidden_size)

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
        if feat.device.type == "cuda" and torch.is_autocast_enabled():
            hidden_fp32 = None if hidden is None else hidden.float()
            with torch.cuda.amp.autocast(enabled=False):
                out_fp32, hidden_out = self.rnn(feat.float(), hidden_fp32)
            return out_fp32.to(feat.dtype), hidden_out
        return self.rnn(feat, hidden)

    def _run_lstm(
        self,
        feat: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if feat.device.type == "cuda" and torch.is_autocast_enabled():
            state_fp32 = None
            if state is not None:
                state_fp32 = (state[0].float(), state[1].float())
            with torch.cuda.amp.autocast(enabled=False):
                out_fp32, state_out = self.lstm(feat.float(), state_fp32)
            return out_fp32.to(feat.dtype), state_out
        return self.lstm(feat, state)

    def _apply_blocks(self, feat: torch.Tensor, blocks: nn.ModuleList) -> torch.Tensor:
        for block in blocks:
            feat = block(feat)
        return feat

    def _estimate_masks_gru(
        self,
        mix_spec: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mag = torch.log1p(mix_spec.abs()).transpose(1, 2)
        feat = self.in_proj(mag)
        feat, hidden = self._run_gru(feat, hidden)
        feat = self._apply_blocks(feat, self.post_blocks)

        masks = torch.sigmoid(self.mask_head(feat))
        b, t, _ = masks.shape
        masks = masks.view(b, t, self.num_speakers, self.freq_bins).permute(0, 2, 3, 1).contiguous()
        est_spec = masks * mix_spec.unsqueeze(1)
        return est_spec, hidden, masks

    def _estimate_masks_tcn(
        self,
        mix_spec: torch.Tensor,
    ) -> Tuple[torch.Tensor, None, torch.Tensor]:
        mag = torch.log1p(mix_spec.abs()).transpose(1, 2)
        feat = self.in_norm(mag)
        feat = self.in_proj(feat)

        x = feat.transpose(1, 2)
        skip_sum = None
        for block in self.tcn_stack:
            x, skip = block.forward_sequence(x)
            skip_sum = skip if skip_sum is None else skip_sum + skip

        if skip_sum is None:
            skip_sum = x
        out = self.tcn_out(skip_sum + x).transpose(1, 2)

        masks = torch.sigmoid(self.mask_head(out))
        b, t, _ = masks.shape
        masks = masks.view(b, t, self.num_speakers, self.freq_bins).permute(0, 2, 3, 1).contiguous()
        est_spec = masks * mix_spec.unsqueeze(1)
        return est_spec, None, masks

    def _estimate_masks_lstm_hybrid(
        self,
        mix_spec: torch.Tensor,
        mix_wave: torch.Tensor,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        mag = torch.log1p(mix_spec.abs()).transpose(1, 2)
        freq_feat = self.freq_encoder(mag)

        time_raw = self.time_encoder(mix_wave.unsqueeze(1)).transpose(1, 2)

        frames = min(freq_feat.shape[1], time_raw.shape[1], mix_spec.shape[-1])
        freq_feat = freq_feat[:, :frames, :]
        time_raw = time_raw[:, :frames, :]
        mix_spec = mix_spec[..., :frames]

        time_feat = self.time_proj(time_raw)
        fused = torch.cat([freq_feat, time_feat], dim=-1)
        fused = self.fusion_proj(fused)

        lstm_out, state = self._run_lstm(fused, None)
        lstm_out = self._apply_blocks(lstm_out, self.lstm_post_blocks)

        masks = torch.sigmoid(self.mask_head(lstm_out))
        b, t, _ = masks.shape
        masks = masks.view(b, t, self.num_speakers, self.freq_bins).permute(0, 2, 3, 1).contiguous()
        est_spec = masks * mix_spec.unsqueeze(1)
        return est_spec, state, masks

    def _estimate_masks(
        self,
        mix_spec: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
        mix_wave: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        if self.architecture == "gru":
            return self._estimate_masks_gru(mix_spec, hidden)
        if self.architecture == "tcn":
            return self._estimate_masks_tcn(mix_spec)
        if mix_wave is None:
            raise ValueError("mix_wave is required for lstm_hybrid architecture")
        return self._estimate_masks_lstm_hybrid(mix_spec, mix_wave)

    def forward(self, mix: torch.Tensor) -> torch.Tensor:
        if mix.ndim != 2:
            raise ValueError(f"Expected mix shape [B, T], got {tuple(mix.shape)}")
        mix_spec = self._stft(mix)
        est_spec, _, _ = self._estimate_masks(mix_spec, hidden=None, mix_wave=mix)
        return self._istft(est_spec, length=mix.shape[-1])

    @torch.no_grad()
    def _forward_step_gru(
        self,
        mix_spec_frame: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mag = torch.log1p(mix_spec_frame.abs()).unsqueeze(1)
        feat = self.in_proj(mag)
        feat, hidden = self._run_gru(feat, hidden)
        feat = self._apply_blocks(feat, self.post_blocks)

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
        mag = torch.log1p(mix_spec_frame.abs()).unsqueeze(1)
        feat = self.in_norm(mag)
        feat = self.in_proj(feat).transpose(1, 2)

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
        out = self.tcn_out(skip_sum + x).transpose(1, 2).squeeze(1)

        masks = torch.sigmoid(self.mask_head(out))
        b, _ = masks.shape
        masks = masks.view(b, self.num_speakers, self.freq_bins)
        est_spec = masks * mix_spec_frame.unsqueeze(1)
        return est_spec, {"caches": new_caches}, masks

    @torch.no_grad()
    def _forward_step_lstm_hybrid(
        self,
        mix_spec_frame: torch.Tensor,
        hidden: Optional[Dict[str, Tuple[torch.Tensor, torch.Tensor]]] = None,
        frame_time: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
        mag = torch.log1p(mix_spec_frame.abs()).unsqueeze(1)
        freq_feat = self.freq_encoder(mag)

        if frame_time is None:
            frame_time = torch.fft.irfft(mix_spec_frame, n=self.n_fft)
            frame_time = frame_time[..., : self.time_kernel_size]
        frame_time = torch.as_tensor(frame_time, device=mix_spec_frame.device, dtype=mix_spec_frame.real.dtype)
        if frame_time.ndim == 1:
            frame_time = frame_time.unsqueeze(0)

        target_len = self.time_kernel_size
        if frame_time.shape[-1] < target_len:
            frame_time = F.pad(frame_time, (target_len - frame_time.shape[-1], 0))
        elif frame_time.shape[-1] > target_len:
            frame_time = frame_time[..., -target_len:]

        time_raw = F.conv1d(
            frame_time.unsqueeze(1),
            self.time_encoder.weight,
            self.time_encoder.bias,
            stride=1,
            padding=0,
        )
        time_raw = time_raw.transpose(1, 2)
        time_feat = self.time_proj(time_raw)

        fused = torch.cat([freq_feat, time_feat], dim=-1)
        fused = self.fusion_proj(fused)

        prev_state = None
        if isinstance(hidden, dict):
            prev_state = hidden.get("lstm", None)

        lstm_out, lstm_state = self._run_lstm(fused, prev_state)
        lstm_out = self._apply_blocks(lstm_out, self.lstm_post_blocks)

        masks = torch.sigmoid(self.mask_head(lstm_out.squeeze(1)))
        b, _ = masks.shape
        masks = masks.view(b, self.num_speakers, self.freq_bins)
        est_spec = masks * mix_spec_frame.unsqueeze(1)
        return est_spec, {"lstm": lstm_state}, masks

    @torch.no_grad()
    def forward_step(
        self,
        mix_spec_frame: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
        frame_time: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        if mix_spec_frame.ndim != 2:
            raise ValueError(f"Expected mix_spec_frame [B, F], got {tuple(mix_spec_frame.shape)}")

        if self.architecture == "gru":
            return self._forward_step_gru(mix_spec_frame, hidden)
        if self.architecture == "tcn":
            return self._forward_step_tcn(mix_spec_frame, hidden)
        return self._forward_step_lstm_hybrid(mix_spec_frame, hidden, frame_time=frame_time)

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
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "post_ffn_blocks": self.post_ffn_blocks,
            "ffn_expansion": self.ffn_expansion,
            "time_encoder_dim": self.time_encoder_dim,
            "time_kernel_size": self.time_kernel_size,
            "fusion_hidden_size": self.fusion_hidden_size,
            "lstm_hidden_size": self.lstm_hidden_size,
            "lstm_num_layers": self.lstm_num_layers,
            "bottleneck_size": self.bottleneck_size,
            "tcn_hidden_size": self.tcn_hidden_size,
            "tcn_kernel_size": self.tcn_kernel_size,
            "tcn_blocks": self.tcn_blocks,
            "tcn_repeats": self.tcn_repeats,
        }
