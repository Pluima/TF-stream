import os
import sys
import warnings
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import separator_kwargs_from_config, validate_model_hparams

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_NBSS_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", "..", "..", "NBSS"))
_NBSS_IMPORT_ERROR = None
if os.path.isdir(_NBSS_ROOT) and _NBSS_ROOT not in sys.path:
    sys.path.insert(0, _NBSS_ROOT)
try:
    import models.arch.OnlineSpatialNet as _osn_module
    if getattr(_osn_module, "Mamba", None) is None:
        class _DummyMamba(nn.Module):
            pass

        _osn_module.Mamba = _DummyMamba
    if hasattr(_osn_module, "CausalConv1d"):
        def _patched_causalconv1d_forward(self, x, state=None):
            # NBSS upstream uses self.kernel_size as int in cached path, while nn.Conv1d stores tuple.
            k = int(self.kernel_size[0]) if isinstance(self.kernel_size, tuple) else int(self.kernel_size)
            if state is None or id(self) not in state:
                x = F.pad(x, pad=(k - 1 - self.look_ahead, self.look_ahead))
            else:
                x = torch.concat([state[id(self)], x], dim=-1)
            if state is not None:
                tail = k - 1
                state[id(self)] = x[..., -tail:] if tail > 0 else x[..., :0]
            return nn.Conv1d.forward(self, x)

        _osn_module.CausalConv1d.forward = _patched_causalconv1d_forward
    from models.arch.OnlineSpatialNet import OnlineSpatialNet as NBSSOnlineSpatialNet
except Exception as _exc:
    NBSSOnlineSpatialNet = None
    _NBSS_IMPORT_ERROR = _exc


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
    - architecture='stereo_lite': lightweight stereo-aware narrow-band GRU.
    - architecture='stereo_stacked_lstm': pure stacked-LSTM stereo branch (no CNN blocks).
      Supports optional azimuth conditioning so output source order follows
      provided directional cues.
    - architecture='stereo_beam_lite': shared-trunk stereo LSTM with cue-conditioned
      multichannel complex decoder and TF-level cross-query competition.
    - Supports mono input [B, T] and stereo input [B, 2, T] / [B, T, 2].
      Stereo is fused in the front-end head to inject spatial cues while keeping
      the downstream separator trunk unchanged.
    """

    def __init__(
        self,
        num_speakers: int = 2,
        n_fft: int = 256,
        hop_length: int = 128,
        win_length: int = 256,
        architecture: str = "lstm_hybrid",
        output_source_channels: int = 1,
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
        # Stereo front-end
        stereo_frontend_dim: int = 48,
        stereo_frontend_layers: int = 2,
        stereo_band_kernel_size: int = 7,
        use_learned_stereo_fusion: bool = True,
        use_azimuth_conditioning: bool = True,
        default_left_azimuth_deg: float = -15.0,
        default_right_azimuth_deg: float = 15.0,
        # Online Spatial Net branch
        osn_num_layers: int = 8,
        osn_dim_hidden: int = 96,
        osn_dim_ffn: int = 192,
        osn_dim_squeeze: int = 8,
        osn_num_heads: int = 4,
        osn_encoder_kernel_size: int = 5,
        osn_freq_kernel_size: int = 5,
        osn_time_kernel_size: int = 3,
        osn_freq_conv_groups: int = 8,
        osn_time_conv_groups: int = 8,
        osn_attention_scope: int = 251,
        osn_attention: str = "ret(2,share_qk)",
        osn_streaming_history: int = 320,
        force_rnn_fp32: bool = True,
    ):
        super().__init__()
        if num_speakers < 2:
            raise ValueError("num_speakers must be >= 2")
        if hop_length > win_length:
            raise ValueError("hop_length must be <= win_length")

        architecture = str(architecture).lower()
        if architecture == "lstm":
            architecture = "lstm_hybrid"
        if architecture not in {
            "gru",
            "tcn",
            "lstm_hybrid",
            "online_spatialnet",
            "stereo_lite",
            "stereo_stacked_lstm",
            "stereo_beam_lite",
        }:
            raise ValueError(f"Unsupported architecture: {architecture}")

        self.num_speakers = int(num_speakers)
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.win_length = int(win_length)
        self.architecture = architecture
        requested_output_channels = int(output_source_channels)
        if requested_output_channels < 1:
            raise ValueError("output_source_channels must be >= 1")
        if self.architecture == "stereo_beam_lite":
            if requested_output_channels not in {1, 2}:
                raise ValueError("output_source_channels must be 1 or 2 for stereo_beam_lite")
            self.output_source_channels = requested_output_channels
        else:
            if requested_output_channels != 1:
                raise ValueError(f"output_source_channels must be 1 for {self.architecture}")
            self.output_source_channels = 1
        self.azimuth_prompt_mode = "left_right" if self.architecture == "stereo_beam_lite" else "degrees"

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
        self.stereo_frontend_dim = int(stereo_frontend_dim)
        self.stereo_frontend_layers = int(stereo_frontend_layers)
        self.stereo_band_kernel_size = int(stereo_band_kernel_size)
        self.use_learned_stereo_fusion = bool(use_learned_stereo_fusion)
        self.use_azimuth_conditioning = bool(use_azimuth_conditioning)
        self.default_left_azimuth_deg = float(default_left_azimuth_deg)
        self.default_right_azimuth_deg = float(default_right_azimuth_deg)
        self.osn_num_layers = int(osn_num_layers)
        self.osn_dim_hidden = int(osn_dim_hidden)
        self.osn_dim_ffn = int(osn_dim_ffn)
        self.osn_dim_squeeze = int(osn_dim_squeeze)
        self.osn_num_heads = int(osn_num_heads)
        self.osn_encoder_kernel_size = int(osn_encoder_kernel_size)
        self.osn_freq_kernel_size = int(osn_freq_kernel_size)
        self.osn_time_kernel_size = int(osn_time_kernel_size)
        self.osn_freq_conv_groups = int(osn_freq_conv_groups)
        self.osn_time_conv_groups = int(osn_time_conv_groups)
        self.osn_attention_scope = int(osn_attention_scope)
        self.osn_attention = str(osn_attention)
        self.osn_streaming_history = int(osn_streaming_history)
        self.force_rnn_fp32 = bool(force_rnn_fp32)
        self._runtime_disable_bf16_rnn = False
        self._warned_bf16_rnn_fallback = False
        self._aux_outputs: Dict[str, torch.Tensor] = {}
        self._eps = 1e-8
        self.stereo_feature_dim = 7  # [mono, left, right, ild, cos_ipd, sin_ipd, coherence]

        self.freq_bins = self.n_fft // 2 + 1

        # Use Hamming window (non-zero boundary) to satisfy center=False iSTFT overlap-add checks.
        self.register_buffer("analysis_window", torch.hamming_window(self.win_length), persistent=False)
        if self.architecture == "stereo_stacked_lstm":
            # Keep this branch strictly CNN-free.
            self.use_learned_stereo_fusion = False
            self._init_stereo_frontend_no_conv()
        else:
            self._init_stereo_frontend()
        if self.architecture in {"online_spatialnet", "stereo_lite", "stereo_stacked_lstm", "stereo_beam_lite"}:
            # These branches consume dedicated stereo complex features directly.
            # Freeze fusion front-end params to avoid DDP unused-parameter reduction errors.
            for p in self.stereo_frontend.parameters():
                p.requires_grad_(False)
            for p in self.stereo_frontend_out.parameters():
                p.requires_grad_(False)
            self.stereo_delta_scale.requires_grad_(False)
            self.stereo_gain_scale.requires_grad_(False)

        if self.architecture == "gru":
            self._init_gru_modules()
        elif self.architecture == "tcn":
            self._init_tcn_modules()
        elif self.architecture == "stereo_lite":
            self._init_stereo_lite_modules()
        elif self.architecture == "stereo_stacked_lstm":
            self._init_stereo_stacked_lstm_modules()
        elif self.architecture == "stereo_beam_lite":
            self._init_stereo_beam_lite_modules()
        elif self.architecture == "online_spatialnet":
            self._init_online_spatialnet_modules()
        else:
            self._init_lstm_hybrid_modules()

    def _init_stereo_frontend(self):
        layers: List[nn.Module] = []
        hidden_dim = max(8, int(self.stereo_frontend_dim))
        n_layers = max(1, int(self.stereo_frontend_layers))
        kernel_size = max(3, int(self.stereo_band_kernel_size))
        if kernel_size % 2 == 0:
            kernel_size += 1
        self.stereo_band_kernel_size = int(kernel_size)

        in_dim = self.stereo_feature_dim
        for _ in range(n_layers):
            layers.append(
                nn.Conv1d(
                    in_channels=in_dim,
                    out_channels=hidden_dim,
                    kernel_size=self.stereo_band_kernel_size,
                    padding=self.stereo_band_kernel_size // 2,
                    bias=True,
                )
            )
            layers.append(nn.GroupNorm(1, hidden_dim))
            layers.append(nn.SiLU())
            if self.dropout > 0.0:
                layers.append(nn.Dropout(self.dropout))
            in_dim = hidden_dim
        self.stereo_frontend = nn.Sequential(*layers)
        self.stereo_frontend_out = nn.Conv1d(hidden_dim, 2, kernel_size=1, bias=True)
        # Keep initial behavior close to mono baseline for stable early training.
        nn.init.zeros_(self.stereo_frontend_out.weight)
        nn.init.zeros_(self.stereo_frontend_out.bias)
        self.stereo_delta_scale = nn.Parameter(torch.tensor(0.10, dtype=torch.float32))
        self.stereo_gain_scale = nn.Parameter(torch.tensor(0.10, dtype=torch.float32))

    def _init_stereo_frontend_no_conv(self):
        # For CNN-free branches, keep placeholders so exported config code remains compatible.
        self.stereo_frontend = nn.Identity()
        self.stereo_frontend_out = nn.Identity()
        self.stereo_delta_scale = nn.Parameter(torch.tensor(0.0, dtype=torch.float32), requires_grad=False)
        self.stereo_gain_scale = nn.Parameter(torch.tensor(0.0, dtype=torch.float32), requires_grad=False)

    def _run_stereo_frontend(self, feats: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            feats: [B, T, F, C]
        Returns:
            delta_logits: [B, T, F]
            gain_logits: [B, T, F]
        """
        b, t, f, c = feats.shape
        x = feats.permute(0, 1, 3, 2).reshape(b * t, c, f).contiguous()  # [B*T, C, F]
        h = self.stereo_frontend(x)  # [B*T, H, F]
        out = self.stereo_frontend_out(h)  # [B*T, 2, F]
        out = out.reshape(b, t, 2, f).permute(0, 1, 3, 2).contiguous()  # [B, T, F, 2]
        return out[..., 0], out[..., 1]

    def _run_stereo_frontend_frame(self, feats: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            feats: [B, F, C]
        Returns:
            delta_logits: [B, F]
            gain_logits: [B, F]
        """
        x = feats.permute(0, 2, 1).contiguous()  # [B, C, F]
        h = self.stereo_frontend(x)  # [B, H, F]
        out = self.stereo_frontend_out(h)  # [B, 2, F]
        return out[:, 0, :], out[:, 1, :]

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

    def _init_stereo_lite_modules(self):
        """
        Lightweight stereo-aware trunk:
        - frame-wise stereo complex features [Re/Im(left/right)]
        - depthwise separable frequency mixing
        - narrow-band causal GRU over time for each frequency bin
        """
        in_dim = 4  # real/imag for stereo channels
        k = max(3, int(self.stereo_band_kernel_size))
        if k % 2 == 0:
            k += 1

        self.slite_in_norm = nn.LayerNorm(in_dim)
        self.slite_in_proj = nn.Linear(in_dim, self.hidden_size)
        self.slite_in_act = nn.SiLU()
        self.slite_dropout = nn.Dropout(self.dropout) if self.dropout > 0.0 else nn.Identity()

        self.slite_freq_dw = nn.Conv1d(
            in_channels=self.hidden_size,
            out_channels=self.hidden_size,
            kernel_size=k,
            padding=k // 2,
            groups=self.hidden_size,
            bias=True,
        )
        self.slite_freq_pw = nn.Conv1d(
            in_channels=self.hidden_size,
            out_channels=self.hidden_size,
            kernel_size=1,
            bias=True,
        )
        self.slite_freq_act = nn.SiLU()
        self.slite_freq_norm = nn.LayerNorm(self.hidden_size)

        self.slite_gru = nn.GRU(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0.0,
        )
        self.slite_out_norm = nn.LayerNorm(self.hidden_size)
        if self.use_azimuth_conditioning:
            self.azimuth_proj = nn.Sequential(
                nn.Linear(2, self.hidden_size),
                nn.SiLU(),
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.SiLU(),
            )
            # Performance-oriented directional path:
            # query-conditioned recurrent trunk + FiLM + query decoder.
            self.slite_query_film = nn.Linear(self.hidden_size, self.hidden_size * 2)
            self.slite_mask_out_query = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.SiLU(),
                nn.Linear(self.hidden_size, 2),
            )
        # Baseline complex ratio mask head (kept for compatibility / fallback).
        self.slite_mask_out = nn.Linear(self.hidden_size, self.num_speakers * 2)

    def _init_stereo_stacked_lstm_modules(self):
        """
        Pure stacked-LSTM stereo trunk (no CNN blocks):
        - frame-wise stereo complex features [Re/Im(left/right)]
        - stacked causal LSTM over time for each frequency bin
        """
        in_dim = 4  # real/imag for stereo channels
        self.slstm_in_norm = nn.LayerNorm(in_dim)
        self.slstm_in_proj = nn.Linear(in_dim, self.hidden_size)
        self.slstm_in_act = nn.SiLU()
        self.slstm_dropout = nn.Dropout(self.dropout) if self.dropout > 0.0 else nn.Identity()

        self.slstm_lstm = nn.LSTM(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0.0,
        )
        self.slstm_out_norm = nn.LayerNorm(self.hidden_size)

        if self.use_azimuth_conditioning:
            self.azimuth_proj = nn.Sequential(
                nn.Linear(2, self.hidden_size),
                nn.SiLU(),
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.SiLU(),
            )
            self.slstm_query_film = nn.Linear(self.hidden_size, self.hidden_size * 2)
            self.slstm_mask_out_query = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.SiLU(),
                nn.Linear(self.hidden_size, 2),
            )
        self.slstm_mask_out = nn.Linear(self.hidden_size, self.num_speakers * 2)

    def _init_online_spatialnet_modules(self):
        if NBSSOnlineSpatialNet is None:
            detail = "" if _NBSS_IMPORT_ERROR is None else f" | import error: {type(_NBSS_IMPORT_ERROR).__name__}: {_NBSS_IMPORT_ERROR}"
            raise RuntimeError(
                "OnlineSpatialNet dependency is unavailable. "
                f"Expected NBSS package under: {_NBSS_ROOT}{detail}"
            )

        attn_scope = max(1, int(self.osn_attention_scope))
        attention = str(self.osn_attention).strip().lower()
        if attention in {"", "mhsa"}:
            attention = f"mhsa({attn_scope})"
        self.osn = NBSSOnlineSpatialNet(
            dim_input=4,  # stereo complex features -> 2 channels * (real, imag)
            dim_output=self.num_speakers * 2,
            num_layers=max(1, int(self.osn_num_layers)),
            dim_squeeze=max(1, int(self.osn_dim_squeeze)),
            num_freqs=self.freq_bins,
            encoder_kernel_size=max(1, int(self.osn_encoder_kernel_size)),
            dim_hidden=max(8, int(self.osn_dim_hidden)),
            dim_ffn=max(8, int(self.osn_dim_ffn)),
            num_heads=max(1, int(self.osn_num_heads)),
            dropout=(self.dropout, self.dropout, self.dropout),
            kernel_size=(max(1, int(self.osn_freq_kernel_size)), max(1, int(self.osn_time_kernel_size))),
            conv_groups=(max(1, int(self.osn_freq_conv_groups)), max(1, int(self.osn_time_conv_groups))),
            norms=["LN", "LN", "GN", "LN", "LN", "LN"],
            padding="zeros",
            full_share=0,
            attention=attention,
            rope=False,
        )

    def _normalize_osn_stereo_spec(self, mix_stereo_spec: Optional[torch.Tensor], mix_spec: torch.Tensor) -> torch.Tensor:
        if mix_stereo_spec is None:
            # Fallback for mono input: duplicate mono reference.
            return torch.stack([mix_spec, mix_spec], dim=1)

        if mix_stereo_spec.ndim != 4:
            raise ValueError(f"Expected mix_stereo_spec [B,2,F,T], got {tuple(mix_stereo_spec.shape)}")
        if mix_stereo_spec.shape[1] == 2 and mix_stereo_spec.shape[2] == self.freq_bins:
            return mix_stereo_spec
        if mix_stereo_spec.shape[2] == 2 and mix_stereo_spec.shape[1] == self.freq_bins:
            return mix_stereo_spec.permute(0, 2, 1, 3).contiguous()
        raise ValueError(f"Cannot infer stereo axis for shape {tuple(mix_stereo_spec.shape)}")

    def _prepare_osn_features(
        self,
        mix_spec: torch.Tensor,
        mix_stereo_spec: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            mix_spec: [B, F, T] complex mono spec
            mix_stereo_spec: [B, 2, F, T] complex or None
        Returns:
            feats: [B, F, T, 4]
            ref_mean_mag: [B, F, 1]
        """
        stereo_spec = self._normalize_osn_stereo_spec(mix_stereo_spec, mix_spec)
        stereo_bftc = stereo_spec.permute(0, 2, 3, 1).contiguous()  # [B,F,T,2]
        ref = stereo_bftc[..., 0]  # [B,F,T]
        ref_mean_mag = torch.clamp(ref.abs().mean(dim=2, keepdim=True), min=self._eps)
        stereo_norm = stereo_bftc / ref_mean_mag.unsqueeze(-1)
        feats = torch.view_as_real(stereo_norm).reshape(stereo_norm.shape[0], self.freq_bins, stereo_norm.shape[2], -1)
        return feats, ref_mean_mag

    def _forward_online_spatialnet(self, mix_spec: torch.Tensor, mix_stereo_spec: Optional[torch.Tensor]) -> torch.Tensor:
        feats, ref_mean_mag = self._prepare_osn_features(mix_spec, mix_stereo_spec)  # [B,F,T,4], [B,F,1]
        out = self.osn(feats, inference=False)  # [B,F,T,S*2]
        b, f, t, _ = out.shape
        out_c = torch.view_as_complex(out.float().reshape(b, f, t, self.num_speakers, 2))  # [B,F,T,S]
        out_c = out_c * ref_mean_mag.unsqueeze(-1)
        return out_c.permute(0, 3, 1, 2).contiguous()  # [B,S,F,T]

    def _run_slite_gru(
        self,
        feat: torch.Tensor,
        hidden: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        use_fp32 = (
            feat.device.type == "cuda"
            and torch.is_autocast_enabled()
            and (self.force_rnn_fp32 or self._runtime_disable_bf16_rnn)
        )
        if use_fp32:
            hidden_fp32 = None if hidden is None else hidden.float()
            with torch.cuda.amp.autocast(enabled=False):
                out_fp32, hidden_out = self.slite_gru(feat.float(), hidden_fp32)
            return out_fp32.to(feat.dtype), hidden_out
        try:
            return self.slite_gru(feat, hidden)
        except RuntimeError as exc:
            msg = str(exc).lower()
            unsupported_bf16 = (
                feat.device.type == "cuda"
                and feat.dtype == torch.bfloat16
                and ("bfloat16" in msg)
                and ("not implemented" in msg or "cudnn_status_not_supported" in msg)
            )
            if not unsupported_bf16:
                raise
            self._runtime_disable_bf16_rnn = True
            if not self._warned_bf16_rnn_fallback:
                warnings.warn(
                    "BF16 recurrent kernels are unavailable in current torch/cuda env. "
                    "Falling back to FP32 for GRU/LSTM blocks.",
                    RuntimeWarning,
                )
                self._warned_bf16_rnn_fallback = True
            hidden_fp32 = None if hidden is None else hidden.float()
            with torch.cuda.amp.autocast(enabled=False):
                out_fp32, hidden_out = self.slite_gru(feat.float(), hidden_fp32)
            return out_fp32.to(feat.dtype), hidden_out

    def _run_slstm(
        self,
        feat: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        use_fp32 = (
            feat.device.type == "cuda"
            and torch.is_autocast_enabled()
            and (self.force_rnn_fp32 or self._runtime_disable_bf16_rnn)
        )
        if use_fp32:
            state_fp32 = None
            if state is not None:
                state_fp32 = (state[0].float(), state[1].float())
            with torch.cuda.amp.autocast(enabled=False):
                out_fp32, state_out = self.slstm_lstm(feat.float(), state_fp32)
            return out_fp32.to(feat.dtype), state_out
        try:
            return self.slstm_lstm(feat, state)
        except RuntimeError as exc:
            msg = str(exc).lower()
            unsupported_bf16 = (
                feat.device.type == "cuda"
                and feat.dtype == torch.bfloat16
                and ("bfloat16" in msg)
                and ("not implemented" in msg or "cudnn_status_not_supported" in msg)
            )
            if not unsupported_bf16:
                raise
            self._runtime_disable_bf16_rnn = True
            if not self._warned_bf16_rnn_fallback:
                warnings.warn(
                    "BF16 recurrent kernels are unavailable in current torch/cuda env. "
                    "Falling back to FP32 for GRU/LSTM blocks.",
                    RuntimeWarning,
                )
                self._warned_bf16_rnn_fallback = True
            state_fp32 = None
            if state is not None:
                state_fp32 = (state[0].float(), state[1].float())
            with torch.cuda.amp.autocast(enabled=False):
                out_fp32, state_out = self.slstm_lstm(feat.float(), state_fp32)
            return out_fp32.to(feat.dtype), state_out

    def _prepare_stereo_lite_features(
        self,
        mix_spec: torch.Tensor,
        mix_stereo_spec: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Args:
            mix_spec: [B, F, T] complex mono reference.
            mix_stereo_spec: [B, 2, F, T] complex or None.
        Returns:
            [B, F, T, H]
        """
        stereo_spec = self._normalize_osn_stereo_spec(mix_stereo_spec, mix_spec)
        left = stereo_spec[:, 0]
        right = stereo_spec[:, 1]

        ref_mag = 0.5 * (left.abs() + right.abs())  # [B,F,T]
        ref_scale = torch.clamp(ref_mag.mean(dim=-1, keepdim=True), min=self._eps)  # [B,F,1]
        left_n = left / ref_scale
        right_n = right / ref_scale

        feat = torch.stack([left_n.real, left_n.imag, right_n.real, right_n.imag], dim=-1)  # [B,F,T,4]
        feat = self.slite_in_norm(feat)
        feat = self.slite_in_proj(feat)
        feat = self.slite_in_act(feat)
        return self.slite_dropout(feat)

    def _prepare_stereo_stacked_lstm_features(
        self,
        mix_spec: torch.Tensor,
        mix_stereo_spec: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Args:
            mix_spec: [B, F, T] complex mono reference.
            mix_stereo_spec: [B, 2, F, T] complex or None.
        Returns:
            [B, F, T, H]
        """
        stereo_spec = self._normalize_osn_stereo_spec(mix_stereo_spec, mix_spec)
        left = stereo_spec[:, 0]
        right = stereo_spec[:, 1]

        ref_mag = 0.5 * (left.abs() + right.abs())  # [B,F,T]
        ref_scale = torch.clamp(ref_mag.mean(dim=-1, keepdim=True), min=self._eps)  # [B,F,1]
        left_n = left / ref_scale
        right_n = right / ref_scale

        feat = torch.stack([left_n.real, left_n.imag, right_n.real, right_n.imag], dim=-1)  # [B,F,T,4]
        feat = self.slstm_in_norm(feat)
        feat = self.slstm_in_proj(feat)
        feat = self.slstm_in_act(feat)
        return self.slstm_dropout(feat)

    def _slite_frequency_mix_sequence(self, feat: torch.Tensor) -> torch.Tensor:
        b, f, t, h = feat.shape
        x = feat.permute(0, 2, 3, 1).reshape(b * t, h, f).contiguous()  # [B*T,H,F]
        x = self.slite_freq_dw(x)
        x = self.slite_freq_act(x)
        x = self.slite_freq_pw(x)
        x = x.reshape(b, t, h, f).permute(0, 3, 1, 2).contiguous()  # [B,F,T,H]
        return self.slite_freq_norm(x + feat)

    def _slite_frequency_mix_frame(self, feat: torch.Tensor) -> torch.Tensor:
        # feat: [B,F,H]
        x = feat.transpose(1, 2).contiguous()  # [B,H,F]
        x = self.slite_freq_dw(x)
        x = self.slite_freq_act(x)
        x = self.slite_freq_pw(x)
        x = x.transpose(1, 2).contiguous()  # [B,F,H]
        return self.slite_freq_norm(x + feat)

    def _default_azimuth_deg(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self.num_speakers == 2:
            base = torch.tensor(
                [self.default_left_azimuth_deg, self.default_right_azimuth_deg],
                device=device,
                dtype=dtype,
            )
        else:
            base = torch.linspace(-45.0, 45.0, steps=self.num_speakers, device=device, dtype=dtype)
        return base.unsqueeze(0).expand(batch_size, -1).contiguous()

    def _azimuth_deg_to_left_right_prompt(self, azimuth_deg: torch.Tensor) -> torch.Tensor:
        left = (azimuth_deg < 0.0).to(dtype=azimuth_deg.dtype)
        right = 1.0 - left
        return torch.stack([left, right], dim=-1)

    def _default_azimuth_prompt(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if self.azimuth_prompt_mode == "left_right":
            if self.num_speakers == 2:
                base = torch.tensor(
                    [[1.0, 0.0], [0.0, 1.0]],
                    device=device,
                    dtype=dtype,
                )
            else:
                base_deg = self._default_azimuth_deg(1, device=device, dtype=dtype).squeeze(0)
                base = self._azimuth_deg_to_left_right_prompt(base_deg)
            return base.unsqueeze(0).expand(batch_size, -1, -1).contiguous()

        az_deg = self._default_azimuth_deg(batch_size, device=device, dtype=dtype)
        az_rad = az_deg * (torch.pi / 180.0)
        return torch.stack([torch.sin(az_rad), torch.cos(az_rad)], dim=-1)

    def _resolve_azimuth_sincos(
        self,
        azimuth_deg: Optional[torch.Tensor],
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Returns:
            az_prompt: [B, S, 2].
            For stereo_beam_lite, this is a left/right prompt [is_left, is_right].
            Otherwise this is [sin(az), cos(az)].
        """
        if azimuth_deg is None:
            return self._default_azimuth_prompt(batch_size, device=device, dtype=dtype)

        az = torch.as_tensor(azimuth_deg, device=device, dtype=dtype)
        if az.ndim == 1:
            if az.shape[0] != self.num_speakers:
                raise ValueError(
                    f"azimuth_deg shape mismatch, expected [{self.num_speakers}] or [B,{self.num_speakers}], got {tuple(az.shape)}"
                )
            az = az.unsqueeze(0).expand(batch_size, -1)
            if self.azimuth_prompt_mode == "left_right":
                return self._azimuth_deg_to_left_right_prompt(az).contiguous()
            az_rad = az * (torch.pi / 180.0)
            return torch.stack([torch.sin(az_rad), torch.cos(az_rad)], dim=-1)

        if az.ndim == 2:
            if az.shape[0] == batch_size and az.shape[1] == self.num_speakers:
                if self.azimuth_prompt_mode == "left_right":
                    return self._azimuth_deg_to_left_right_prompt(az).contiguous()
                az_rad = az * (torch.pi / 180.0)
                return torch.stack([torch.sin(az_rad), torch.cos(az_rad)], dim=-1)
            if az.shape == (self.num_speakers, 2):
                return az.unsqueeze(0).expand(batch_size, -1, -1).contiguous()
            raise ValueError(
                f"azimuth_deg shape mismatch, expected [B,{self.num_speakers}] or [{self.num_speakers},2], got {tuple(az.shape)}"
            )

        if az.ndim == 3:
            if az.shape[0] == batch_size and az.shape[1] == self.num_speakers and az.shape[2] == 2:
                return az
            raise ValueError(
                f"azimuth_deg shape mismatch, expected [B,{self.num_speakers},2], got {tuple(az.shape)}"
            )

        raise ValueError(f"Unsupported azimuth_deg shape: {tuple(az.shape)}")

    def _azimuth_embedding(
        self,
        azimuth_deg: Optional[torch.Tensor],
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        if not self.use_azimuth_conditioning:
            return None
        az_sincos = self._resolve_azimuth_sincos(
            azimuth_deg=azimuth_deg,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        )  # [B,S,2]
        return self.azimuth_proj(az_sincos)  # [B,S,H]

    def _forward_stereo_lite(
        self,
        mix_spec: torch.Tensor,
        mix_stereo_spec: Optional[torch.Tensor],
        azimuth_deg: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            mix_spec: [B,F,T] complex.
            mix_stereo_spec: [B,2,F,T] complex or None.
        Returns:
            est_spec: [B,S,F,T] complex.
        """
        feat = self._prepare_stereo_lite_features(mix_spec, mix_stereo_spec)  # [B,F,T,H]
        feat = self._slite_frequency_mix_sequence(feat)

        b, f, t, h = feat.shape

        az_embed = self._azimuth_embedding(
            azimuth_deg=azimuth_deg,
            batch_size=b,
            device=feat.device,
            dtype=feat.dtype,
        )  # [B,S,H] or None

        if az_embed is not None:
            # Strong cue injection: each query gets its own recurrent feature stream.
            q = az_embed.unsqueeze(2).unsqueeze(3)  # [B,S,1,1,H]
            seq_in = feat.unsqueeze(1) + q  # [B,S,F,T,H]
            seq_q = seq_in.reshape(b * self.num_speakers * f, t, h)
            out_q, _ = self._run_slite_gru(seq_q, None)
            out_q = out_q.reshape(b, self.num_speakers, f, t, h)
            out_q = self.slite_out_norm(out_q + seq_in)

            gamma_beta = self.slite_query_film(az_embed).unsqueeze(2).unsqueeze(3)  # [B,S,1,1,2H]
            gamma, beta = torch.chunk(gamma_beta, 2, dim=-1)
            out_q = out_q * (1.0 + 0.5 * torch.tanh(gamma)) + 0.5 * beta

            mask_ri = torch.tanh(self.slite_mask_out_query(out_q))  # [B,S,F,T,2]
            mask_c = torch.view_as_complex(mask_ri.float().contiguous()).to(dtype=mix_spec.dtype)  # [B,S,F,T]
        else:
            seq = feat.reshape(b * f, t, h)
            out, _ = self._run_slite_gru(seq, None)
            out = out.reshape(b, f, t, h)
            out = self.slite_out_norm(out + feat)
            mask_ri = torch.tanh(self.slite_mask_out(out))  # [B,F,T,S*2]
            mask_c = torch.view_as_complex(
                mask_ri.float().reshape(b, f, t, self.num_speakers, 2)
            ).to(dtype=mix_spec.dtype)  # [B,F,T,S]
            mask_c = mask_c.permute(0, 3, 1, 2).contiguous()  # [B,S,F,T]

        est_spec = mask_c * mix_spec.unsqueeze(1)  # [B,S,F,T]
        return est_spec

    def _forward_stereo_stacked_lstm(
        self,
        mix_spec: torch.Tensor,
        mix_stereo_spec: Optional[torch.Tensor],
        azimuth_deg: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            mix_spec: [B,F,T] complex.
            mix_stereo_spec: [B,2,F,T] complex or None.
        Returns:
            est_spec: [B,S,F,T] complex.
        """
        feat = self._prepare_stereo_stacked_lstm_features(mix_spec, mix_stereo_spec)  # [B,F,T,H]
        b, f, t, h = feat.shape

        az_embed = self._azimuth_embedding(
            azimuth_deg=azimuth_deg,
            batch_size=b,
            device=feat.device,
            dtype=feat.dtype,
        )  # [B,S,H] or None

        if az_embed is not None:
            q = az_embed.unsqueeze(2).unsqueeze(3)  # [B,S,1,1,H]
            seq_in = feat.unsqueeze(1) + q  # [B,S,F,T,H]
            seq_q = seq_in.reshape(b * self.num_speakers * f, t, h)
            out_q, _ = self._run_slstm(seq_q, None)
            out_q = out_q.reshape(b, self.num_speakers, f, t, h)
            out_q = self.slstm_out_norm(out_q + seq_in)

            gamma_beta = self.slstm_query_film(az_embed).unsqueeze(2).unsqueeze(3)  # [B,S,1,1,2H]
            gamma, beta = torch.chunk(gamma_beta, 2, dim=-1)
            out_q = out_q * (1.0 + 0.5 * torch.tanh(gamma)) + 0.5 * beta

            mask_ri = torch.tanh(self.slstm_mask_out_query(out_q))  # [B,S,F,T,2]
            mask_c = torch.view_as_complex(mask_ri.float().contiguous()).to(dtype=mix_spec.dtype)  # [B,S,F,T]
        else:
            seq = feat.reshape(b * f, t, h)
            out, _ = self._run_slstm(seq, None)
            out = out.reshape(b, f, t, h)
            out = self.slstm_out_norm(out + feat)
            mask_ri = torch.tanh(self.slstm_mask_out(out))  # [B,F,T,S*2]
            mask_c = torch.view_as_complex(
                mask_ri.float().reshape(b, f, t, self.num_speakers, 2)
            ).to(dtype=mix_spec.dtype)  # [B,F,T,S]
            mask_c = mask_c.permute(0, 3, 1, 2).contiguous()  # [B,S,F,T]

        est_spec = mask_c * mix_spec.unsqueeze(1)  # [B,S,F,T]
        return est_spec

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

    def _normalize_mix_input(self, mix: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Normalize input tensor shape.

        Args:
            mix: [B, T], [B, 2, T], or [B, T, 2]
        Returns:
            mix_mono: [B, T]
            mix_stereo: [B, 2, T] or None
        """
        if mix.ndim == 2:
            return mix, None
        if mix.ndim != 3:
            raise ValueError(f"Expected mix shape [B,T], [B,2,T], or [B,T,2], got {tuple(mix.shape)}")

        if mix.shape[1] == 2:
            mix_stereo = mix
        elif mix.shape[2] == 2:
            mix_stereo = mix.transpose(1, 2).contiguous()
        else:
            raise ValueError(f"Cannot infer stereo channel axis from shape {tuple(mix.shape)}")

        mix_mono = torch.mean(mix_stereo, dim=1)
        return mix_mono, mix_stereo

    def _stft_stereo(self, mix_stereo: torch.Tensor) -> torch.Tensor:
        if mix_stereo.ndim != 3 or mix_stereo.shape[1] != 2:
            raise ValueError(f"Expected mix_stereo [B,2,T], got {tuple(mix_stereo.shape)}")
        b, c, t = mix_stereo.shape
        flat = mix_stereo.reshape(b * c, t)
        spec = self._stft(flat)
        return spec.reshape(b, c, self.freq_bins, -1)

    def _normalize_stereo_spec_frame(self, stereo_spec_frame: torch.Tensor) -> torch.Tensor:
        """
        Args:
            stereo_spec_frame: [B, 2, F] or [B, F, 2]
        Returns:
            [B, 2, F]
        """
        if stereo_spec_frame.ndim != 3:
            raise ValueError(
                f"Expected stereo_spec_frame [B,2,F] or [B,F,2], got {tuple(stereo_spec_frame.shape)}"
            )
        if stereo_spec_frame.shape[1] == 2 and stereo_spec_frame.shape[2] == self.freq_bins:
            return stereo_spec_frame
        if stereo_spec_frame.shape[2] == 2 and stereo_spec_frame.shape[1] == self.freq_bins:
            return stereo_spec_frame.transpose(1, 2).contiguous()
        raise ValueError(
            f"Cannot infer stereo channel axis for frame spec shape {tuple(stereo_spec_frame.shape)}"
        )

    def _build_stereo_features(
        self,
        mix_spec: torch.Tensor,
        stereo_spec: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Build per-TF stereo feature tensor.

        Args:
            mix_spec: [B, F, T] complex
            stereo_spec: [B, 2, F, T] complex or None
        Returns:
            features: [B, T, F, C]
        """
        mono_log = torch.log1p(mix_spec.abs()).transpose(1, 2)  # [B, T, F]
        if stereo_spec is None:
            zeros = torch.zeros_like(mono_log)
            return torch.stack([mono_log, mono_log, mono_log, zeros, zeros, zeros, zeros], dim=-1)

        left = stereo_spec[:, 0]  # [B, F, T]
        right = stereo_spec[:, 1]  # [B, F, T]
        left_mag = torch.clamp(left.abs(), min=self._eps)
        right_mag = torch.clamp(right.abs(), min=self._eps)

        left_log = torch.log1p(left_mag).transpose(1, 2)
        right_log = torch.log1p(right_mag).transpose(1, 2)
        ild = (torch.log(left_mag) - torch.log(right_mag)).transpose(1, 2)

        ipd = torch.angle(left) - torch.angle(right)
        ipd = torch.atan2(torch.sin(ipd), torch.cos(ipd))
        cos_ipd = torch.cos(ipd).transpose(1, 2)
        sin_ipd = torch.sin(ipd).transpose(1, 2)

        cross = left * torch.conj(right)
        coherence = (cross.real / (left_mag * right_mag + self._eps)).transpose(1, 2)

        return torch.stack([mono_log, left_log, right_log, ild, cos_ipd, sin_ipd, coherence], dim=-1)

    def _build_stereo_features_frame(
        self,
        mix_spec_frame: torch.Tensor,
        stereo_spec_frame: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            mix_spec_frame: [B, F] complex
            stereo_spec_frame: [B, 2, F] complex or None
        Returns:
            features: [B, F, C]
        """
        mono_log = torch.log1p(mix_spec_frame.abs())  # [B, F]
        if stereo_spec_frame is None:
            zeros = torch.zeros_like(mono_log)
            return torch.stack([mono_log, mono_log, mono_log, zeros, zeros, zeros, zeros], dim=-1)

        stereo_spec_frame = self._normalize_stereo_spec_frame(stereo_spec_frame)
        left = stereo_spec_frame[:, 0]  # [B, F]
        right = stereo_spec_frame[:, 1]  # [B, F]

        left_mag = torch.clamp(left.abs(), min=self._eps)
        right_mag = torch.clamp(right.abs(), min=self._eps)
        left_log = torch.log1p(left_mag)
        right_log = torch.log1p(right_mag)
        ild = torch.log(left_mag) - torch.log(right_mag)

        ipd = torch.angle(left) - torch.angle(right)
        ipd = torch.atan2(torch.sin(ipd), torch.cos(ipd))
        cos_ipd = torch.cos(ipd)
        sin_ipd = torch.sin(ipd)

        cross = left * torch.conj(right)
        coherence = cross.real / (left_mag * right_mag + self._eps)

        return torch.stack([mono_log, left_log, right_log, ild, cos_ipd, sin_ipd, coherence], dim=-1)

    def _build_input_mag(
        self,
        mix_spec: torch.Tensor,
        stereo_spec: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Build front-end magnitude features consumed by the unchanged separator trunk.
        """
        feats = self._build_stereo_features(mix_spec, stereo_spec=stereo_spec)  # [B, T, F, C]
        base_mag = feats[..., 0]
        if not self.use_learned_stereo_fusion:
            return base_mag

        delta_logits, gain_logits = self._run_stereo_frontend(feats)
        delta = torch.tanh(delta_logits)
        gain_centered = torch.sigmoid(gain_logits) * 2.0 - 1.0  # [-1, 1]
        delta_scale = torch.tanh(self.stereo_delta_scale).to(dtype=delta.dtype, device=delta.device)
        gain_scale = torch.tanh(self.stereo_gain_scale).to(dtype=delta.dtype, device=delta.device)
        gain = 1.0 + 0.5 * gain_scale * gain_centered
        return base_mag * gain + delta_scale * delta

    def _build_input_mag_frame(
        self,
        mix_spec_frame: torch.Tensor,
        stereo_spec_frame: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        feats = self._build_stereo_features_frame(mix_spec_frame, stereo_spec_frame=stereo_spec_frame)  # [B,F,C]
        base_mag = feats[..., 0]
        if not self.use_learned_stereo_fusion:
            return base_mag

        delta_logits, gain_logits = self._run_stereo_frontend_frame(feats)
        delta = torch.tanh(delta_logits)
        gain_centered = torch.sigmoid(gain_logits) * 2.0 - 1.0  # [-1, 1]
        delta_scale = torch.tanh(self.stereo_delta_scale).to(dtype=delta.dtype, device=delta.device)
        gain_scale = torch.tanh(self.stereo_gain_scale).to(dtype=delta.dtype, device=delta.device)
        gain = 1.0 + 0.5 * gain_scale * gain_centered
        return base_mag * gain + delta_scale * delta

    def _istft(self, est_spec: torch.Tensor, length: int) -> torch.Tensor:
        if est_spec.ndim == 4:
            b, s, f, t = est_spec.shape
            out_shape = (b, s, -1)
            flat = est_spec.reshape(b * s, f, t)
        elif est_spec.ndim == 5:
            b, s, c, f, t = est_spec.shape
            out_shape = (b, s, c, -1)
            flat = est_spec.reshape(b * s * c, f, t)
        else:
            raise ValueError(f"Expected est_spec [B,S,F,T] or [B,S,C,F,T], got {tuple(est_spec.shape)}")

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
        return wav.reshape(*out_shape)

    def _run_gru(self, feat: torch.Tensor, hidden: Optional[torch.Tensor]):
        use_fp32 = (
            feat.device.type == "cuda"
            and torch.is_autocast_enabled()
            and (self.force_rnn_fp32 or self._runtime_disable_bf16_rnn)
        )
        if use_fp32:
            hidden_fp32 = None if hidden is None else hidden.float()
            with torch.cuda.amp.autocast(enabled=False):
                out_fp32, hidden_out = self.rnn(feat.float(), hidden_fp32)
            return out_fp32.to(feat.dtype), hidden_out
        try:
            return self.rnn(feat, hidden)
        except RuntimeError as exc:
            msg = str(exc).lower()
            unsupported_bf16 = (
                feat.device.type == "cuda"
                and feat.dtype == torch.bfloat16
                and ("bfloat16" in msg)
                and ("not implemented" in msg or "cudnn_status_not_supported" in msg)
            )
            if not unsupported_bf16:
                raise
            self._runtime_disable_bf16_rnn = True
            if not self._warned_bf16_rnn_fallback:
                warnings.warn(
                    "BF16 recurrent kernels are unavailable in current torch/cuda env. "
                    "Falling back to FP32 for GRU/LSTM blocks.",
                    RuntimeWarning,
                )
                self._warned_bf16_rnn_fallback = True
            hidden_fp32 = None if hidden is None else hidden.float()
            with torch.cuda.amp.autocast(enabled=False):
                out_fp32, hidden_out = self.rnn(feat.float(), hidden_fp32)
            return out_fp32.to(feat.dtype), hidden_out

    def _run_lstm(
        self,
        feat: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        use_fp32 = (
            feat.device.type == "cuda"
            and torch.is_autocast_enabled()
            and (self.force_rnn_fp32 or self._runtime_disable_bf16_rnn)
        )
        if use_fp32:
            state_fp32 = None
            if state is not None:
                state_fp32 = (state[0].float(), state[1].float())
            with torch.cuda.amp.autocast(enabled=False):
                out_fp32, state_out = self.lstm(feat.float(), state_fp32)
            return out_fp32.to(feat.dtype), state_out
        try:
            return self.lstm(feat, state)
        except RuntimeError as exc:
            msg = str(exc).lower()
            unsupported_bf16 = (
                feat.device.type == "cuda"
                and feat.dtype == torch.bfloat16
                and ("bfloat16" in msg)
                and ("not implemented" in msg or "cudnn_status_not_supported" in msg)
            )
            if not unsupported_bf16:
                raise
            self._runtime_disable_bf16_rnn = True
            if not self._warned_bf16_rnn_fallback:
                warnings.warn(
                    "BF16 recurrent kernels are unavailable in current torch/cuda env. "
                    "Falling back to FP32 for GRU/LSTM blocks.",
                    RuntimeWarning,
                )
                self._warned_bf16_rnn_fallback = True
            state_fp32 = None
            if state is not None:
                state_fp32 = (state[0].float(), state[1].float())
            with torch.cuda.amp.autocast(enabled=False):
                out_fp32, state_out = self.lstm(feat.float(), state_fp32)
            return out_fp32.to(feat.dtype), state_out

    def _apply_blocks(self, feat: torch.Tensor, blocks: nn.ModuleList) -> torch.Tensor:
        for block in blocks:
            feat = block(feat)
        return feat

    def get_auxiliary_outputs(self) -> Dict[str, torch.Tensor]:
        aux = getattr(self, "_aux_outputs", None)
        return aux if isinstance(aux, dict) else {}

    def _clear_auxiliary_outputs(self):
        self._aux_outputs = {}

    def _estimate_masks_gru(
        self,
        mix_spec: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
        input_mag: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mag = input_mag if input_mag is not None else torch.log1p(mix_spec.abs()).transpose(1, 2)
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
        input_mag: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, None, torch.Tensor]:
        mag = input_mag if input_mag is not None else torch.log1p(mix_spec.abs()).transpose(1, 2)
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
        input_mag: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        mag = input_mag if input_mag is not None else torch.log1p(mix_spec.abs()).transpose(1, 2)
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
        input_mag: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        if self.architecture == "gru":
            return self._estimate_masks_gru(mix_spec, hidden, input_mag=input_mag)
        if self.architecture == "tcn":
            return self._estimate_masks_tcn(mix_spec, input_mag=input_mag)
        if mix_wave is None:
            raise ValueError("mix_wave is required for lstm_hybrid architecture")
        return self._estimate_masks_lstm_hybrid(mix_spec, mix_wave, input_mag=input_mag)

    def forward(self, mix: torch.Tensor, azimuth_deg: Optional[torch.Tensor] = None) -> torch.Tensor:
        self._clear_auxiliary_outputs()
        mix_mono, mix_stereo = self._normalize_mix_input(mix)
        mix_spec = self._stft(mix_mono)
        stereo_spec = self._stft_stereo(mix_stereo) if mix_stereo is not None else None
        if self.architecture == "stereo_lite":
            est_spec = self._forward_stereo_lite(mix_spec, stereo_spec, azimuth_deg=azimuth_deg)
            return self._istft(est_spec, length=mix_mono.shape[-1])
        if self.architecture == "stereo_stacked_lstm":
            est_spec = self._forward_stereo_stacked_lstm(mix_spec, stereo_spec, azimuth_deg=azimuth_deg)
            return self._istft(est_spec, length=mix_mono.shape[-1])
        if self.architecture == "stereo_beam_lite":
            est_spec = self._forward_stereo_beam_lite(mix_spec, stereo_spec, azimuth_deg=azimuth_deg)
            return self._istft(est_spec, length=mix_mono.shape[-1])
        if self.architecture == "online_spatialnet":
            est_spec = self._forward_online_spatialnet(mix_spec, stereo_spec)
            return self._istft(est_spec, length=mix_mono.shape[-1])

        input_mag = self._build_input_mag(mix_spec, stereo_spec=stereo_spec)
        est_spec, _, _ = self._estimate_masks(
            mix_spec,
            hidden=None,
            mix_wave=mix_mono,
            input_mag=input_mag,
        )
        return self._istft(est_spec, length=mix_mono.shape[-1])

    @torch.no_grad()
    def _forward_step_gru(
        self,
        mix_spec_frame: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
        input_mag_frame: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mag = input_mag_frame.unsqueeze(1) if input_mag_frame is not None else torch.log1p(mix_spec_frame.abs()).unsqueeze(1)
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
        input_mag_frame: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, List[torch.Tensor]], torch.Tensor]:
        mag = input_mag_frame.unsqueeze(1) if input_mag_frame is not None else torch.log1p(mix_spec_frame.abs()).unsqueeze(1)
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
        input_mag_frame: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
        mag = input_mag_frame.unsqueeze(1) if input_mag_frame is not None else torch.log1p(mix_spec_frame.abs()).unsqueeze(1)
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
    def _forward_step_online_spatialnet_windowed(
        self,
        mix_spec_frame: torch.Tensor,
        hidden: Optional[Dict[str, torch.Tensor]] = None,
        mix_spec_frame_stereo: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """
        Streaming step for OnlineSpatialNet branch.

        This keeps a bounded history of normalized stereo TF features and re-runs
        the causal OnlineSpatialNet on that history, then emits the last frame.
        """
        if mix_spec_frame_stereo is None:
            mix_spec_frame_stereo = torch.stack([mix_spec_frame, mix_spec_frame], dim=1)
        else:
            mix_spec_frame_stereo = self._normalize_stereo_spec_frame(mix_spec_frame_stereo)

        b = mix_spec_frame.shape[0]
        ref_mag = torch.clamp(mix_spec_frame_stereo[:, 0, :].abs(), min=self._eps)  # [B,F]

        if isinstance(hidden, dict) and hidden.get("arch", "") == "online_spatialnet":
            prev_mean = hidden.get("ref_mean_mag", None)
            prev_count = hidden.get("ref_count", None)
            feat_hist = hidden.get("feat_hist", None)
        else:
            prev_mean, prev_count, feat_hist = None, None, None

        if (prev_mean is None) or (prev_count is None):
            ref_count = torch.ones_like(ref_mag)
            ref_mean_mag = ref_mag
        else:
            ref_count = prev_count + 1.0
            ref_mean_mag = (prev_mean * prev_count + ref_mag) / torch.clamp(ref_count, min=1.0)
        ref_mean_mag = torch.clamp(ref_mean_mag, min=self._eps)

        stereo_norm = mix_spec_frame_stereo / ref_mean_mag.unsqueeze(1)  # [B,2,F]
        frame_feat = torch.view_as_real(stereo_norm.transpose(1, 2).contiguous()).reshape(b, self.freq_bins, -1)  # [B,F,4]

        if feat_hist is None:
            feat_seq = frame_feat.unsqueeze(2)  # [B,F,1,4]
        else:
            feat_seq = torch.cat([feat_hist, frame_feat.unsqueeze(2)], dim=2)
        max_hist = max(4, int(self.osn_streaming_history))
        if feat_seq.shape[2] > max_hist:
            feat_seq = feat_seq[:, :, -max_hist:, :].contiguous()

        out_seq = self.osn(feat_seq, inference=False)  # [B,F,T,S*2]
        out_last = out_seq[:, :, -1, :]  # [B,F,S*2]
        out_c = torch.view_as_complex(out_last.float().reshape(b, self.freq_bins, self.num_speakers, 2))  # [B,F,S]
        out_c = out_c * ref_mean_mag.unsqueeze(-1)
        est_spec = out_c.permute(0, 2, 1).contiguous()  # [B,S,F]

        mag_denom = torch.clamp(mix_spec_frame.abs().unsqueeze(1), min=self._eps)
        pseudo_mask = torch.clamp(est_spec.abs() / mag_denom, min=0.0, max=10.0)

        new_hidden = {
            "arch": "online_spatialnet_windowed",
            "feat_hist": feat_seq.detach(),
            "ref_mean_mag": ref_mean_mag.detach(),
            "ref_count": ref_count.detach(),
        }
        return est_spec, new_hidden, pseudo_mask

    @torch.no_grad()
    def _forward_step_online_spatialnet_true(
        self,
        mix_spec_frame: torch.Tensor,
        hidden: Optional[Dict[str, torch.Tensor]] = None,
        mix_spec_frame_stereo: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        if mix_spec_frame_stereo is None:
            mix_spec_frame_stereo = torch.stack([mix_spec_frame, mix_spec_frame], dim=1)
        else:
            mix_spec_frame_stereo = self._normalize_stereo_spec_frame(mix_spec_frame_stereo)

        b = mix_spec_frame.shape[0]
        ref_mag = torch.clamp(mix_spec_frame_stereo[:, 0, :].abs(), min=self._eps)  # [B,F]

        osn_state = None
        frame_index = 0
        prev_mean = None
        prev_count = None
        if isinstance(hidden, dict) and hidden.get("arch", "") == "online_spatialnet_true":
            osn_state = hidden.get("osn_state", None)
            frame_index = int(hidden.get("frame_index", 0))
            prev_mean = hidden.get("ref_mean_mag", None)
            prev_count = hidden.get("ref_count", None)

        if (prev_mean is None) or (prev_count is None) or (prev_mean.shape != ref_mag.shape):
            ref_count = torch.ones_like(ref_mag)
            ref_mean_mag = ref_mag
        else:
            ref_count = prev_count + 1.0
            ref_mean_mag = (prev_mean * prev_count + ref_mag) / torch.clamp(ref_count, min=1.0)
        ref_mean_mag = torch.clamp(ref_mean_mag, min=self._eps)

        if (osn_state is None) or (not isinstance(osn_state, dict)):
            osn_state = {
                "encoder": {},
                "layers": [dict() for _ in range(len(self.osn.layers))],
            }

        stereo_norm = mix_spec_frame_stereo / ref_mean_mag.unsqueeze(1)  # [B,2,F]
        frame_feat = torch.view_as_real(stereo_norm.transpose(1, 2).contiguous()).reshape(b, self.freq_bins, -1)  # [B,F,4]

        # encoder expects [B*F, C_in, T]
        x = frame_feat.reshape(b * self.freq_bins, 1, -1).permute(0, 2, 1).contiguous()  # [B*F,4,1]
        x = self.osn.encoder(x, state=osn_state["encoder"]).permute(0, 2, 1).contiguous()  # [B*F,1,H]
        hdim = x.shape[-1]
        x = x.reshape(b, self.freq_bins, 1, hdim)

        layer_states = osn_state.get("layers", [])
        if len(layer_states) != len(self.osn.layers):
            layer_states = [dict() for _ in range(len(self.osn.layers))]
            osn_state["layers"] = layer_states

        # recurrent relative position for one-step retention update
        if hasattr(self.osn, "pos") and (self.osn.pos is not None):
            rel_pos = self.osn.pos.forward(slen=frame_index, activate_recurrent=True)
        else:
            rel_pos = self.osn.get_causal_mask(
                slen=1,
                device=x.device,
                chunkwise_recurrent=False,
                batch_size=b,
                inference=False,
            )

        for i, layer in enumerate(self.osn.layers):
            x, _ = layer(
                x,
                rel_pos,
                chunkwise_recurrent=False,
                rope=self.osn.rope,
                state=layer_states[i],
                inference=False,
            )

        out = self.osn.decoder(x)  # [B,F,1,S*2]
        out_last = out[:, :, 0, :]  # [B,F,S*2]
        out_c = torch.view_as_complex(out_last.float().reshape(b, self.freq_bins, self.num_speakers, 2))  # [B,F,S]
        out_c = out_c * ref_mean_mag.unsqueeze(-1)
        est_spec = out_c.permute(0, 2, 1).contiguous()  # [B,S,F]

        mag_denom = torch.clamp(mix_spec_frame.abs().unsqueeze(1), min=self._eps)
        pseudo_mask = torch.clamp(est_spec.abs() / mag_denom, min=0.0, max=10.0)

        new_hidden = {
            "arch": "online_spatialnet_true",
            "osn_state": osn_state,
            "frame_index": frame_index + 1,
            "ref_mean_mag": ref_mean_mag.detach(),
            "ref_count": ref_count.detach(),
        }
        return est_spec, new_hidden, pseudo_mask

    @torch.no_grad()
    def _forward_step_online_spatialnet(
        self,
        mix_spec_frame: torch.Tensor,
        hidden: Optional[Dict[str, torch.Tensor]] = None,
        mix_spec_frame_stereo: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        attention = str(getattr(self, "osn_attention", "")).lower()
        if attention in {"", "mhsa"}:
            attention = "mhsa"
        if attention.startswith("ret"):
            return self._forward_step_online_spatialnet_true(
                mix_spec_frame=mix_spec_frame,
                hidden=hidden,
                mix_spec_frame_stereo=mix_spec_frame_stereo,
            )
        # For non-retention attention, fallback to windowed re-computation.
        return self._forward_step_online_spatialnet_windowed(
            mix_spec_frame=mix_spec_frame,
            hidden=hidden,
            mix_spec_frame_stereo=mix_spec_frame_stereo,
        )

    @torch.no_grad()
    def _forward_step_stereo_lite(
        self,
        mix_spec_frame: torch.Tensor,
        hidden: Optional[Dict[str, torch.Tensor]] = None,
        mix_spec_frame_stereo: Optional[torch.Tensor] = None,
        azimuth_deg: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        if mix_spec_frame_stereo is None:
            mix_spec_frame_stereo = torch.stack([mix_spec_frame, mix_spec_frame], dim=1)
        else:
            mix_spec_frame_stereo = self._normalize_stereo_spec_frame(mix_spec_frame_stereo)

        left = mix_spec_frame_stereo[:, 0]  # [B,F]
        right = mix_spec_frame_stereo[:, 1]  # [B,F]
        ref_mag = torch.clamp(0.5 * (left.abs() + right.abs()), min=self._eps)

        prev_gru = None
        prev_mean = None
        prev_count = None
        if isinstance(hidden, dict) and hidden.get("arch", "") == "stereo_lite":
            prev_gru = hidden.get("gru_state", None)
            prev_mean = hidden.get("ref_mean_mag", None)
            prev_count = hidden.get("ref_count", None)

        if (prev_mean is None) or (prev_count is None) or (prev_mean.shape != ref_mag.shape):
            ref_count = torch.ones_like(ref_mag)
            ref_mean_mag = ref_mag
        else:
            ref_count = prev_count + 1.0
            ref_mean_mag = (prev_mean * prev_count + ref_mag) / torch.clamp(ref_count, min=1.0)
        ref_mean_mag = torch.clamp(ref_mean_mag, min=self._eps)

        left_n = left / ref_mean_mag
        right_n = right / ref_mean_mag
        feat = torch.stack([left_n.real, left_n.imag, right_n.real, right_n.imag], dim=-1)  # [B,F,4]
        feat = self.slite_in_norm(feat)
        feat = self.slite_in_proj(feat)
        feat = self.slite_in_act(feat)
        feat = self.slite_dropout(feat)
        feat = self._slite_frequency_mix_frame(feat)  # [B,F,H]

        b, f, h = feat.shape

        az_embed = self._azimuth_embedding(
            azimuth_deg=azimuth_deg,
            batch_size=b,
            device=feat.device,
            dtype=feat.dtype,
        )  # [B,S,H] or None

        if az_embed is not None:
            # Query-conditioned recurrent step for stronger directional binding.
            q = az_embed.unsqueeze(2)  # [B,S,1,H]
            step_in = feat.unsqueeze(1) + q  # [B,S,F,H]
            step_in_flat = step_in.reshape(b * self.num_speakers * f, 1, h)

            q_gru_state = None
            if isinstance(prev_gru, torch.Tensor) and prev_gru.ndim == 5:
                # [L,B,S,F,H] -> [L,B*S*F,H]
                if (
                    prev_gru.shape[0] == self.num_layers
                    and prev_gru.shape[1] == b
                    and prev_gru.shape[2] == self.num_speakers
                    and prev_gru.shape[3] == f
                ):
                    q_gru_state = prev_gru.reshape(self.num_layers, b * self.num_speakers * f, h).contiguous()

            out_q, q_state_new = self._run_slite_gru(step_in_flat, q_gru_state)
            out_q = out_q.squeeze(1).reshape(b, self.num_speakers, f, h)
            out_q = self.slite_out_norm(out_q + step_in)

            gamma_beta = self.slite_query_film(az_embed).unsqueeze(2)  # [B,S,1,2H]
            gamma, beta = torch.chunk(gamma_beta, 2, dim=-1)
            out_q = out_q * (1.0 + 0.5 * torch.tanh(gamma)) + 0.5 * beta

            mask_ri = torch.tanh(self.slite_mask_out_query(out_q))  # [B,S,F,2]
            mask_c = torch.view_as_complex(mask_ri.float().contiguous()).to(dtype=mix_spec_frame.dtype)  # [B,S,F]
            gru_state_pack = q_state_new.reshape(self.num_layers, b, self.num_speakers, f, h).detach()
        else:
            gru_in = feat.reshape(b * f, 1, h)
            gru_state = None
            if isinstance(prev_gru, torch.Tensor) and prev_gru.ndim == 4:
                if prev_gru.shape[0] == self.num_layers and prev_gru.shape[1] == b and prev_gru.shape[2] == f:
                    gru_state = prev_gru.reshape(self.num_layers, b * f, h).contiguous()

            out, gru_state_new = self._run_slite_gru(gru_in, gru_state)
            out = out.squeeze(1).reshape(b, f, h)
            out = self.slite_out_norm(out + feat)
            mask_ri = torch.tanh(self.slite_mask_out(out))  # [B,F,S*2]
            mask_c = torch.view_as_complex(
                mask_ri.float().reshape(b, f, self.num_speakers, 2)
            ).to(dtype=mix_spec_frame.dtype)  # [B,F,S]
            mask_c = mask_c.permute(0, 2, 1).contiguous()  # [B,S,F]
            gru_state_pack = gru_state_new.reshape(self.num_layers, b, f, h).detach()

        est_spec = mask_c * mix_spec_frame.unsqueeze(1)  # [B,S,F]
        pseudo_mask = mask_c.abs()

        new_hidden = {
            "arch": "stereo_lite",
            "gru_state": gru_state_pack,
            "ref_mean_mag": ref_mean_mag.detach(),
            "ref_count": ref_count.detach(),
        }
        return est_spec, new_hidden, pseudo_mask

    @torch.no_grad()
    def _forward_step_stereo_stacked_lstm(
        self,
        mix_spec_frame: torch.Tensor,
        hidden: Optional[Dict[str, torch.Tensor]] = None,
        mix_spec_frame_stereo: Optional[torch.Tensor] = None,
        azimuth_deg: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        if mix_spec_frame_stereo is None:
            mix_spec_frame_stereo = torch.stack([mix_spec_frame, mix_spec_frame], dim=1)
        else:
            mix_spec_frame_stereo = self._normalize_stereo_spec_frame(mix_spec_frame_stereo)

        left = mix_spec_frame_stereo[:, 0]  # [B,F]
        right = mix_spec_frame_stereo[:, 1]  # [B,F]
        ref_mag = torch.clamp(0.5 * (left.abs() + right.abs()), min=self._eps)

        prev_lstm = None
        prev_mean = None
        prev_count = None
        if isinstance(hidden, dict) and hidden.get("arch", "") == "stereo_stacked_lstm":
            prev_lstm = hidden.get("lstm_state", None)
            prev_mean = hidden.get("ref_mean_mag", None)
            prev_count = hidden.get("ref_count", None)

        if (prev_mean is None) or (prev_count is None) or (prev_mean.shape != ref_mag.shape):
            ref_count = torch.ones_like(ref_mag)
            ref_mean_mag = ref_mag
        else:
            ref_count = prev_count + 1.0
            ref_mean_mag = (prev_mean * prev_count + ref_mag) / torch.clamp(ref_count, min=1.0)
        ref_mean_mag = torch.clamp(ref_mean_mag, min=self._eps)

        left_n = left / ref_mean_mag
        right_n = right / ref_mean_mag
        feat = torch.stack([left_n.real, left_n.imag, right_n.real, right_n.imag], dim=-1)  # [B,F,4]
        feat = self.slstm_in_norm(feat)
        feat = self.slstm_in_proj(feat)
        feat = self.slstm_in_act(feat)
        feat = self.slstm_dropout(feat)

        b, f, h = feat.shape

        az_embed = self._azimuth_embedding(
            azimuth_deg=azimuth_deg,
            batch_size=b,
            device=feat.device,
            dtype=feat.dtype,
        )  # [B,S,H] or None

        if az_embed is not None:
            q = az_embed.unsqueeze(2)  # [B,S,1,H]
            step_in = feat.unsqueeze(1) + q  # [B,S,F,H]
            step_in_flat = step_in.reshape(b * self.num_speakers * f, 1, h)

            q_lstm_state = None
            if isinstance(prev_lstm, tuple) and len(prev_lstm) == 2:
                prev_h, prev_c = prev_lstm
                if (
                    isinstance(prev_h, torch.Tensor)
                    and isinstance(prev_c, torch.Tensor)
                    and prev_h.ndim == 5
                    and prev_c.ndim == 5
                    and prev_h.shape[0] == self.num_layers
                    and prev_h.shape[1] == b
                    and prev_h.shape[2] == self.num_speakers
                    and prev_h.shape[3] == f
                ):
                    q_lstm_state = (
                        prev_h.reshape(self.num_layers, b * self.num_speakers * f, h).contiguous(),
                        prev_c.reshape(self.num_layers, b * self.num_speakers * f, h).contiguous(),
                    )

            out_q, q_state_new = self._run_slstm(step_in_flat, q_lstm_state)
            out_q = out_q.squeeze(1).reshape(b, self.num_speakers, f, h)
            out_q = self.slstm_out_norm(out_q + step_in)

            gamma_beta = self.slstm_query_film(az_embed).unsqueeze(2)  # [B,S,1,2H]
            gamma, beta = torch.chunk(gamma_beta, 2, dim=-1)
            out_q = out_q * (1.0 + 0.5 * torch.tanh(gamma)) + 0.5 * beta

            mask_ri = torch.tanh(self.slstm_mask_out_query(out_q))  # [B,S,F,2]
            mask_c = torch.view_as_complex(mask_ri.float().contiguous()).to(dtype=mix_spec_frame.dtype)  # [B,S,F]
            lstm_state_pack = (
                q_state_new[0].reshape(self.num_layers, b, self.num_speakers, f, h).detach(),
                q_state_new[1].reshape(self.num_layers, b, self.num_speakers, f, h).detach(),
            )
        else:
            lstm_in = feat.reshape(b * f, 1, h)
            lstm_state = None
            if isinstance(prev_lstm, tuple) and len(prev_lstm) == 2:
                prev_h, prev_c = prev_lstm
                if (
                    isinstance(prev_h, torch.Tensor)
                    and isinstance(prev_c, torch.Tensor)
                    and prev_h.ndim == 4
                    and prev_c.ndim == 4
                    and prev_h.shape[0] == self.num_layers
                    and prev_h.shape[1] == b
                    and prev_h.shape[2] == f
                ):
                    lstm_state = (
                        prev_h.reshape(self.num_layers, b * f, h).contiguous(),
                        prev_c.reshape(self.num_layers, b * f, h).contiguous(),
                    )

            out, lstm_state_new = self._run_slstm(lstm_in, lstm_state)
            out = out.squeeze(1).reshape(b, f, h)
            out = self.slstm_out_norm(out + feat)
            mask_ri = torch.tanh(self.slstm_mask_out(out))  # [B,F,S*2]
            mask_c = torch.view_as_complex(
                mask_ri.float().reshape(b, f, self.num_speakers, 2)
            ).to(dtype=mix_spec_frame.dtype)  # [B,F,S]
            mask_c = mask_c.permute(0, 2, 1).contiguous()  # [B,S,F]
            lstm_state_pack = (
                lstm_state_new[0].reshape(self.num_layers, b, f, h).detach(),
                lstm_state_new[1].reshape(self.num_layers, b, f, h).detach(),
            )

        est_spec = mask_c * mix_spec_frame.unsqueeze(1)  # [B,S,F]
        pseudo_mask = mask_c.abs()

        new_hidden = {
            "arch": "stereo_stacked_lstm",
            "lstm_state": lstm_state_pack,
            "ref_mean_mag": ref_mean_mag.detach(),
            "ref_count": ref_count.detach(),
        }
        return est_spec, new_hidden, pseudo_mask


    def _init_stereo_beam_lite_modules(self):
        in_dim = 4
        self.sbeam_output_channels = int(self.output_source_channels)
        self.sbeam_in_norm = nn.LayerNorm(in_dim)
        self.sbeam_in_proj = nn.Linear(in_dim, self.hidden_size)
        self.sbeam_in_act = nn.SiLU()
        self.sbeam_dropout = nn.Dropout(self.dropout) if self.dropout > 0.0 else nn.Identity()

        self.sbeam_lstm = nn.LSTM(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0.0,
        )
        self.sbeam_out_norm = nn.LayerNorm(self.hidden_size)
        self.sbeam_query_gate_floor = 0.25

        if self.use_azimuth_conditioning:
            self.azimuth_proj = nn.Sequential(
                nn.Linear(2, self.hidden_size),
                nn.SiLU(),
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.SiLU(),
            )
            self.sbeam_query_film = nn.Linear(self.hidden_size, self.hidden_size * 2)
            self.sbeam_query_refine = nn.Sequential(
                nn.Linear(self.hidden_size * 2, self.hidden_size),
                nn.SiLU(),
                nn.Dropout(self.dropout) if self.dropout > 0.0 else nn.Identity(),
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.SiLU(),
            )
            comp_hidden = max(32, self.hidden_size // 2)
            self.sbeam_query_competition = nn.Sequential(
                nn.Linear(self.hidden_size, comp_hidden),
                nn.SiLU(),
                nn.Linear(comp_hidden, 1),
            )
            self.sbeam_filter_out_query = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.SiLU(),
                nn.Linear(self.hidden_size, self.sbeam_output_channels * 4),
            )
            self.sbeam_query_temperature = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.sbeam_filter_out = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.SiLU(),
            nn.Linear(self.hidden_size, self.num_speakers * self.sbeam_output_channels * 4),
        )

    def _run_sbeam_lstm(
        self,
        feat: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        use_fp32 = (
            feat.device.type == "cuda"
            and torch.is_autocast_enabled()
            and (self.force_rnn_fp32 or self._runtime_disable_bf16_rnn)
        )
        if use_fp32:
            state_fp32 = None
            if state is not None:
                state_fp32 = (state[0].float(), state[1].float())
            with torch.cuda.amp.autocast(enabled=False):
                out_fp32, state_out = self.sbeam_lstm(feat.float(), state_fp32)
            return out_fp32.to(feat.dtype), state_out
        try:
            return self.sbeam_lstm(feat, state)
        except RuntimeError as exc:
            msg = str(exc).lower()
            unsupported_bf16 = (
                feat.device.type == "cuda"
                and feat.dtype == torch.bfloat16
                and ("bfloat16" in msg)
                and ("not implemented" in msg or "cudnn_status_not_supported" in msg)
            )
            if not unsupported_bf16:
                raise
            self._runtime_disable_bf16_rnn = True
            if not self._warned_bf16_rnn_fallback:
                warnings.warn(
                    "BF16 recurrent kernels are unavailable in current torch/cuda env. "
                    "Falling back to FP32 for GRU/LSTM blocks.",
                    RuntimeWarning,
                )
                self._warned_bf16_rnn_fallback = True
            state_fp32 = None
            if state is not None:
                state_fp32 = (state[0].float(), state[1].float())
            with torch.cuda.amp.autocast(enabled=False):
                out_fp32, state_out = self.sbeam_lstm(feat.float(), state_fp32)
            return out_fp32.to(feat.dtype), state_out

    def _prepare_stereo_beam_lite_features(
        self,
        mix_spec: torch.Tensor,
        mix_stereo_spec: Optional[torch.Tensor],
    ) -> torch.Tensor:
        stereo_spec = self._normalize_osn_stereo_spec(mix_stereo_spec, mix_spec)
        left = stereo_spec[:, 0]
        right = stereo_spec[:, 1]

        ref_mag = 0.5 * (left.abs() + right.abs())
        ref_scale = torch.clamp(ref_mag.mean(dim=-1, keepdim=True), min=self._eps)
        left_n = left / ref_scale
        right_n = right / ref_scale

        feat = torch.stack([left_n.real, left_n.imag, right_n.real, right_n.imag], dim=-1)
        feat = self.sbeam_in_norm(feat)
        feat = self.sbeam_in_proj(feat)
        feat = self.sbeam_in_act(feat)
        return self.sbeam_dropout(feat)

    def _decode_stereo_beam_lite_sequence(
        self,
        h_shared: torch.Tensor,
        stereo_spec: torch.Tensor,
        azimuth_deg: Optional[torch.Tensor],
    ) -> torch.Tensor:
        b, f, t, h = h_shared.shape
        stereo_x = stereo_spec.permute(0, 2, 3, 1).unsqueeze(1).unsqueeze(-2).contiguous()
        az_embed = self._azimuth_embedding(
            azimuth_deg=azimuth_deg,
            batch_size=b,
            device=h_shared.device,
            dtype=h_shared.dtype,
        )

        if az_embed is not None:
            shared_q = h_shared.unsqueeze(1).expand(-1, self.num_speakers, -1, -1, -1)
            gamma_beta = self.sbeam_query_film(az_embed).unsqueeze(2).unsqueeze(3)
            gamma, beta = torch.chunk(gamma_beta, 2, dim=-1)
            shared_q = shared_q * (1.0 + 0.5 * torch.tanh(gamma)) + 0.5 * beta

            q = az_embed.unsqueeze(2).unsqueeze(3).expand(-1, -1, f, t, -1)
            query_feat = self.sbeam_query_refine(torch.cat([shared_q, q], dim=-1))

            comp_logits = self.sbeam_query_competition(query_feat).squeeze(-1)
            temp = torch.clamp(self.sbeam_query_temperature, min=0.25, max=4.0)
            temp = temp.to(device=h_shared.device, dtype=h_shared.dtype)
            query_assign = torch.softmax(comp_logits / temp, dim=1)
            gate = torch.clamp(
                query_assign * float(self.num_speakers),
                min=float(self.sbeam_query_gate_floor),
                max=float(self.num_speakers),
            )
            filter_ri = torch.tanh(self.sbeam_filter_out_query(query_feat)) * gate.unsqueeze(-1)
            self._aux_outputs["query_assign"] = query_assign
        else:
            filter_ri = torch.tanh(self.sbeam_filter_out(h_shared))
            filter_ri = filter_ri.reshape(
                b,
                f,
                t,
                self.num_speakers,
                self.sbeam_output_channels * 4,
            ).permute(0, 3, 1, 2, 4).contiguous()

        weights = torch.view_as_complex(
            filter_ri.float().reshape(
                b,
                self.num_speakers,
                f,
                t,
                self.sbeam_output_channels,
                2,
                2,
            )
        )
        est_spec = (weights.to(dtype=stereo_x.dtype) * stereo_x).sum(dim=-1)
        return est_spec.permute(0, 1, 4, 2, 3).contiguous()

    def _decode_stereo_beam_lite_frame(
        self,
        h_shared: torch.Tensor,
        stereo_spec_frame: torch.Tensor,
        azimuth_deg: Optional[torch.Tensor],
    ) -> torch.Tensor:
        b, f, h = h_shared.shape
        stereo_x = stereo_spec_frame.transpose(1, 2).unsqueeze(1).unsqueeze(-2).contiguous()
        az_embed = self._azimuth_embedding(
            azimuth_deg=azimuth_deg,
            batch_size=b,
            device=h_shared.device,
            dtype=h_shared.dtype,
        )

        if az_embed is not None:
            shared_q = h_shared.unsqueeze(1).expand(-1, self.num_speakers, -1, -1)
            gamma_beta = self.sbeam_query_film(az_embed).unsqueeze(2)
            gamma, beta = torch.chunk(gamma_beta, 2, dim=-1)
            shared_q = shared_q * (1.0 + 0.5 * torch.tanh(gamma)) + 0.5 * beta

            q = az_embed.unsqueeze(2).expand(-1, -1, f, -1)
            query_feat = self.sbeam_query_refine(torch.cat([shared_q, q], dim=-1))

            comp_logits = self.sbeam_query_competition(query_feat).squeeze(-1)
            temp = torch.clamp(self.sbeam_query_temperature, min=0.25, max=4.0)
            temp = temp.to(device=h_shared.device, dtype=h_shared.dtype)
            query_assign = torch.softmax(comp_logits / temp, dim=1)
            gate = torch.clamp(
                query_assign * float(self.num_speakers),
                min=float(self.sbeam_query_gate_floor),
                max=float(self.num_speakers),
            )
            filter_ri = torch.tanh(self.sbeam_filter_out_query(query_feat)) * gate.unsqueeze(-1)
            self._aux_outputs["query_assign"] = query_assign
        else:
            filter_ri = torch.tanh(self.sbeam_filter_out(h_shared))
            filter_ri = filter_ri.reshape(
                b,
                f,
                self.num_speakers,
                self.sbeam_output_channels * 4,
            ).permute(0, 2, 1, 3).contiguous()

        weights = torch.view_as_complex(
            filter_ri.float().reshape(
                b,
                self.num_speakers,
                f,
                self.sbeam_output_channels,
                2,
                2,
            )
        )
        est_spec = (weights.to(dtype=stereo_x.dtype) * stereo_x).sum(dim=-1)
        return est_spec.permute(0, 1, 3, 2).contiguous()

    def _forward_stereo_beam_lite(
        self,
        mix_spec: torch.Tensor,
        mix_stereo_spec: Optional[torch.Tensor],
        azimuth_deg: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        stereo_spec = self._normalize_osn_stereo_spec(mix_stereo_spec, mix_spec)
        feat = self._prepare_stereo_beam_lite_features(mix_spec, stereo_spec)
        b, f, t, h = feat.shape

        seq = feat.reshape(b * f, t, h)
        out, _ = self._run_sbeam_lstm(seq, None)
        out = out.reshape(b, f, t, h)
        h_shared = self.sbeam_out_norm(out + feat)
        return self._decode_stereo_beam_lite_sequence(h_shared, stereo_spec, azimuth_deg=azimuth_deg)

    @torch.no_grad()
    def _forward_step_stereo_beam_lite(
        self,
        mix_spec_frame: torch.Tensor,
        hidden: Optional[Dict[str, torch.Tensor]] = None,
        mix_spec_frame_stereo: Optional[torch.Tensor] = None,
        azimuth_deg: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        if mix_spec_frame_stereo is None:
            mix_spec_frame_stereo = torch.stack([mix_spec_frame, mix_spec_frame], dim=1)
        else:
            mix_spec_frame_stereo = self._normalize_stereo_spec_frame(mix_spec_frame_stereo)

        left = mix_spec_frame_stereo[:, 0]
        right = mix_spec_frame_stereo[:, 1]
        ref_mag = torch.clamp(0.5 * (left.abs() + right.abs()), min=self._eps)

        prev_lstm = None
        prev_mean = None
        prev_count = None
        if isinstance(hidden, dict) and hidden.get("arch", "") == "stereo_beam_lite":
            prev_lstm = hidden.get("lstm_state", None)
            prev_mean = hidden.get("ref_mean_mag", None)
            prev_count = hidden.get("ref_count", None)

        if (prev_mean is None) or (prev_count is None) or (prev_mean.shape != ref_mag.shape):
            ref_count = torch.ones_like(ref_mag)
            ref_mean_mag = ref_mag
        else:
            ref_count = prev_count + 1.0
            ref_mean_mag = (prev_mean * prev_count + ref_mag) / torch.clamp(ref_count, min=1.0)
        ref_mean_mag = torch.clamp(ref_mean_mag, min=self._eps)

        left_n = left / ref_mean_mag
        right_n = right / ref_mean_mag
        feat = torch.stack([left_n.real, left_n.imag, right_n.real, right_n.imag], dim=-1)
        feat = self.sbeam_in_norm(feat)
        feat = self.sbeam_in_proj(feat)
        feat = self.sbeam_in_act(feat)
        feat = self.sbeam_dropout(feat)

        b, f, h = feat.shape
        lstm_state = None
        if isinstance(prev_lstm, tuple) and len(prev_lstm) == 2:
            prev_h, prev_c = prev_lstm
            if (
                isinstance(prev_h, torch.Tensor)
                and isinstance(prev_c, torch.Tensor)
                and prev_h.ndim == 4
                and prev_c.ndim == 4
                and prev_h.shape[0] == self.num_layers
                and prev_h.shape[1] == b
                and prev_h.shape[2] == f
            ):
                lstm_state = (
                    prev_h.reshape(self.num_layers, b * f, h).contiguous(),
                    prev_c.reshape(self.num_layers, b * f, h).contiguous(),
                )

        lstm_in = feat.reshape(b * f, 1, h)
        out, lstm_state_new = self._run_sbeam_lstm(lstm_in, lstm_state)
        out = out.squeeze(1).reshape(b, f, h)
        h_shared = self.sbeam_out_norm(out + feat)

        est_spec = self._decode_stereo_beam_lite_frame(
            h_shared,
            mix_spec_frame_stereo,
            azimuth_deg=azimuth_deg,
        )
        mag_denom = torch.clamp(ref_mag.unsqueeze(1), min=self._eps)
        pseudo_mask = torch.clamp(est_spec.abs().mean(dim=2) / mag_denom, min=0.0, max=10.0)

        new_hidden = {
            "arch": "stereo_beam_lite",
            "lstm_state": (
                lstm_state_new[0].reshape(self.num_layers, b, f, h).detach(),
                lstm_state_new[1].reshape(self.num_layers, b, f, h).detach(),
            ),
            "ref_mean_mag": ref_mean_mag.detach(),
            "ref_count": ref_count.detach(),
        }
        return est_spec, new_hidden, pseudo_mask

    @torch.no_grad()
    def forward_step(
        self,
        mix_spec_frame: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
        frame_time: Optional[torch.Tensor] = None,
        mix_spec_frame_stereo: Optional[torch.Tensor] = None,
        azimuth_deg: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        self._clear_auxiliary_outputs()
        if mix_spec_frame.ndim != 2:
            raise ValueError(f"Expected mix_spec_frame [B, F], got {tuple(mix_spec_frame.shape)}")

        if self.architecture == "online_spatialnet":
            return self._forward_step_online_spatialnet(
                mix_spec_frame=mix_spec_frame,
                hidden=hidden if isinstance(hidden, dict) else None,
                mix_spec_frame_stereo=mix_spec_frame_stereo,
            )
        if self.architecture == "stereo_lite":
            return self._forward_step_stereo_lite(
                mix_spec_frame=mix_spec_frame,
                hidden=hidden if isinstance(hidden, dict) else None,
                mix_spec_frame_stereo=mix_spec_frame_stereo,
                azimuth_deg=azimuth_deg,
            )
        if self.architecture == "stereo_stacked_lstm":
            return self._forward_step_stereo_stacked_lstm(
                mix_spec_frame=mix_spec_frame,
                hidden=hidden if isinstance(hidden, dict) else None,
                mix_spec_frame_stereo=mix_spec_frame_stereo,
                azimuth_deg=azimuth_deg,
            )
        if self.architecture == "stereo_beam_lite":
            return self._forward_step_stereo_beam_lite(
                mix_spec_frame=mix_spec_frame,
                hidden=hidden if isinstance(hidden, dict) else None,
                mix_spec_frame_stereo=mix_spec_frame_stereo,
                azimuth_deg=azimuth_deg,
            )

        input_mag_frame = self._build_input_mag_frame(
            mix_spec_frame,
            stereo_spec_frame=mix_spec_frame_stereo,
        )

        if self.architecture == "gru":
            return self._forward_step_gru(mix_spec_frame, hidden, input_mag_frame=input_mag_frame)
        if self.architecture == "tcn":
            return self._forward_step_tcn(mix_spec_frame, hidden, input_mag_frame=input_mag_frame)
        return self._forward_step_lstm_hybrid(
            mix_spec_frame,
            hidden,
            frame_time=frame_time,
            input_mag_frame=input_mag_frame,
        )

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
            "output_source_channels": self.output_source_channels,
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
            "use_learned_stereo_fusion": self.use_learned_stereo_fusion,
            "stereo_frontend_dim": self.stereo_frontend_dim,
            "stereo_frontend_layers": self.stereo_frontend_layers,
            "stereo_band_kernel_size": self.stereo_band_kernel_size,
            "use_azimuth_conditioning": self.use_azimuth_conditioning,
            "default_left_azimuth_deg": self.default_left_azimuth_deg,
            "default_right_azimuth_deg": self.default_right_azimuth_deg,
            "stereo_spatial_scale": float(torch.tanh(self.stereo_delta_scale).detach().cpu().item()),
            "stereo_gain_scale": float(torch.tanh(self.stereo_gain_scale).detach().cpu().item()),
            "osn_num_layers": self.osn_num_layers,
            "osn_dim_hidden": self.osn_dim_hidden,
            "osn_dim_ffn": self.osn_dim_ffn,
            "osn_dim_squeeze": self.osn_dim_squeeze,
            "osn_num_heads": self.osn_num_heads,
            "osn_encoder_kernel_size": self.osn_encoder_kernel_size,
            "osn_freq_kernel_size": self.osn_freq_kernel_size,
            "osn_time_kernel_size": self.osn_time_kernel_size,
            "osn_freq_conv_groups": self.osn_freq_conv_groups,
            "osn_time_conv_groups": self.osn_time_conv_groups,
            "osn_attention_scope": self.osn_attention_scope,
            "osn_attention": self.osn_attention,
            "osn_streaming_history": self.osn_streaming_history,
            "force_rnn_fp32": bool(self.force_rnn_fp32),
            "bottleneck_size": self.bottleneck_size,
            "tcn_hidden_size": self.tcn_hidden_size,
            "tcn_kernel_size": self.tcn_kernel_size,
            "tcn_blocks": self.tcn_blocks,
            "tcn_repeats": self.tcn_repeats,
        }


def build_separator_from_config(cfg: dict) -> LightweightCausalSeparator:
    kwargs = separator_kwargs_from_config(cfg or {})
    validate_model_hparams(kwargs)
    return LightweightCausalSeparator(**kwargs)


def adapt_separator_state_dict_for_model(
    model: LightweightCausalSeparator,
    state_dict: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    adapted = dict(state_dict)
    if str(getattr(model, "architecture", "")).lower() != "stereo_beam_lite":
        return adapted

    model_state = model.state_dict()

    def _resize_output_channel_blocks(
        tensor: torch.Tensor,
        num_blocks: int,
        src_out_channels: int,
        dst_out_channels: int,
    ) -> torch.Tensor:
        if src_out_channels == dst_out_channels:
            return tensor
        if tensor.shape[0] % num_blocks != 0:
            return tensor
        block_len = tensor.shape[0] // num_blocks
        if block_len % src_out_channels != 0:
            return tensor
        per_channel = block_len // src_out_channels
        reshaped = tensor.reshape(num_blocks, src_out_channels, per_channel, *tensor.shape[1:])
        if dst_out_channels == 1:
            resized = reshaped.mean(dim=1, keepdim=True)
        elif src_out_channels == 1:
            resized = reshaped.expand(-1, dst_out_channels, -1, *([-1] * (tensor.ndim - 1)))
        else:
            return tensor
        return resized.reshape(num_blocks * dst_out_channels * per_channel, *tensor.shape[1:])

    upgrades = (
        ("sbeam_filter_out_query.2.weight", 1),
        ("sbeam_filter_out_query.2.bias", 1),
        ("sbeam_filter_out.2.weight", int(getattr(model, "num_speakers", 2))),
        ("sbeam_filter_out.2.bias", int(getattr(model, "num_speakers", 2))),
    )
    for key, num_blocks in upgrades:
        if key not in adapted or key not in model_state:
            continue
        src = adapted[key]
        dst = model_state[key]
        if src.shape == dst.shape:
            continue
        if src.ndim < 1 or src.shape[1:] != dst.shape[1:]:
            continue
        if (src.shape[0] % num_blocks) != 0 or (dst.shape[0] % num_blocks) != 0:
            continue
        src_block_len = src.shape[0] // num_blocks
        dst_block_len = dst.shape[0] // num_blocks
        if src_block_len == dst_block_len:
            continue
        if src_block_len % 4 != 0 or dst_block_len % 4 != 0:
            continue
        src_out_channels = src_block_len // 4
        dst_out_channels = dst_block_len // 4
        resized = _resize_output_channel_blocks(
            src,
            num_blocks=num_blocks,
            src_out_channels=src_out_channels,
            dst_out_channels=dst_out_channels,
        )
        if resized.shape == dst.shape:
            adapted[key] = resized.to(device=src.device, dtype=src.dtype)

    return adapted


class AzimuthConditionedSelectionHead(nn.Module):
    """Select a direction-target source from frozen separated sources."""

    def __init__(self, num_speakers: int = 2, hidden_size: int = 128, dropout: float = 0.0):
        super().__init__()
        self.num_speakers = int(num_speakers)
        self.hidden_size = int(hidden_size)
        self.dropout = float(dropout)

        in_dim = 4  # [energy, corr_left, corr_right, ild]
        self.spk_proj = nn.Sequential(
            nn.Linear(in_dim, self.hidden_size),
            nn.SiLU(),
            nn.Dropout(self.dropout) if self.dropout > 0.0 else nn.Identity(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.SiLU(),
        )
        self.azimuth_proj = nn.Sequential(
            nn.Linear(2, self.hidden_size),  # [sin(az), cos(az)]
            nn.SiLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
        )
        self.score = nn.Linear(self.hidden_size, 1)
        self._eps = 1e-8

    def _normalize_mix_stereo(self, mix_stereo: torch.Tensor) -> torch.Tensor:
        if mix_stereo.ndim != 3:
            raise ValueError(f"Expected mix_stereo [B,2,T] or [B,T,2], got {tuple(mix_stereo.shape)}")
        if mix_stereo.shape[1] == 2:
            return mix_stereo
        if mix_stereo.shape[2] == 2:
            return mix_stereo.transpose(1, 2).contiguous()
        raise ValueError(f"Cannot infer stereo channel axis from shape {tuple(mix_stereo.shape)}")

    def forward(
        self,
        est_sources: torch.Tensor,
        mix_stereo: torch.Tensor,
        azimuth_deg: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            est_sources: [B, S, T] or [B, S, 2, T]
            mix_stereo: [B, 2, T] or [B, T, 2]
            azimuth_deg: [B] or [B,1] in degrees
        Returns:
            target: [B, T]
            weights: [B, S]
        """
        if est_sources.ndim == 4:
            est_sources = torch.mean(est_sources, dim=2)
        elif est_sources.ndim != 3:
            raise ValueError(f"Expected est_sources [B,S,T] or [B,S,2,T], got {tuple(est_sources.shape)}")
        mix_stereo = self._normalize_mix_stereo(mix_stereo)
        if est_sources.shape[0] != mix_stereo.shape[0]:
            raise ValueError("Batch size mismatch between est_sources and mix_stereo")
        if est_sources.shape[-1] != mix_stereo.shape[-1]:
            raise ValueError("Time length mismatch between est_sources and mix_stereo")

        left = mix_stereo[:, 0:1, :]
        right = mix_stereo[:, 1:2, :]

        energy = torch.mean(est_sources ** 2, dim=-1)
        corr_left = torch.mean(est_sources * left, dim=-1)
        corr_right = torch.mean(est_sources * right, dim=-1)

        left_e = torch.mean((est_sources * left) ** 2, dim=-1)
        right_e = torch.mean((est_sources * right) ** 2, dim=-1)
        ild = torch.log(torch.clamp(left_e, min=self._eps)) - torch.log(torch.clamp(right_e, min=self._eps))

        spk_feat = torch.stack([energy, corr_left, corr_right, ild], dim=-1)  # [B,S,4]
        spk_embed = self.spk_proj(spk_feat)

        azimuth_deg = azimuth_deg.reshape(-1).to(device=est_sources.device, dtype=est_sources.dtype)
        azimuth_rad = azimuth_deg * torch.pi / 180.0
        az_feat = torch.stack([torch.sin(azimuth_rad), torch.cos(azimuth_rad)], dim=-1)
        az_embed = self.azimuth_proj(az_feat).unsqueeze(1)  # [B,1,H]

        logits = self.score(torch.tanh(spk_embed + az_embed)).squeeze(-1)  # [B,S]
        weights = torch.softmax(logits, dim=1)
        target = torch.sum(weights.unsqueeze(-1) * est_sources, dim=1)
        return target, weights


class FrozenSeparatorDirectionalExtractor(nn.Module):
    """
    Freeze a pretrained separator, then optionally unfreeze a few front/back stages
    for stereo-domain adaptation while training an azimuth-conditioned selection head.
    """

    def __init__(
        self,
        separator: LightweightCausalSeparator,
        head_hidden_size: int = 128,
        head_dropout: float = 0.0,
        unfreeze_head_layers: int = 0,
        unfreeze_tail_layers: int = 0,
    ):
        super().__init__()
        self.separator = separator
        self.unfreeze_head_layers = max(0, int(unfreeze_head_layers))
        self.unfreeze_tail_layers = max(0, int(unfreeze_tail_layers))
        self._unfrozen_separator_stage_names: List[str] = []
        self._has_trainable_separator = False
        self._freeze_separator_all()
        self._apply_separator_partial_unfreeze()

        self.directional_head = AzimuthConditionedSelectionHead(
            num_speakers=int(separator.num_speakers),
            hidden_size=int(head_hidden_size),
            dropout=float(head_dropout),
        )

    def _freeze_separator_all(self):
        for p in self.separator.parameters():
            p.requires_grad = False
        self.separator.eval()

    def _separator_stages(self) -> List[Tuple[str, nn.Module]]:
        arch = str(getattr(self.separator, "architecture", "")).lower()
        stages: List[Tuple[str, nn.Module]] = []
        if arch == "lstm_hybrid":
            stages = [
                ("freq_encoder", self.separator.freq_encoder),
                ("time_encoder", self.separator.time_encoder),
                ("time_proj", self.separator.time_proj),
                ("fusion_proj", self.separator.fusion_proj),
                ("lstm", self.separator.lstm),
                ("lstm_post_blocks", self.separator.lstm_post_blocks),
                ("mask_head", self.separator.mask_head),
            ]
        elif arch == "tcn":
            stages = [
                ("in_norm", self.separator.in_norm),
                ("in_proj", self.separator.in_proj),
                ("tcn_stack", self.separator.tcn_stack),
                ("tcn_out", self.separator.tcn_out),
                ("mask_head", self.separator.mask_head),
            ]
        elif arch == "stereo_lite":
            stages = [
                ("slite_in_proj", getattr(self.separator, "slite_in_proj", None)),
                ("slite_gru", getattr(self.separator, "slite_gru", None)),
                ("slite_mask_out", getattr(self.separator, "slite_mask_out", None)),
                ("slite_mask_out_query", getattr(self.separator, "slite_mask_out_query", None)),
            ]
        elif arch == "stereo_stacked_lstm":
            stages = [
                ("slstm_in_proj", getattr(self.separator, "slstm_in_proj", None)),
                ("slstm_lstm", getattr(self.separator, "slstm_lstm", None)),
                ("slstm_mask_out", getattr(self.separator, "slstm_mask_out", None)),
                ("slstm_mask_out_query", getattr(self.separator, "slstm_mask_out_query", None)),
            ]
        elif arch == "stereo_beam_lite":
            stages = [
                ("sbeam_in_proj", getattr(self.separator, "sbeam_in_proj", None)),
                ("sbeam_lstm", getattr(self.separator, "sbeam_lstm", None)),
                ("sbeam_query_refine", getattr(self.separator, "sbeam_query_refine", None)),
                ("sbeam_query_competition", getattr(self.separator, "sbeam_query_competition", None)),
                ("sbeam_filter_out", getattr(self.separator, "sbeam_filter_out", None)),
                ("sbeam_filter_out_query", getattr(self.separator, "sbeam_filter_out_query", None)),
            ]
        elif arch == "online_spatialnet":
            stages = [("osn", getattr(self.separator, "osn", None))]
        else:
            stages = [
                ("in_proj", self.separator.in_proj),
                ("rnn", self.separator.rnn),
                ("post_blocks", self.separator.post_blocks),
                ("mask_head", self.separator.mask_head),
            ]
        return [(name, mod) for name, mod in stages if mod is not None]

    def _set_module_requires_grad(self, module: nn.Module, requires_grad: bool):
        for p in module.parameters():
            p.requires_grad = bool(requires_grad)

    def _apply_separator_partial_unfreeze(self):
        stages = self._separator_stages()
        if not stages:
            self._has_trainable_separator = False
            self._unfrozen_separator_stage_names = []
            return

        selected: Dict[str, nn.Module] = {}
        if self.unfreeze_head_layers > 0:
            n = min(self.unfreeze_head_layers, len(stages))
            for name, module in stages[:n]:
                selected[name] = module
        if self.unfreeze_tail_layers > 0:
            n = min(self.unfreeze_tail_layers, len(stages))
            for name, module in stages[-n:]:
                selected[name] = module

        for name, module in selected.items():
            self._set_module_requires_grad(module, True)

        self._unfrozen_separator_stage_names = list(selected.keys())
        self._has_trainable_separator = any(p.requires_grad for p in self.separator.parameters())

    def train(self, mode: bool = True):
        super().train(mode)
        # If part of separator is unfrozen, run separator in train mode; otherwise keep eval.
        if self._has_trainable_separator:
            self.separator.train(mode)
        else:
            self.separator.eval()
        return self

    def _normalize_mix_stereo(self, mix_stereo: torch.Tensor) -> torch.Tensor:
        if mix_stereo.ndim != 3:
            raise ValueError(f"Expected mix_stereo [B,2,T] or [B,T,2], got {tuple(mix_stereo.shape)}")
        if mix_stereo.shape[1] == 2:
            return mix_stereo
        if mix_stereo.shape[2] == 2:
            return mix_stereo.transpose(1, 2).contiguous()
        raise ValueError(f"Cannot infer stereo channel axis from shape {tuple(mix_stereo.shape)}")

    def forward(
        self,
        mix_stereo: torch.Tensor,
        azimuth_deg: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            mix_stereo: [B,2,T] or [B,T,2]
            azimuth_deg: [B] or [B,1]
        Returns:
            target: [B,T] selected directional source
            weights: [B,S] source selection weights
            est_sources: [B,S,T] or [B,S,2,T] separator outputs
        """
        mix_stereo = self._normalize_mix_stereo(mix_stereo)
        if self._has_trainable_separator:
            est_sources = self.separator(mix_stereo)
        else:
            with torch.no_grad():
                est_sources = self.separator(mix_stereo)
        est_sources_for_head = torch.mean(est_sources, dim=2) if est_sources.ndim == 4 else est_sources
        target, weights = self.directional_head(est_sources_for_head, mix_stereo, azimuth_deg)
        return target, weights, est_sources

    def head_config(self) -> dict:
        return {
            "num_speakers": int(self.directional_head.num_speakers),
            "hidden_size": int(self.directional_head.hidden_size),
            "dropout": float(self.directional_head.dropout),
            "unfreeze_head_layers": int(self.unfreeze_head_layers),
            "unfreeze_tail_layers": int(self.unfreeze_tail_layers),
            "unfrozen_separator_stages": list(self._unfrozen_separator_stage_names),
        }
