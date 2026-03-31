"""Shared frontend, feature, and recurrent helpers."""

import warnings
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn


class SeparatorSharedMixin:
    def _init_stereo_frontend(self):
        layers = []
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
        nn.init.zeros_(self.stereo_frontend_out.weight)
        nn.init.zeros_(self.stereo_frontend_out.bias)
        self.stereo_delta_scale = nn.Parameter(torch.tensor(0.10, dtype=torch.float32))
        self.stereo_gain_scale = nn.Parameter(torch.tensor(0.10, dtype=torch.float32))

    def _init_stereo_frontend_no_conv(self):
        self.stereo_frontend = nn.Identity()
        self.stereo_frontend_out = nn.Identity()
        self.stereo_delta_scale = nn.Parameter(torch.tensor(0.0, dtype=torch.float32), requires_grad=False)
        self.stereo_gain_scale = nn.Parameter(torch.tensor(0.0, dtype=torch.float32), requires_grad=False)

    def _run_stereo_frontend(self, feats: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b, t, f, c = feats.shape
        x = feats.permute(0, 1, 3, 2).reshape(b * t, c, f).contiguous()
        h = self.stereo_frontend(x)
        out = self.stereo_frontend_out(h)
        out = out.reshape(b, t, 2, f).permute(0, 1, 3, 2).contiguous()
        return out[..., 0], out[..., 1]

    def _run_stereo_frontend_frame(self, feats: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = feats.permute(0, 2, 1).contiguous()
        h = self.stereo_frontend(x)
        out = self.stereo_frontend_out(h)
        return out[:, 0, :], out[:, 1, :]

    def _build_mask_head(self, in_dim: int) -> nn.Module:
        out_dim = self.num_speakers * self.freq_bins
        hidden_dim = self.mask_hidden_size if self.mask_hidden_size > 0 else in_dim
        if self.mask_head_layers <= 1:
            return nn.Linear(in_dim, out_dim)

        layers = []
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
        layers = [
            nn.LayerNorm(self.freq_bins),
            nn.Linear(self.freq_bins, out_dim),
            nn.SiLU(),
        ]
        if self.dropout > 0.0:
            layers.append(nn.Dropout(self.dropout))
        return nn.Sequential(*layers)

    def _normalize_osn_stereo_spec(
        self,
        mix_stereo_spec: Optional[torch.Tensor],
        mix_spec: torch.Tensor,
    ) -> torch.Tensor:
        if mix_stereo_spec is None:
            return torch.stack([mix_spec, mix_spec], dim=1)
        if mix_stereo_spec.ndim != 4:
            raise ValueError(f"Expected mix_stereo_spec [B,2,F,T], got {tuple(mix_stereo_spec.shape)}")
        if mix_stereo_spec.shape[1] == 2 and mix_stereo_spec.shape[2] == self.freq_bins:
            return mix_stereo_spec
        if mix_stereo_spec.shape[2] == 2 and mix_stereo_spec.shape[1] == self.freq_bins:
            return mix_stereo_spec.permute(0, 2, 1, 3).contiguous()
        raise ValueError(f"Cannot infer stereo axis for shape {tuple(mix_stereo_spec.shape)}")

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
        prompt_mode = getattr(self, "azimuth_prompt_mode", "degrees")
        if prompt_mode == "left_right":
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
        if azimuth_deg is None:
            return self._default_azimuth_prompt(batch_size, device=device, dtype=dtype)

        az = torch.as_tensor(azimuth_deg, device=device, dtype=dtype)
        prompt_mode = getattr(self, "azimuth_prompt_mode", "degrees")
        if az.ndim == 1:
            if az.shape[0] != self.num_speakers:
                raise ValueError(
                    f"azimuth_deg shape mismatch, expected [{self.num_speakers}] or [B,{self.num_speakers}], got {tuple(az.shape)}"
                )
            az = az.unsqueeze(0).expand(batch_size, -1)
            if prompt_mode == "left_right":
                return self._azimuth_deg_to_left_right_prompt(az).contiguous()
            az_rad = az * (torch.pi / 180.0)
            return torch.stack([torch.sin(az_rad), torch.cos(az_rad)], dim=-1)

        if az.ndim == 2:
            if az.shape[0] == batch_size and az.shape[1] == self.num_speakers:
                if prompt_mode == "left_right":
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
        )
        return self.azimuth_proj(az_sincos)

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
        if stereo_spec_frame.ndim != 3:
            raise ValueError(
                f"Expected stereo_spec_frame [B,2,F] or [B,F,2], got {tuple(stereo_spec_frame.shape)}"
            )
        if stereo_spec_frame.shape[1] == 2 and stereo_spec_frame.shape[2] == self.freq_bins:
            return stereo_spec_frame
        if stereo_spec_frame.shape[2] == 2 and stereo_spec_frame.shape[1] == self.freq_bins:
            return stereo_spec_frame.transpose(1, 2).contiguous()
        raise ValueError(f"Cannot infer stereo channel axis for frame spec shape {tuple(stereo_spec_frame.shape)}")

    def _build_stereo_features(
        self,
        mix_spec: torch.Tensor,
        stereo_spec: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        mono_log = torch.log1p(mix_spec.abs()).transpose(1, 2)
        if stereo_spec is None:
            zeros = torch.zeros_like(mono_log)
            return torch.stack([mono_log, mono_log, mono_log, zeros, zeros, zeros, zeros], dim=-1)

        left = stereo_spec[:, 0]
        right = stereo_spec[:, 1]
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
        mono_log = torch.log1p(mix_spec_frame.abs())
        if stereo_spec_frame is None:
            zeros = torch.zeros_like(mono_log)
            return torch.stack([mono_log, mono_log, mono_log, zeros, zeros, zeros, zeros], dim=-1)

        stereo_spec_frame = self._normalize_stereo_spec_frame(stereo_spec_frame)
        left = stereo_spec_frame[:, 0]
        right = stereo_spec_frame[:, 1]

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
        feats = self._build_stereo_features(mix_spec, stereo_spec=stereo_spec)
        base_mag = feats[..., 0]
        if not self.use_learned_stereo_fusion:
            return base_mag

        delta_logits, gain_logits = self._run_stereo_frontend(feats)
        delta = torch.tanh(delta_logits)
        gain_centered = torch.sigmoid(gain_logits) * 2.0 - 1.0
        delta_scale = torch.tanh(self.stereo_delta_scale).to(dtype=delta.dtype, device=delta.device)
        gain_scale = torch.tanh(self.stereo_gain_scale).to(dtype=delta.dtype, device=delta.device)
        gain = 1.0 + 0.5 * gain_scale * gain_centered
        return base_mag * gain + delta_scale * delta

    def _build_input_mag_frame(
        self,
        mix_spec_frame: torch.Tensor,
        stereo_spec_frame: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        feats = self._build_stereo_features_frame(mix_spec_frame, stereo_spec_frame=stereo_spec_frame)
        base_mag = feats[..., 0]
        if not self.use_learned_stereo_fusion:
            return base_mag

        delta_logits, gain_logits = self._run_stereo_frontend_frame(feats)
        delta = torch.tanh(delta_logits)
        gain_centered = torch.sigmoid(gain_logits) * 2.0 - 1.0
        delta_scale = torch.tanh(self.stereo_delta_scale).to(dtype=delta.dtype, device=delta.device)
        gain_scale = torch.tanh(self.stereo_gain_scale).to(dtype=delta.dtype, device=delta.device)
        gain = 1.0 + 0.5 * gain_scale * gain_centered
        return base_mag * gain + delta_scale * delta

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

    def _run_recurrent_module(
        self,
        module: nn.Module,
        feat: torch.Tensor,
        state: Optional[Any],
    ):
        use_fp32 = (
            feat.device.type == "cuda"
            and torch.is_autocast_enabled()
            and (self.force_rnn_fp32 or self._runtime_disable_bf16_rnn)
        )
        state_fp32 = self._recurrent_state_to_fp32(state)
        if use_fp32:
            with torch.cuda.amp.autocast(enabled=False):
                out_fp32, state_out = module(feat.float(), state_fp32)
            return out_fp32.to(feat.dtype), state_out
        try:
            return module(feat, state)
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
            with torch.cuda.amp.autocast(enabled=False):
                out_fp32, state_out = module(feat.float(), state_fp32)
            return out_fp32.to(feat.dtype), state_out

    def _recurrent_state_to_fp32(self, state: Optional[Any]):
        if state is None:
            return None
        if isinstance(state, tuple):
            return tuple(x.float() for x in state)
        if torch.is_tensor(state):
            return state.float()
        return state

    def _apply_blocks(self, feat: torch.Tensor, blocks: nn.ModuleList) -> torch.Tensor:
        for block in blocks:
            feat = block(feat)
        return feat
