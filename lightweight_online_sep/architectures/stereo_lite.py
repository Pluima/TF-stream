"""Stereo-lite architecture helpers."""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


class StereoLiteArchitectureMixin:
    def _init_stereo_lite_modules(self):
        in_dim = 4
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
            self.slite_query_film = nn.Linear(self.hidden_size, self.hidden_size * 2)
            self.slite_mask_out_query = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.SiLU(),
                nn.Linear(self.hidden_size, 2),
            )
        self.slite_mask_out = nn.Linear(self.hidden_size, self.num_speakers * 2)

    def _run_slite_gru(self, feat: torch.Tensor, hidden: Optional[torch.Tensor]):
        return self._run_recurrent_module(self.slite_gru, feat, hidden)

    def _prepare_stereo_lite_features(
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
        feat = self.slite_in_norm(feat)
        feat = self.slite_in_proj(feat)
        feat = self.slite_in_act(feat)
        return self.slite_dropout(feat)

    def _slite_frequency_mix_sequence(self, feat: torch.Tensor) -> torch.Tensor:
        b, f, t, h = feat.shape
        x = feat.permute(0, 2, 3, 1).reshape(b * t, h, f).contiguous()
        x = self.slite_freq_dw(x)
        x = self.slite_freq_act(x)
        x = self.slite_freq_pw(x)
        x = x.reshape(b, t, h, f).permute(0, 3, 1, 2).contiguous()
        return self.slite_freq_norm(x + feat)

    def _slite_frequency_mix_frame(self, feat: torch.Tensor) -> torch.Tensor:
        x = feat.transpose(1, 2).contiguous()
        x = self.slite_freq_dw(x)
        x = self.slite_freq_act(x)
        x = self.slite_freq_pw(x)
        x = x.transpose(1, 2).contiguous()
        return self.slite_freq_norm(x + feat)

    def _forward_stereo_lite(
        self,
        mix_spec: torch.Tensor,
        mix_stereo_spec: Optional[torch.Tensor],
        azimuth_deg: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        feat = self._prepare_stereo_lite_features(mix_spec, mix_stereo_spec)
        feat = self._slite_frequency_mix_sequence(feat)

        b, f, t, h = feat.shape
        az_embed = self._azimuth_embedding(
            azimuth_deg=azimuth_deg,
            batch_size=b,
            device=feat.device,
            dtype=feat.dtype,
        )

        if az_embed is not None:
            q = az_embed.unsqueeze(2).unsqueeze(3)
            seq_in = feat.unsqueeze(1) + q
            seq_q = seq_in.reshape(b * self.num_speakers * f, t, h)
            out_q, _ = self._run_slite_gru(seq_q, None)
            out_q = out_q.reshape(b, self.num_speakers, f, t, h)
            out_q = self.slite_out_norm(out_q + seq_in)

            gamma_beta = self.slite_query_film(az_embed).unsqueeze(2).unsqueeze(3)
            gamma, beta = torch.chunk(gamma_beta, 2, dim=-1)
            out_q = out_q * (1.0 + 0.5 * torch.tanh(gamma)) + 0.5 * beta

            mask_ri = torch.tanh(self.slite_mask_out_query(out_q))
            mask_c = torch.view_as_complex(mask_ri.float().contiguous()).to(dtype=mix_spec.dtype)
        else:
            seq = feat.reshape(b * f, t, h)
            out, _ = self._run_slite_gru(seq, None)
            out = out.reshape(b, f, t, h)
            out = self.slite_out_norm(out + feat)
            mask_ri = torch.tanh(self.slite_mask_out(out))
            mask_c = torch.view_as_complex(
                mask_ri.float().reshape(b, f, t, self.num_speakers, 2)
            ).to(dtype=mix_spec.dtype)
            mask_c = mask_c.permute(0, 3, 1, 2).contiguous()

        return mask_c * mix_spec.unsqueeze(1)

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

        left = mix_spec_frame_stereo[:, 0]
        right = mix_spec_frame_stereo[:, 1]
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
        feat = torch.stack([left_n.real, left_n.imag, right_n.real, right_n.imag], dim=-1)
        feat = self.slite_in_norm(feat)
        feat = self.slite_in_proj(feat)
        feat = self.slite_in_act(feat)
        feat = self.slite_dropout(feat)
        feat = self._slite_frequency_mix_frame(feat)

        b, f, h = feat.shape
        az_embed = self._azimuth_embedding(
            azimuth_deg=azimuth_deg,
            batch_size=b,
            device=feat.device,
            dtype=feat.dtype,
        )

        if az_embed is not None:
            q = az_embed.unsqueeze(2)
            step_in = feat.unsqueeze(1) + q
            step_in_flat = step_in.reshape(b * self.num_speakers * f, 1, h)

            q_gru_state = None
            if isinstance(prev_gru, torch.Tensor) and prev_gru.ndim == 5:
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

            gamma_beta = self.slite_query_film(az_embed).unsqueeze(2)
            gamma, beta = torch.chunk(gamma_beta, 2, dim=-1)
            out_q = out_q * (1.0 + 0.5 * torch.tanh(gamma)) + 0.5 * beta

            mask_ri = torch.tanh(self.slite_mask_out_query(out_q))
            mask_c = torch.view_as_complex(mask_ri.float().contiguous()).to(dtype=mix_spec_frame.dtype)
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
            mask_ri = torch.tanh(self.slite_mask_out(out))
            mask_c = torch.view_as_complex(
                mask_ri.float().reshape(b, f, self.num_speakers, 2)
            ).to(dtype=mix_spec_frame.dtype)
            mask_c = mask_c.permute(0, 2, 1).contiguous()
            gru_state_pack = gru_state_new.reshape(self.num_layers, b, f, h).detach()

        est_spec = mask_c * mix_spec_frame.unsqueeze(1)
        pseudo_mask = mask_c.abs()
        new_hidden = {
            "arch": "stereo_lite",
            "gru_state": gru_state_pack,
            "ref_mean_mag": ref_mean_mag.detach(),
            "ref_count": ref_count.detach(),
        }
        return est_spec, new_hidden, pseudo_mask
