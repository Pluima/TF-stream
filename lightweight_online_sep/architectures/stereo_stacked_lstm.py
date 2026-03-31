"""Pure stacked-LSTM stereo architecture helpers."""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


class StereoStackedLSTMArchitectureMixin:
    def _init_stereo_stacked_lstm_modules(self):
        in_dim = 4
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

    def _run_slstm(
        self,
        feat: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        return self._run_recurrent_module(self.slstm_lstm, feat, state)

    def _prepare_stereo_stacked_lstm_features(
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
        feat = self.slstm_in_norm(feat)
        feat = self.slstm_in_proj(feat)
        feat = self.slstm_in_act(feat)
        return self.slstm_dropout(feat)

    def _forward_stereo_stacked_lstm(
        self,
        mix_spec: torch.Tensor,
        mix_stereo_spec: Optional[torch.Tensor],
        azimuth_deg: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        feat = self._prepare_stereo_stacked_lstm_features(mix_spec, mix_stereo_spec)
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
            out_q, _ = self._run_slstm(seq_q, None)
            out_q = out_q.reshape(b, self.num_speakers, f, t, h)
            out_q = self.slstm_out_norm(out_q + seq_in)

            gamma_beta = self.slstm_query_film(az_embed).unsqueeze(2).unsqueeze(3)
            gamma, beta = torch.chunk(gamma_beta, 2, dim=-1)
            out_q = out_q * (1.0 + 0.5 * torch.tanh(gamma)) + 0.5 * beta

            mask_ri = torch.tanh(self.slstm_mask_out_query(out_q))
            mask_c = torch.view_as_complex(mask_ri.float().contiguous()).to(dtype=mix_spec.dtype)
        else:
            seq = feat.reshape(b * f, t, h)
            out, _ = self._run_slstm(seq, None)
            out = out.reshape(b, f, t, h)
            out = self.slstm_out_norm(out + feat)
            mask_ri = torch.tanh(self.slstm_mask_out(out))
            mask_c = torch.view_as_complex(
                mask_ri.float().reshape(b, f, t, self.num_speakers, 2)
            ).to(dtype=mix_spec.dtype)
            mask_c = mask_c.permute(0, 3, 1, 2).contiguous()

        return mask_c * mix_spec.unsqueeze(1)

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

        left = mix_spec_frame_stereo[:, 0]
        right = mix_spec_frame_stereo[:, 1]
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
        feat = torch.stack([left_n.real, left_n.imag, right_n.real, right_n.imag], dim=-1)
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
        )

        if az_embed is not None:
            q = az_embed.unsqueeze(2)
            step_in = feat.unsqueeze(1) + q
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

            gamma_beta = self.slstm_query_film(az_embed).unsqueeze(2)
            gamma, beta = torch.chunk(gamma_beta, 2, dim=-1)
            out_q = out_q * (1.0 + 0.5 * torch.tanh(gamma)) + 0.5 * beta

            mask_ri = torch.tanh(self.slstm_mask_out_query(out_q))
            mask_c = torch.view_as_complex(mask_ri.float().contiguous()).to(dtype=mix_spec_frame.dtype)
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
            mask_ri = torch.tanh(self.slstm_mask_out(out))
            mask_c = torch.view_as_complex(
                mask_ri.float().reshape(b, f, self.num_speakers, 2)
            ).to(dtype=mix_spec_frame.dtype)
            mask_c = mask_c.permute(0, 2, 1).contiguous()
            lstm_state_pack = (
                lstm_state_new[0].reshape(self.num_layers, b, f, h).detach(),
                lstm_state_new[1].reshape(self.num_layers, b, f, h).detach(),
            )

        est_spec = mask_c * mix_spec_frame.unsqueeze(1)
        pseudo_mask = mask_c.abs()
        new_hidden = {
            "arch": "stereo_stacked_lstm",
            "lstm_state": lstm_state_pack,
            "ref_mean_mag": ref_mean_mag.detach(),
            "ref_count": ref_count.detach(),
        }
        return est_spec, new_hidden, pseudo_mask
