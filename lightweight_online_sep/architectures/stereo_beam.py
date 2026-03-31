"""Shared-trunk stereo beam-lite architecture helpers."""

import warnings
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


class StereoBeamLiteArchitectureMixin:
    def _init_stereo_beam_lite_modules(self):
        in_dim = 4
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
                nn.Linear(self.hidden_size, 4),
            )
            self.sbeam_query_temperature = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.sbeam_filter_out = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.SiLU(),
            nn.Linear(self.hidden_size, self.num_speakers * 4),
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
        stereo_x = stereo_spec.permute(0, 2, 3, 1).unsqueeze(1).contiguous()  # [B,1,F,T,2]
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
            if hasattr(self, "_aux_outputs") and isinstance(self._aux_outputs, dict):
                self._aux_outputs["query_assign"] = query_assign
        else:
            filter_ri = torch.tanh(self.sbeam_filter_out(h_shared))
            filter_ri = filter_ri.reshape(b, f, t, self.num_speakers, 4).permute(0, 3, 1, 2, 4).contiguous()

        weights = torch.view_as_complex(filter_ri.float().reshape(b, self.num_speakers, f, t, 2, 2))
        return (weights.to(dtype=stereo_x.dtype) * stereo_x).sum(dim=-1)

    def _decode_stereo_beam_lite_frame(
        self,
        h_shared: torch.Tensor,
        stereo_spec_frame: torch.Tensor,
        azimuth_deg: Optional[torch.Tensor],
    ) -> torch.Tensor:
        b, f, h = h_shared.shape
        stereo_x = stereo_spec_frame.transpose(1, 2).unsqueeze(1).contiguous()  # [B,1,F,2]
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
            if hasattr(self, "_aux_outputs") and isinstance(self._aux_outputs, dict):
                self._aux_outputs["query_assign"] = query_assign
        else:
            filter_ri = torch.tanh(self.sbeam_filter_out(h_shared))
            filter_ri = filter_ri.reshape(b, f, self.num_speakers, 4).permute(0, 2, 1, 3).contiguous()

        weights = torch.view_as_complex(filter_ri.float().reshape(b, self.num_speakers, f, 2, 2))
        return (weights.to(dtype=stereo_x.dtype) * stereo_x).sum(dim=-1)

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
        pseudo_mask = torch.clamp(est_spec.abs() / mag_denom, min=0.0, max=10.0)

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
