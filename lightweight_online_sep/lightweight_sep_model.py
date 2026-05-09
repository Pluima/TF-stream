from typing import Optional

import torch
import torch.nn as nn

from .config import separator_kwargs_from_config, validate_model_hparams


class LightweightCausalSeparator(nn.Module):
    """Runtime-only stereo_beam_lite separator for 2 speakers with azimuth conditioning."""

    num_speakers = 2

    def __init__(
        self,
        n_fft: int = 512,
        hop_length: int = 128,
        win_length: int = 512,
        dropout: float = 0.08,
        hidden_size: int = 128,
        num_layers: int = 3,
    ):
        super().__init__()
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.win_length = int(win_length)
        self.dropout = float(dropout)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.freq_bins = self.n_fft // 2 + 1
        self._eps = 1e-8

        self.register_buffer(
            "analysis_window",
            torch.hamming_window(self.win_length),
            persistent=False,
        )
        self._init_modules()

    def _init_modules(self) -> None:
        self.sbeam_in_norm = nn.LayerNorm(4)
        self.sbeam_in_proj = nn.Linear(4, self.hidden_size)
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

    @staticmethod
    def _azimuth_deg_to_left_right_prompt(azimuth_deg: torch.Tensor) -> torch.Tensor:
        left = (azimuth_deg < 0.0).to(dtype=azimuth_deg.dtype)
        return torch.stack([left, 1.0 - left], dim=-1)

    def _resolve_azimuth_prompt(
        self,
        azimuth_deg: torch.Tensor,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        azimuth = torch.as_tensor(azimuth_deg, device=device, dtype=dtype)
        if azimuth.ndim == 1:
            azimuth = azimuth.unsqueeze(0).expand(batch_size, -1)
        if azimuth.shape != (batch_size, self.num_speakers):
            raise ValueError(
                f"Expected azimuth_deg shape [{batch_size},{self.num_speakers}], "
                f"got {tuple(azimuth.shape)}"
            )
        return self._azimuth_deg_to_left_right_prompt(azimuth).contiguous()

    def _azimuth_embedding(
        self,
        azimuth_deg: torch.Tensor,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        prompt = self._resolve_azimuth_prompt(azimuth_deg, batch_size, device, dtype)
        return self.azimuth_proj(prompt)

    def _validate_stereo_spec_frame(self, stereo_spec_frame: torch.Tensor) -> torch.Tensor:
        if stereo_spec_frame.ndim != 3 or stereo_spec_frame.shape[1:] != (self.num_speakers, self.freq_bins):
            raise ValueError(
                f"Expected stereo_spec_frame [B,2,{self.freq_bins}], "
                f"got {tuple(stereo_spec_frame.shape)}"
            )
        return stereo_spec_frame

    def _decode_frame(
        self,
        h_shared: torch.Tensor,
        stereo_spec_frame: torch.Tensor,
        azimuth_deg: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, freq_bins, _ = h_shared.shape
        stereo_x = stereo_spec_frame.transpose(1, 2).unsqueeze(1).contiguous()
        az_embed = self._azimuth_embedding(
            azimuth_deg=azimuth_deg,
            batch_size=batch_size,
            device=h_shared.device,
            dtype=h_shared.dtype,
        )

        shared_query = h_shared.unsqueeze(1).expand(-1, self.num_speakers, -1, -1)
        gamma_beta = self.sbeam_query_film(az_embed).unsqueeze(2)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=-1)
        shared_query = shared_query * (1.0 + 0.5 * torch.tanh(gamma)) + 0.5 * beta

        query = az_embed.unsqueeze(2).expand(-1, -1, freq_bins, -1)
        query_feat = self.sbeam_query_refine(torch.cat([shared_query, query], dim=-1))
        comp_logits = self.sbeam_query_competition(query_feat).squeeze(-1)
        temperature = torch.clamp(self.sbeam_query_temperature, min=0.25, max=4.0)
        temperature = temperature.to(device=h_shared.device, dtype=h_shared.dtype)
        query_assign = torch.softmax(comp_logits / temperature, dim=1)
        gate = torch.clamp(
            query_assign * float(self.num_speakers),
            min=float(self.sbeam_query_gate_floor),
            max=float(self.num_speakers),
        )
        filter_ri = torch.tanh(self.sbeam_filter_out_query(query_feat)) * gate.unsqueeze(-1)

        weights = torch.view_as_complex(
            filter_ri.float()
            .reshape(batch_size, self.num_speakers, freq_bins, 2, 2)
            .contiguous()
        )
        return (weights.to(dtype=stereo_x.dtype) * stereo_x).sum(dim=-1)

    @torch.no_grad()
    def forward_step(
        self,
        mix_spec_frame_stereo: torch.Tensor,
        azimuth_deg: torch.Tensor,
        hidden: Optional[dict] = None,
    ) -> tuple[torch.Tensor, dict]:
        if azimuth_deg is None:
            raise ValueError("azimuth_deg is required for this runtime separator")
        mix_spec_frame_stereo = self._validate_stereo_spec_frame(mix_spec_frame_stereo)

        left = mix_spec_frame_stereo[:, 0]
        right = mix_spec_frame_stereo[:, 1]
        ref_mag = torch.clamp(0.5 * (left.abs() + right.abs()), min=self._eps)

        if hidden is None:
            ref_count = torch.ones_like(ref_mag)
            ref_mean_mag = ref_mag
            lstm_state = None
        else:
            prev_count = hidden["ref_count"]
            prev_mean = hidden["ref_mean_mag"]
            ref_count = prev_count + 1.0
            ref_mean_mag = (prev_mean * prev_count + ref_mag) / torch.clamp(ref_count, min=1.0)
            prev_h, prev_c = hidden["lstm_state"]
            lstm_state = None

        ref_mean_mag = torch.clamp(ref_mean_mag, min=self._eps)
        left_norm = left / ref_mean_mag
        right_norm = right / ref_mean_mag
        feat = torch.stack([left_norm.real, left_norm.imag, right_norm.real, right_norm.imag], dim=-1)
        feat = self.sbeam_in_norm(feat)
        feat = self.sbeam_in_proj(feat)
        feat = self.sbeam_in_act(feat)
        feat = self.sbeam_dropout(feat)

        batch_size, freq_bins, hidden_size = feat.shape
        if hidden is not None:
            lstm_state = (
                prev_h.reshape(self.num_layers, batch_size * freq_bins, hidden_size).contiguous(),
                prev_c.reshape(self.num_layers, batch_size * freq_bins, hidden_size).contiguous(),
            )

        lstm_input = feat.reshape(batch_size * freq_bins, 1, hidden_size)
        out, lstm_state_new = self.sbeam_lstm(lstm_input, lstm_state)
        out = out.squeeze(1).reshape(batch_size, freq_bins, hidden_size)
        h_shared = self.sbeam_out_norm(out + feat)

        est_spec = self._decode_frame(
            h_shared,
            mix_spec_frame_stereo,
            azimuth_deg=azimuth_deg,
        )
        new_hidden = {
            "lstm_state": (
                lstm_state_new[0].reshape(self.num_layers, batch_size, freq_bins, hidden_size).detach(),
                lstm_state_new[1].reshape(self.num_layers, batch_size, freq_bins, hidden_size).detach(),
            ),
            "ref_mean_mag": ref_mean_mag.detach(),
            "ref_count": ref_count.detach(),
        }
        return est_spec, new_hidden


def build_separator_from_config(cfg: dict) -> LightweightCausalSeparator:
    kwargs = separator_kwargs_from_config(cfg or {})
    validate_model_hparams(kwargs)
    return LightweightCausalSeparator(**kwargs)
