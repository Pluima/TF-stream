"""GRU, TCN, and LSTM-hybrid architecture helpers."""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..blocks import CausalTCNBlock, ResidualFFNBlock


class ClassicArchitectureMixin:
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

        blocks = []
        for _ in range(max(1, self.tcn_repeats)):
            for b in range(max(1, self.tcn_blocks)):
                blocks.append(
                    CausalTCNBlock(
                        channels=self.bottleneck_size,
                        hidden_channels=self.tcn_hidden_size,
                        kernel_size=self.tcn_kernel_size,
                        dilation=2 ** b,
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
        time_proj_layers = [
            nn.LayerNorm(self.time_encoder_dim),
            nn.Linear(self.time_encoder_dim, self.hidden_size),
            nn.SiLU(),
        ]
        if self.dropout > 0.0:
            time_proj_layers.append(nn.Dropout(self.dropout))
        self.time_proj = nn.Sequential(*time_proj_layers)

        fusion_layers = [
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

    def _run_gru(self, feat: torch.Tensor, hidden: Optional[torch.Tensor]):
        return self._run_recurrent_module(self.rnn, feat, hidden)

    def _run_lstm(
        self,
        feat: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        return self._run_recurrent_module(self.lstm, feat, state)

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

        prev_caches = [] if not isinstance(hidden, dict) else hidden.get("caches", [])
        new_caches = []
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

        prev_state = None if not isinstance(hidden, dict) else hidden.get("lstm", None)
        lstm_out, lstm_state = self._run_lstm(fused, prev_state)
        lstm_out = self._apply_blocks(lstm_out, self.lstm_post_blocks)

        masks = torch.sigmoid(self.mask_head(lstm_out.squeeze(1)))
        b, _ = masks.shape
        masks = masks.view(b, self.num_speakers, self.freq_bins)
        est_spec = masks * mix_spec_frame.unsqueeze(1)
        return est_spec, {"lstm": lstm_state}, masks
