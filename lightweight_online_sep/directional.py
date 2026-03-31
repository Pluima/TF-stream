"""Directional selection head and frozen-separator wrapper."""

from typing import TYPE_CHECKING, Dict, List, Tuple

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from .lightweight_sep_model import LightweightCausalSeparator


class AzimuthConditionedSelectionHead(nn.Module):
    """Select a direction-target source from frozen separated sources."""

    def __init__(self, num_speakers: int = 2, hidden_size: int = 128, dropout: float = 0.0):
        super().__init__()
        self.num_speakers = int(num_speakers)
        self.hidden_size = int(hidden_size)
        self.dropout = float(dropout)

        in_dim = 4
        self.spk_proj = nn.Sequential(
            nn.Linear(in_dim, self.hidden_size),
            nn.SiLU(),
            nn.Dropout(self.dropout) if self.dropout > 0.0 else nn.Identity(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.SiLU(),
        )
        self.azimuth_proj = nn.Sequential(
            nn.Linear(2, self.hidden_size),
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
        if est_sources.ndim != 3:
            raise ValueError(f"Expected est_sources [B,S,T], got {tuple(est_sources.shape)}")
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

        spk_feat = torch.stack([energy, corr_left, corr_right, ild], dim=-1)
        spk_embed = self.spk_proj(spk_feat)

        azimuth_deg = azimuth_deg.reshape(-1).to(device=est_sources.device, dtype=est_sources.dtype)
        azimuth_rad = azimuth_deg * torch.pi / 180.0
        az_feat = torch.stack([torch.sin(azimuth_rad), torch.cos(azimuth_rad)], dim=-1)
        az_embed = self.azimuth_proj(az_feat).unsqueeze(1)

        logits = self.score(torch.tanh(spk_embed + az_embed)).squeeze(-1)
        weights = torch.softmax(logits, dim=1)
        target = torch.sum(weights.unsqueeze(-1) * est_sources, dim=1)
        return target, weights


class FrozenSeparatorDirectionalExtractor(nn.Module):
    """Freeze a pretrained separator and train a lightweight directional head."""

    def __init__(
        self,
        separator: "LightweightCausalSeparator",
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
        if arch == "lstm_hybrid":
            stages = [
                ("freq_encoder", getattr(self.separator, "freq_encoder", None)),
                ("time_encoder", getattr(self.separator, "time_encoder", None)),
                ("time_proj", getattr(self.separator, "time_proj", None)),
                ("fusion_proj", getattr(self.separator, "fusion_proj", None)),
                ("lstm", getattr(self.separator, "lstm", None)),
                ("lstm_post_blocks", getattr(self.separator, "lstm_post_blocks", None)),
                ("mask_head", getattr(self.separator, "mask_head", None)),
            ]
        elif arch == "tcn":
            stages = [
                ("in_norm", getattr(self.separator, "in_norm", None)),
                ("in_proj", getattr(self.separator, "in_proj", None)),
                ("tcn_stack", getattr(self.separator, "tcn_stack", None)),
                ("tcn_out", getattr(self.separator, "tcn_out", None)),
                ("mask_head", getattr(self.separator, "mask_head", None)),
            ]
        elif arch == "stereo_lite":
            stages = [
                ("slite_in_proj", getattr(self.separator, "slite_in_proj", None)),
                ("slite_gru", getattr(self.separator, "slite_gru", None)),
                ("slite_mask_out", getattr(self.separator, "slite_mask_out", None)),
            ]
        elif arch == "stereo_stacked_lstm":
            stages = [
                ("slstm_in_proj", getattr(self.separator, "slstm_in_proj", None)),
                ("slstm_lstm", getattr(self.separator, "slstm_lstm", None)),
                ("slstm_mask_out", getattr(self.separator, "slstm_mask_out", None)),
            ]
        elif arch == "stereo_beam_lite":
            stages = [
                ("sbeam_in_proj", getattr(self.separator, "sbeam_in_proj", None)),
                ("sbeam_lstm", getattr(self.separator, "sbeam_lstm", None)),
                ("sbeam_filter_out", getattr(self.separator, "sbeam_filter_out", None)),
                ("sbeam_filter_out_query", getattr(self.separator, "sbeam_filter_out_query", None)),
            ]
        elif arch == "online_spatialnet":
            stages = [("osn", getattr(self.separator, "osn", None))]
        else:
            stages = [
                ("in_proj", getattr(self.separator, "in_proj", None)),
                ("rnn", getattr(self.separator, "rnn", None)),
                ("post_blocks", getattr(self.separator, "post_blocks", None)),
                ("mask_head", getattr(self.separator, "mask_head", None)),
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
        mix_stereo = self._normalize_mix_stereo(mix_stereo)
        if self._has_trainable_separator:
            est_sources = self.separator(mix_stereo)
        else:
            with torch.no_grad():
                est_sources = self.separator(mix_stereo)
        target, weights = self.directional_head(est_sources, mix_stereo, azimuth_deg)
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
