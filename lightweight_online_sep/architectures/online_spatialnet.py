"""OnlineSpatialNet architecture helpers and dependency patching."""

import os
import sys
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_NBSS_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", "..", "..", "..", "NBSS"))
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


class OnlineSpatialNetArchitectureMixin:
    def _init_online_spatialnet_modules(self):
        if NBSSOnlineSpatialNet is None:
            detail = "" if _NBSS_IMPORT_ERROR is None else (
                f" | import error: {type(_NBSS_IMPORT_ERROR).__name__}: {_NBSS_IMPORT_ERROR}"
            )
            raise RuntimeError(
                "OnlineSpatialNet dependency is unavailable. "
                f"Expected NBSS package under: {_NBSS_ROOT}{detail}"
            )

        attn_scope = max(1, int(self.osn_attention_scope))
        attention = str(self.osn_attention).strip().lower()
        if attention in {"", "mhsa"}:
            attention = f"mhsa({attn_scope})"
        self.osn = NBSSOnlineSpatialNet(
            dim_input=4,
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

    def _prepare_osn_features(
        self,
        mix_spec: torch.Tensor,
        mix_stereo_spec: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        stereo_spec = self._normalize_osn_stereo_spec(mix_stereo_spec, mix_spec)
        stereo_bftc = stereo_spec.permute(0, 2, 3, 1).contiguous()
        ref = stereo_bftc[..., 0]
        ref_mean_mag = torch.clamp(ref.abs().mean(dim=2, keepdim=True), min=self._eps)
        stereo_norm = stereo_bftc / ref_mean_mag.unsqueeze(-1)
        feats = torch.view_as_real(stereo_norm).reshape(stereo_norm.shape[0], self.freq_bins, stereo_norm.shape[2], -1)
        return feats, ref_mean_mag

    def _forward_online_spatialnet(
        self,
        mix_spec: torch.Tensor,
        mix_stereo_spec: Optional[torch.Tensor],
    ) -> torch.Tensor:
        feats, ref_mean_mag = self._prepare_osn_features(mix_spec, mix_stereo_spec)
        out = self.osn(feats, inference=False)
        b, f, t, _ = out.shape
        out_c = torch.view_as_complex(out.float().reshape(b, f, t, self.num_speakers, 2))
        out_c = out_c * ref_mean_mag.unsqueeze(-1)
        return out_c.permute(0, 3, 1, 2).contiguous()

    @torch.no_grad()
    def _forward_step_online_spatialnet_windowed(
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
        ref_mag = torch.clamp(mix_spec_frame_stereo[:, 0, :].abs(), min=self._eps)

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

        stereo_norm = mix_spec_frame_stereo / ref_mean_mag.unsqueeze(1)
        frame_feat = torch.view_as_real(stereo_norm.transpose(1, 2).contiguous()).reshape(b, self.freq_bins, -1)

        if feat_hist is None:
            feat_seq = frame_feat.unsqueeze(2)
        else:
            feat_seq = torch.cat([feat_hist, frame_feat.unsqueeze(2)], dim=2)
        max_hist = max(4, int(self.osn_streaming_history))
        if feat_seq.shape[2] > max_hist:
            feat_seq = feat_seq[:, :, -max_hist:, :].contiguous()

        out_seq = self.osn(feat_seq, inference=False)
        out_last = out_seq[:, :, -1, :]
        out_c = torch.view_as_complex(out_last.float().reshape(b, self.freq_bins, self.num_speakers, 2))
        out_c = out_c * ref_mean_mag.unsqueeze(-1)
        est_spec = out_c.permute(0, 2, 1).contiguous()

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
        ref_mag = torch.clamp(mix_spec_frame_stereo[:, 0, :].abs(), min=self._eps)

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

        stereo_norm = mix_spec_frame_stereo / ref_mean_mag.unsqueeze(1)
        frame_feat = torch.view_as_real(stereo_norm.transpose(1, 2).contiguous()).reshape(b, self.freq_bins, -1)

        x = frame_feat.reshape(b * self.freq_bins, 1, -1).permute(0, 2, 1).contiguous()
        x = self.osn.encoder(x, state=osn_state["encoder"]).permute(0, 2, 1).contiguous()
        hdim = x.shape[-1]
        x = x.reshape(b, self.freq_bins, 1, hdim)

        layer_states = osn_state.get("layers", [])
        if len(layer_states) != len(self.osn.layers):
            layer_states = [dict() for _ in range(len(self.osn.layers))]
            osn_state["layers"] = layer_states

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

        out = self.osn.decoder(x)
        out_last = out[:, :, 0, :]
        out_c = torch.view_as_complex(out_last.float().reshape(b, self.freq_bins, self.num_speakers, 2))
        out_c = out_c * ref_mean_mag.unsqueeze(-1)
        est_spec = out_c.permute(0, 2, 1).contiguous()

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
        return self._forward_step_online_spatialnet_windowed(
            mix_spec_frame=mix_spec_frame,
            hidden=hidden,
            mix_spec_frame_stereo=mix_spec_frame_stereo,
        )
