from typing import Any, Optional

import numpy as np
import torch


class OnlineSeparatorStreamer:
    """Stateful low-latency streaming wrapper for LightweightCausalSeparator."""

    def __init__(
        self,
        model: torch.nn.Module,
        device: str = "cpu",
        target_azimuth_deg: Any = (15.0, -15.0),
    ):
        self.model = model.eval()
        self.device = torch.device(device)
        self.model.to(self.device)

        self.n_fft = int(model.n_fft)
        self.hop_length = int(model.hop_length)
        self.win_length = int(model.win_length)
        self.num_speakers = int(model.num_speakers)

        self.window = model.analysis_window.to(self.device)
        self.window_sq = (self.window ** 2)
        az = np.asarray(target_azimuth_deg, dtype=np.float32).reshape(-1)
        if az.shape[0] != self.num_speakers:
            raise ValueError(
                f"target_azimuth_deg must have {self.num_speakers} values, got {az.shape[0]}"
            )
        self.target_azimuth_deg = torch.as_tensor(az, device=self.device, dtype=torch.float32).unsqueeze(0)

        self._eps = 1e-8
        self.reset()

    def reset(self):
        self.hidden: Optional[Any] = None
        self.analysis_buffer = torch.zeros(2, self.win_length - self.hop_length, device=self.device)
        self.ola_num = torch.zeros(self.num_speakers, self.win_length, device=self.device)
        self.ola_den = torch.zeros(self.win_length, device=self.device)
        self.output_buffer = torch.zeros(self.num_speakers, 0, device=self.device)

    @torch.no_grad()
    def _emit_ready_frames(self) -> torch.Tensor:
        '''
        get audio input -> rfft-> model process -> irfft
        '''
        emitted = []

        while self.analysis_buffer.shape[1] >= self.win_length:
            frame_stereo = self.analysis_buffer[:, : self.win_length]  # [2, W]
            self.analysis_buffer = self.analysis_buffer[:, self.hop_length :]

            frame_win_stereo = frame_stereo * self.window.unsqueeze(0)
            mix_spec_stereo = torch.fft.rfft(frame_win_stereo, n=self.n_fft).unsqueeze(0)  # [1, 2, F]
            est_spec, self.hidden = self.model.forward_step(
                mix_spec_frame_stereo=mix_spec_stereo,
                hidden=self.hidden,
                azimuth_deg=self.target_azimuth_deg,
            )

            est_spec = est_spec.squeeze(0)
            if est_spec.ndim != 2:
                raise RuntimeError(f"Unsupported forward_step output shape: {tuple(est_spec.shape)}")
            est_frame = torch.fft.irfft(est_spec, n=self.n_fft)[:, : self.win_length]
            est_frame = est_frame * self.window.unsqueeze(0)

            self.ola_num = self.ola_num + est_frame
            self.ola_den = self.ola_den + self.window_sq

            out_num = self.ola_num[:, : self.hop_length]
            out_den = torch.clamp(self.ola_den[: self.hop_length], min=self._eps).unsqueeze(0)
            emitted.append(out_num / out_den)

            self.ola_num = torch.cat(
                [
                    self.ola_num[:, self.hop_length :],
                    torch.zeros(self.num_speakers, self.hop_length, device=self.device),
                ],
                dim=1,
            )
            self.ola_den = torch.cat(
                [
                    self.ola_den[self.hop_length :],
                    torch.zeros(self.hop_length, device=self.device),
                ],
                dim=0,
            )

        if not emitted:
            return torch.zeros(self.num_speakers, 0, device=self.device)
        return torch.cat(emitted, dim=1)

    @torch.no_grad()
    def process_chunk(self, chunk: np.ndarray) -> np.ndarray:
        """
        Args:
            chunk: stereo chunk with shape [T, 2]
        Returns:
            separated chunk [S, T]
        """
        chunk_2ch = np.asarray(chunk, dtype=np.float32)
        if chunk_2ch.ndim != 2 or chunk_2ch.shape[1] != 2:
            raise ValueError(f"Expected stereo chunk [T,2], got {tuple(chunk_2ch.shape)}")
        chunk_2ch = np.ascontiguousarray(chunk_2ch.T)
        chunk_tensor = torch.as_tensor(chunk_2ch, dtype=torch.float32, device=self.device)
        in_len = int(chunk_tensor.shape[1])

        self.analysis_buffer = torch.cat([self.analysis_buffer, chunk_tensor], dim=1)
        new_out = self._emit_ready_frames()
        if new_out.numel() > 0:
            self.output_buffer = torch.cat([self.output_buffer, new_out], dim=1)

        if self.output_buffer.shape[1] < in_len:
            pad_len = in_len - self.output_buffer.shape[1]
            pad = torch.zeros(self.num_speakers, pad_len, device=self.device)
            self.output_buffer = torch.cat([self.output_buffer, pad], dim=1)

        out = self.output_buffer[:, :in_len]
        self.output_buffer = self.output_buffer[:, in_len:]
        return out.detach().cpu().numpy()

    @torch.no_grad()
    def flush(self) -> np.ndarray:
        """Flush residual states by feeding zero paddings once."""
        tail = np.zeros((self.win_length, 2), dtype=np.float32)
        _ = self.process_chunk(tail)
        if self.output_buffer.shape[1] == 0:
            return np.zeros((self.num_speakers, 0), dtype=np.float32)
        out = self.output_buffer.detach().cpu().numpy()
        self.output_buffer = torch.zeros(self.num_speakers, 0, device=self.device)
        return out
