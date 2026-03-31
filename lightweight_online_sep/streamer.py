from typing import Any, Optional

import numpy as np
import torch


class OnlineSeparatorStreamer:
    """Stateful low-latency streaming wrapper for LightweightCausalSeparator."""

    def __init__(
        self,
        model: torch.nn.Module,
        device: str = "cpu",
        target_azimuth_deg: Optional[np.ndarray] = None,
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
        self.target_azimuth_deg = None
        if target_azimuth_deg is not None:
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

    def _normalize_input_chunk(self, chunk: np.ndarray) -> np.ndarray:
        x = np.asarray(chunk, dtype=np.float32)
        if x.ndim == 1:
            x = np.stack([x, x], axis=0)  # [2, T]
        elif x.ndim == 2:
            if x.shape[0] == 2:
                pass  # [2, T]
            elif x.shape[1] == 2:
                x = x.T  # [2, T]
            elif x.shape[0] == 1:
                x = np.repeat(x, 2, axis=0)
            elif x.shape[1] == 1:
                x = np.repeat(x.T, 2, axis=0)
            else:
                x = x[:2, :] if x.shape[0] > x.shape[1] else x[:, :2].T
        else:
            raise ValueError(f"Unsupported chunk shape: {tuple(x.shape)}")

        return np.ascontiguousarray(x, dtype=np.float32)

    @torch.no_grad()
    def _emit_ready_frames(self) -> torch.Tensor:
        emitted = []

        while self.analysis_buffer.shape[1] >= self.win_length:
            frame_stereo = self.analysis_buffer[:, : self.win_length]  # [2, W]
            self.analysis_buffer = self.analysis_buffer[:, self.hop_length :]

            frame_mono = torch.mean(frame_stereo, dim=0)  # [W]
            frame_win_mono = frame_mono * self.window
            mix_spec = torch.fft.rfft(frame_win_mono, n=self.n_fft).unsqueeze(0)  # [1, F]

            frame_win_stereo = frame_stereo * self.window.unsqueeze(0)
            mix_spec_stereo = torch.fft.rfft(frame_win_stereo, n=self.n_fft).unsqueeze(0)  # [1, 2, F]
            est_spec, self.hidden, _ = self.model.forward_step(
                mix_spec,
                self.hidden,
                frame_time=frame_mono.unsqueeze(0),
                mix_spec_frame_stereo=mix_spec_stereo,
                azimuth_deg=self.target_azimuth_deg,
            )

            est_spec = est_spec.squeeze(0)
            if est_spec.ndim == 2:
                est_frame = torch.fft.irfft(est_spec, n=self.n_fft)[:, : self.win_length]
            elif est_spec.ndim == 3:
                s, c, f = est_spec.shape
                flat_spec = est_spec.reshape(s * c, f)
                est_frame = torch.fft.irfft(flat_spec, n=self.n_fft)[:, : self.win_length]
                est_frame = est_frame.reshape(s, c, self.win_length).mean(dim=1)
            else:
                raise RuntimeError(f"Unsupported forward_step output shape: {tuple(est_spec.shape)}")
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
            chunk: mono or stereo chunk, shape [T], [T,2], or [2,T]
        Returns:
            separated chunk [S, T]
        """
        chunk_2ch = self._normalize_input_chunk(chunk)
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
        tail = np.zeros((2, self.win_length), dtype=np.float32)
        _ = self.process_chunk(tail)
        if self.output_buffer.shape[1] == 0:
            return np.zeros((self.num_speakers, 0), dtype=np.float32)
        out = self.output_buffer.detach().cpu().numpy()
        self.output_buffer = torch.zeros(self.num_speakers, 0, device=self.device)
        return out
