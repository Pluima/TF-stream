import argparse
import ctypes
import math
import os
import queue
import sys
import threading
import time
from datetime import datetime
from typing import Any, Optional, Tuple

import numpy as np
import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

from lightweight_online_sep.lightweight_sep_model import (
    LightweightCausalSeparator,
    adapt_separator_state_dict_for_model,
    build_separator_from_config,
)
from lightweight_online_sep.streamer import OnlineSeparatorStreamer


DEFAULT_CKPT = "./checkpoint/lightweight_sep_best_possible.pt"

SYSTEM_LIBSTDCXX = "/usr/lib/x86_64-linux-gnu/libstdc++.so.6"


class InputChannelCompensator:
    """
    Online pre-emphasis to reduce persistent channel mismatch between two microphones.

    Design goals:
    - Correct slow-varying hardware gain mismatch.
    - Avoid over-correcting true spatial ILD cues (slow EMA + partial correction).
    - Update calibration mainly on high-coherence chunks.
    """

    def __init__(
        self,
        enabled: bool = True,
        remove_dc: bool = False,
        power_ema: float = 0.98,
        gain_smooth: float = 0.90,
        balance_strength: float = 0.99,
        max_balance_db: float = 50.0,
        update_corr_threshold: float = 0.60,
    ):
        self.enabled = bool(enabled)
        self.remove_dc = bool(remove_dc)
        self.power_ema = float(np.clip(power_ema, 0.0, 0.9999))
        self.gain_smooth = float(np.clip(gain_smooth, 0.0, 0.9999))
        self.balance_strength = float(np.clip(balance_strength, 0.0, 1.0))
        self.max_balance_db = max(0.0, float(max_balance_db))
        self.update_corr_threshold = float(np.clip(update_corr_threshold, 0.0, 0.999))
        self.eps = 1e-8

        self._power = np.ones(2, dtype=np.float32)
        self._gain = np.ones(2, dtype=np.float32)
        self._updates = 0

    def _chunk_corr(self, x: np.ndarray) -> float:
        l = np.asarray(x[:, 0], dtype=np.float32)
        r = np.asarray(x[:, 1], dtype=np.float32)
        l = l - float(np.mean(l))
        r = r - float(np.mean(r))
        denom = float(np.linalg.norm(l) * np.linalg.norm(r) + self.eps)
        if denom <= self.eps:
            return 0.0
        return float(np.dot(l, r) / denom)

    def process(self, chunk: np.ndarray) -> np.ndarray:
        x = _ensure_stereo(chunk).astype(np.float32, copy=False)
        if (not self.enabled) or x.size == 0:
            return np.ascontiguousarray(x, dtype=np.float32)

        if self.remove_dc:
            x = x - np.mean(x, axis=0, keepdims=True)

        corr = abs(self._chunk_corr(x))
        chunk_power = np.mean(x * x, axis=0).astype(np.float32) + self.eps
        should_update = corr >= self.update_corr_threshold
        if should_update:
            if self._updates == 0:
                self._power = chunk_power
            else:
                self._power = self.power_ema * self._power + (1.0 - self.power_ema) * chunk_power
            self._updates += 1

        # Symmetric gain correction around unity.
        half_log_ratio = 0.5 * math.log(float(self._power[0]) / float(self._power[1]))
        corr_log = self.balance_strength * half_log_ratio
        max_log = self.max_balance_db * math.log(10.0) / 20.0
        corr_log = float(np.clip(corr_log, -max_log, max_log))
        target_gain = np.asarray([math.exp(-corr_log), math.exp(corr_log)], dtype=np.float32)
        self._gain = self.gain_smooth * self._gain + (1.0 - self.gain_smooth) * target_gain

        y = x * self._gain[None, :]
        return np.ascontiguousarray(y, dtype=np.float32)

    def status(self) -> dict:
        p_ratio_db = 10.0 * math.log10(float(self._power[0] + self.eps) / float(self._power[1] + self.eps))
        g_l_db = 20.0 * math.log10(float(self._gain[0] + self.eps))
        g_r_db = 20.0 * math.log10(float(self._gain[1] + self.eps))
        return {
            "updates": int(self._updates),
            "power_ratio_db": float(p_ratio_db),
            "gain_l_db": float(g_l_db),
            "gain_r_db": float(g_r_db),
        }


class FastMNMF2SeparatorStreamer:
    """
    Sliding-context wrapper around pyroomacoustics FastMNMF2 for quasi-real-time use.

    This is not truly streaming in the same sense as the learned causal separator:
    each inference step re-estimates sources on a recent context window, then emits
    only the newest samples.
    """

    def __init__(
        self,
        sample_rate: int,
        block_size: int,
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: Optional[int] = None,
        context_seconds: float = 2.0,
        n_iter: int = 10,
        n_components: int = 8,
        reference_mic: int = 0,
        accelerate: bool = True,
        num_speakers: int = 2,
    ):
        try:
            from pyroomacoustics.bss.fastmnmf2 import fastmnmf2 as pra_fastmnmf2
        except Exception as exc:
            pra_fastmnmf2 = self._try_import_fastmnmf2_with_fallback(exc)

        self.fastmnmf2 = pra_fastmnmf2
        self.sample_rate = int(sample_rate)
        self.block_size = int(block_size)
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.win_length = int(win_length) if win_length is not None else int(n_fft)
        self.context_seconds = float(context_seconds)
        self.context_samples = max(
            int(round(self.context_seconds * self.sample_rate)),
            self.win_length + self.block_size,
        )
        self.n_iter = max(1, int(n_iter))
        self.n_components = max(1, int(n_components))
        self.reference_mic = int(reference_mic)
        self.accelerate = bool(accelerate)
        self.num_speakers = int(num_speakers)
        if self.num_speakers != 2:
            raise ValueError("FastMNMF2 streamer currently supports exactly 2 output sources.")
        if self.hop_length <= 0 or self.win_length <= 0 or self.n_fft <= 0:
            raise ValueError("FastMNMF2 STFT parameters must be positive.")
        if self.win_length > self.n_fft:
            raise ValueError("FastMNMF2 win_length must be <= n_fft.")

        self.window = torch.hann_window(self.win_length, periodic=True)
        self.reset()

    @staticmethod
    def _try_import_fastmnmf2_with_fallback(original_error: Exception):
        error_text = str(original_error)
        if ("GLIBCXX_" in error_text or "libstdc++.so.6" in error_text) and os.path.exists(SYSTEM_LIBSTDCXX):
            try:
                ctypes.CDLL(SYSTEM_LIBSTDCXX, mode=ctypes.RTLD_GLOBAL)
                from pyroomacoustics.bss.fastmnmf2 import fastmnmf2 as pra_fastmnmf2

                print(f"[FastMNMF2] loaded system libstdc++ fallback: {SYSTEM_LIBSTDCXX}")
                return pra_fastmnmf2
            except Exception as retry_exc:
                raise SystemExit(
                    "Failed to import pyroomacoustics FastMNMF2 backend even after loading system libstdc++. "
                    f"Original error: {original_error} | Retry error: {retry_exc}"
                ) from retry_exc

        raise SystemExit(
            "Failed to import pyroomacoustics FastMNMF2 backend. "
            "Please check your pyroomacoustics installation and shared-library setup. "
            f"Original error: {original_error}"
        ) from original_error

    def reset(self) -> None:
        self.input_buffer = np.zeros((0, 2), dtype=np.float32)

    @staticmethod
    def _normalize_input_chunk(chunk: np.ndarray) -> np.ndarray:
        x = np.asarray(chunk, dtype=np.float32)
        if x.ndim == 1:
            x = np.repeat(x[:, None], 2, axis=1)
        elif x.ndim == 2:
            if x.shape[1] == 1:
                x = np.repeat(x, 2, axis=1)
            elif x.shape[0] == 2 and x.shape[1] != 2:
                x = x.T
            elif x.shape[1] > 2:
                x = x[:, :2]
        else:
            raise ValueError(f"Unsupported chunk shape: {tuple(x.shape)}")

        if x.ndim != 2 or x.shape[1] < 2:
            raise ValueError(f"FastMNMF2 expects stereo chunks, got shape {tuple(x.shape)}")
        return np.ascontiguousarray(x[:, :2], dtype=np.float32)

    def _stft(self, audio: np.ndarray) -> np.ndarray:
        wav = torch.as_tensor(np.asarray(audio, dtype=np.float32).T.copy(), dtype=torch.float32)
        spec = torch.stft(
            wav,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=True,
            return_complex=True,
        )  # [C, F, N]
        return spec.permute(2, 1, 0).contiguous().cpu().numpy()  # [N, F, C]

    def _istft(self, spec: np.ndarray, length: int) -> np.ndarray:
        spec_t = torch.as_tensor(spec).permute(2, 1, 0).contiguous()  # [S, F, N]
        wav = torch.istft(
            spec_t,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=True,
            length=int(length),
        )  # [S, T]
        return wav.transpose(0, 1).contiguous().cpu().numpy()  # [T, S]

    def _separate_context(self, audio_context: np.ndarray) -> np.ndarray:
        X = self._stft(audio_context)
        Y = self.fastmnmf2(
            X,
            n_src=self.num_speakers,
            n_iter=self.n_iter,
            n_components=self.n_components,
            mic_index=self.reference_mic,
            accelerate=self.accelerate,
        )
        if Y.ndim != 3:
            raise RuntimeError(
                "FastMNMF2 returned unexpected output shape "
                f"{tuple(Y.shape)}. This streamer expects mic_index to be an integer."
            )
        wav = self._istft(Y, length=audio_context.shape[0])  # [T, S]
        return np.ascontiguousarray(wav.T, dtype=np.float32)  # [S, T]

    def process_chunk(self, chunk: np.ndarray) -> np.ndarray:
        chunk_2ch = self._normalize_input_chunk(chunk)
        in_len = int(chunk_2ch.shape[0])

        self.input_buffer = np.concatenate([self.input_buffer, chunk_2ch], axis=0)
        if self.input_buffer.shape[0] > self.context_samples:
            self.input_buffer = self.input_buffer[-self.context_samples :]

        if self.input_buffer.shape[0] < max(self.win_length, self.hop_length * 2):
            return np.zeros((self.num_speakers, in_len), dtype=np.float32)

        recent = np.ascontiguousarray(self.input_buffer, dtype=np.float32)
        sep_recent = self._separate_context(recent)
        if sep_recent.shape[1] < in_len:
            pad = np.zeros((self.num_speakers, in_len - sep_recent.shape[1]), dtype=np.float32)
            sep_recent = np.concatenate([sep_recent, pad], axis=1)
        return np.ascontiguousarray(sep_recent[:, -in_len:], dtype=np.float32)

    def flush(self) -> np.ndarray:
        return np.zeros((self.num_speakers, 0), dtype=np.float32)


class AsyncSeparationWorker:
    """Run separator inference in a background thread for quasi-real-time playback."""

    def __init__(
        self,
        streamer: Any,
        block_size: int,
        queue_size: int = 128,
        process_chunks_per_step: int = 4,
        enable_permutation_align: bool = True,
        perm_align_window: int = 2048,
        perm_switch_margin: float = 0.03,
        enable_output_loudness_norm: bool = True,
        target_rms: float = 0.08,
        gain_smooth: float = 0.92,
        min_gain_db: float = -12.0,
        max_gain_db: float = 18.0,
        peak_limit: float = 0.95,
    ):
        self.streamer = streamer
        self.block_size = int(block_size)
        self.process_chunks_per_step = max(1, int(process_chunks_per_step))

        self.input_queue: queue.Queue = queue.Queue(maxsize=max(1, int(queue_size)))
        self.output_queue: queue.Queue = queue.Queue(maxsize=max(1, int(queue_size)))
        self._stop_token = object()
        self._thread = threading.Thread(target=self._run, daemon=True)

        self.dropped_input_chunks = 0
        self.dropped_output_chunks = 0
        self.processed_input_chunks = 0
        self.perm_swaps = 0

        self.enable_permutation_align = bool(enable_permutation_align)
        self.perm_align_window = max(256, int(perm_align_window))
        self.perm_switch_margin = float(perm_switch_margin)
        self._perm_tail = np.zeros((0, 2), dtype=np.float32)

        self.enable_output_loudness_norm = bool(enable_output_loudness_norm)
        self.target_rms = max(1e-4, float(target_rms))
        self.gain_smooth = float(np.clip(gain_smooth, 0.0, 0.999))
        self.min_gain = float(10.0 ** (float(min_gain_db) / 20.0))
        self.max_gain = float(10.0 ** (float(max_gain_db) / 20.0))
        self.peak_limit = float(np.clip(peak_limit, 0.1, 1.0))
        self._gain_state = 1.0

    def start(self) -> None:
        self._thread.start()

    def enqueue_input(self, audio_chunk: np.ndarray) -> None:
        chunk = _ensure_stereo(audio_chunk)
        try:
            self.input_queue.put_nowait(chunk)
            return
        except queue.Full:
            pass

        # Keep latency bounded: drop oldest chunk to make room for latest input.
        try:
            _ = self.input_queue.get_nowait()
            self.dropped_input_chunks += 1
        except queue.Empty:
            self.dropped_input_chunks += 1

        try:
            self.input_queue.put_nowait(chunk)
        except queue.Full:
            self.dropped_input_chunks += 1

    def dequeue_output(self) -> np.ndarray:
        try:
            return self.output_queue.get_nowait()
        except queue.Empty:
            return None

    def _enqueue_output(self, out_chunk: np.ndarray) -> None:
        if out_chunk is None:
            return
        try:
            self.output_queue.put_nowait(out_chunk)
            return
        except queue.Full:
            pass

        try:
            _ = self.output_queue.get_nowait()
            self.dropped_output_chunks += 1
        except queue.Empty:
            self.dropped_output_chunks += 1

        try:
            self.output_queue.put_nowait(out_chunk)
        except queue.Full:
            self.dropped_output_chunks += 1

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        aa = np.asarray(a, dtype=np.float32).reshape(-1)
        bb = np.asarray(b, dtype=np.float32).reshape(-1)
        denom = float(np.linalg.norm(aa) * np.linalg.norm(bb) + 1e-8)
        if denom <= 1e-8:
            return 0.0
        return float(np.dot(aa, bb) / denom)

    def _apply_permutation_alignment(self, sep: np.ndarray) -> np.ndarray:
        """
        Keep source-channel identity stable across chunks by comparing continuity
        between previous aligned tail and current chunk head.
        """
        sep = np.asarray(sep, dtype=np.float32)
        if (not self.enable_permutation_align) or sep.ndim != 2 or sep.shape[1] != 2:
            if sep.ndim == 2 and sep.shape[1] == 2:
                self._perm_tail = (
                    np.concatenate([self._perm_tail, sep], axis=0)
                    if self._perm_tail.size
                    else sep.copy()
                )
                if self._perm_tail.shape[0] > self.perm_align_window:
                    self._perm_tail = self._perm_tail[-self.perm_align_window :]
            return sep

        if self._perm_tail.shape[0] > 0:
            n = min(self.perm_align_window, self._perm_tail.shape[0], sep.shape[0])
            if n >= 128:
                prev = self._perm_tail[-n:, :]  # [N, 2]
                cur = sep[:n, :]  # [N, 2]
                prev_energy = float(np.mean(prev * prev))
                cur_energy = float(np.mean(cur * cur))
                if prev_energy > 1e-7 and cur_energy > 1e-7:
                    keep_score = self._cosine_similarity(prev[:, 0], cur[:, 0]) + self._cosine_similarity(prev[:, 1], cur[:, 1])
                    swap_score = self._cosine_similarity(prev[:, 0], cur[:, 1]) + self._cosine_similarity(prev[:, 1], cur[:, 0])
                    if swap_score > keep_score + self.perm_switch_margin:
                        sep = sep[:, [1, 0]]
                        self.perm_swaps += 1

        self._perm_tail = (
            np.concatenate([self._perm_tail, sep], axis=0)
            if self._perm_tail.size
            else sep.copy()
        )
        if self._perm_tail.shape[0] > self.perm_align_window:
            self._perm_tail = self._perm_tail[-self.perm_align_window :]
        return sep

    def _apply_loudness_normalization(self, sep: np.ndarray) -> np.ndarray:
        """
        Running RMS loudness control with smooth gain and peak limiter.
        """
        sep = np.asarray(sep, dtype=np.float32)
        if (not self.enable_output_loudness_norm) or sep.size == 0:
            return sep

        rms = float(np.sqrt(np.mean(sep * sep) + 1e-8))
        target_gain = self.target_rms / max(rms, 1e-6)
        target_gain = float(np.clip(target_gain, self.min_gain, self.max_gain))
        self._gain_state = self.gain_smooth * self._gain_state + (1.0 - self.gain_smooth) * target_gain

        out = sep * float(self._gain_state)
        peak = float(np.max(np.abs(out))) if out.size else 0.0
        if peak > self.peak_limit:
            out = out * float(self.peak_limit / max(peak, 1e-8))
        return np.ascontiguousarray(out, dtype=np.float32)

    def _process_pending(self, pending_chunks: list) -> None:
        if not pending_chunks:
            return

        lengths = [int(c.shape[0]) for c in pending_chunks]
        concat = np.concatenate(pending_chunks, axis=0)
        sep = self.streamer.process_chunk(concat)  # [2, T]
        sep = np.ascontiguousarray(sep.T, dtype=np.float32)  # [T, 2]
        sep = self._apply_permutation_alignment(sep)
        sep = self._apply_loudness_normalization(sep)

        start = 0
        for length in lengths:
            end = start + length
            self._enqueue_output(sep[start:end, :2])
            self.processed_input_chunks += 1
            start = end

    def _run(self) -> None:
        pending_chunks = []
        stop_requested = False

        while True:
            got_item = False
            try:
                item = self.input_queue.get(timeout=0.05)
                got_item = True
            except queue.Empty:
                item = None

            if got_item:
                if item is self._stop_token:
                    stop_requested = True
                else:
                    pending_chunks.append(item)

            should_process = len(pending_chunks) >= self.process_chunks_per_step
            if stop_requested and pending_chunks:
                should_process = True

            if should_process:
                self._process_pending(pending_chunks)
                pending_chunks = []

            if stop_requested and not pending_chunks and self.input_queue.empty():
                break

        # Final flush for residual states.
        tail = self.streamer.flush()
        if tail is not None and tail.size > 0 and tail.shape[1] > 0:
            tail_chunk = np.ascontiguousarray(tail.T, dtype=np.float32)
            tail_chunk = self._apply_permutation_alignment(tail_chunk)
            tail_chunk = self._apply_loudness_normalization(tail_chunk)
            self._enqueue_output(tail_chunk)

    def stop(self, join_timeout_sec: float = 30.0) -> None:
        while True:
            try:
                self.input_queue.put(self._stop_token, timeout=0.1)
                break
            except queue.Full:
                continue
        self._thread.join(timeout=join_timeout_sec)

    def drain_outputs(self) -> list:
        outputs = []
        while True:
            try:
                outputs.append(self.output_queue.get_nowait())
            except queue.Empty:
                break
        return outputs


def clean_state_dict(state_dict: dict) -> dict:
    cleaned = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            cleaned[k[7:]] = v
        elif k.startswith("_orig_mod."):
            cleaned[k[10:]] = v
        else:
            cleaned[k] = v
    return cleaned


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Quasi-real-time streaming speech separation")
    parser.add_argument(
        "--separator-backend",
        choices=["learned", "fastmnmf2"],
        default="learned",
        help="Separator backend. `learned` uses your trained checkpoint; `fastmnmf2` uses pyroomacoustics blind separation.",
    )
    parser.add_argument("--checkpoint", default=DEFAULT_CKPT, help="Checkpoint path for `learned` backend.")
    parser.add_argument("--device", default="mps" if torch.cuda.is_available() else "cpu", help="Inference device for `learned` backend.")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument(
        "--block-size",
        type=int,
        default=512,
        help="Audio callback block size. For `learned`, must be a multiple of model hop_length.",
    )
    parser.add_argument("--input-channels", type=int, default=2)
    parser.add_argument("--input-device", type=int, default=None)
    parser.add_argument("--output-device", type=int, default=None)

    parser.add_argument("--async-queue-size", type=int, default=128, help="Queue size for async inference")
    parser.add_argument(
        "--process-chunks-per-step",
        type=int,
        default=4,
        help="Background worker processes this many chunks together each step",
    )
    parser.add_argument(
        "--startup-latency-blocks",
        type=int,
        default=6,
        help="Wait for this many output blocks before playback starts",
    )
    parser.add_argument(
        "--fallback-output",
        choices=["zeros", "passthrough"],
        default="passthrough",
        help="Output mode when model output queue is temporarily empty",
    )
    parser.add_argument("--stats-interval-sec", type=float, default=2.0)
    parser.add_argument(
        "--enable-input-channel-compensation",
        type=int,
        default=0,
        help="Enable online compensation for persistent dual-mic mismatch before model inference.",
    )
    parser.add_argument(
        "--input-remove-dc",
        type=int,
        default=0,
        help="Remove per-chunk DC offset on each input channel before compensation.",
    )
    parser.add_argument(
        "--input-balance-strength",
        type=float,
        default=0.90,
        help="Compensation strength in [0,1]. Higher means stronger channel gain equalization.",
    )
    parser.add_argument(
        "--input-power-ema",
        type=float,
        default=0.98,
        help="EMA factor for channel power tracking (closer to 1.0 means slower updates).",
    )
    parser.add_argument(
        "--input-gain-smooth",
        type=float,
        default=0.90,
        help="Smoothing factor for applied channel gains.",
    )
    parser.add_argument(
        "--input-max-balance-db",
        type=float,
        default=30.0,
        help="Maximum absolute compensation gain per side (dB).",
    )
    parser.add_argument(
        "--input-update-corr-threshold",
        type=float,
        default=0.60,
        help="Update channel calibration only when |L/R correlation| exceeds this threshold.",
    )
    parser.add_argument(
        "--use-azimuth-cue",
        type=int,
        default=1,
        help="Use azimuth cues to condition output source ordering for the `learned` backend.",
    )
    parser.add_argument(
        "--target-azimuths-deg",
        type=str,
        default="",
        help="Comma-separated azimuth degrees for each output source, e.g. '-15,15'. Empty=auto from distance/spacing.",
    )
    parser.add_argument(
        "--source-distance-m",
        type=float,
        default=1.0,
        help="Auto azimuth geometry: source distance to listener (meters).",
    )
    parser.add_argument(
        "--speaker-spacing-m",
        type=float,
        default=0.6,
        help="Auto azimuth geometry: left-right spacing between two speakers (meters).",
    )
    parser.add_argument(
        "--enable-permutation-align",
        type=int,
        default=0,
        help="Enable cross-chunk source permutation alignment.",
    )
    parser.add_argument(
        "--perm-align-window",
        type=int,
        default=2048,
        help="Alignment tail window in samples.",
    )
    parser.add_argument(
        "--perm-switch-margin",
        type=float,
        default=0.03,
        help="Swap only when swapped score exceeds keep score by this margin.",
    )
    parser.add_argument(
        "--enable-output-loudness-norm",
        type=int,
        default=1,
        help="Enable output loudness normalization for monitoring.",
    )
    parser.add_argument(
        "--target-rms",
        type=float,
        default=0.08,
        help="Target output RMS when loudness normalization is enabled.",
    )
    parser.add_argument(
        "--gain-smooth",
        type=float,
        default=0.92,
        help="Running gain smoothing factor in [0, 1). Higher is smoother.",
    )
    parser.add_argument(
        "--min-gain-db",
        type=float,
        default=-12.0,
        help="Minimum gain (dB) for loudness normalization.",
    )
    parser.add_argument(
        "--max-gain-db",
        type=float,
        default=18.0,
        help="Maximum gain (dB) for loudness normalization.",
    )
    parser.add_argument(
        "--peak-limit",
        type=float,
        default=0.95,
        help="Peak limiter ceiling after gain normalization.",
    )
    parser.add_argument(
        "--fastmnmf2-context-seconds",
        type=float,
        default=2.0,
        help="Recent context duration reprocessed on each FastMNMF2 step.",
    )
    parser.add_argument("--fastmnmf2-nfft", type=int, default=1024, help="STFT n_fft for FastMNMF2 backend.")
    parser.add_argument("--fastmnmf2-hop-length", type=int, default=256, help="STFT hop_length for FastMNMF2 backend.")
    parser.add_argument("--fastmnmf2-win-length", type=int, default=1024, help="STFT win_length for FastMNMF2 backend.")
    parser.add_argument("--fastmnmf2-num-iter", type=int, default=10, help="FastMNMF2 iteration count per update.")
    parser.add_argument("--fastmnmf2-n-components", type=int, default=8, help="FastMNMF2 NMF components per source.")
    parser.add_argument("--fastmnmf2-reference-mic", type=int, default=0, help="Reference microphone index used for FastMNMF2 source images.")
    parser.add_argument(
        "--fastmnmf2-accelerate",
        type=int,
        default=1,
        help="Enable FastMNMF2 acceleration path when supported.",
    )

    parser.add_argument(
        "--save-dir",
        default="./recordings",
        help="Directory to save raw and separated wav files when stopping with Ctrl+C",
    )
    parser.add_argument(
        "--save-prefix",
        default="streaming_sep",
        help="Filename prefix for saved wav files",
    )
    return parser.parse_args()


def load_model(checkpoint_path: str, device: str):
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = ckpt.get("model_config", {})
    model = build_separator_from_config(cfg)
    state = adapt_separator_state_dict_for_model(model, clean_state_dict(ckpt["model_state_dict"]))
    load_info = model.load_state_dict(state, strict=False)
    missing = list(getattr(load_info, "missing_keys", []))
    unexpected = list(getattr(load_info, "unexpected_keys", []))
    if missing:
        print(f"[load_model] missing keys ({len(missing)}): {missing[:8]}")
    if unexpected:
        print(f"[load_model] unexpected keys ({len(unexpected)}): {unexpected[:8]}")
    model.to(device).eval()
    return model


def build_separator_backend(args: argparse.Namespace) -> Tuple[Any, dict]:
    backend = str(args.separator_backend).lower()
    if backend == "learned":
        model = load_model(args.checkpoint, args.device)
        if args.block_size % model.hop_length != 0:
            raise SystemExit(
                f"block-size ({args.block_size}) must be multiple of hop_length ({model.hop_length})"
            )

        target_azimuths = _resolve_target_azimuths(args, num_speakers=int(model.num_speakers))
        if target_azimuths is None:
            print("[Azimuth cue] disabled")
        else:
            print(f"[Azimuth cue] enabled: {target_azimuths.tolist()} deg")
            if not bool(getattr(model, "use_azimuth_conditioning", False)):
                print(
                    "[Azimuth cue] note: loaded checkpoint/model does not enable azimuth conditioning; "
                    "cue values will not change model behavior."
                )

        streamer = OnlineSeparatorStreamer(
            model=model,
            device=args.device,
            target_azimuth_deg=target_azimuths,
        )
        return streamer, {
            "backend": backend,
            "name": "learned",
            "num_speakers": int(model.num_speakers),
            "hop_length": int(model.hop_length),
        }

    if int(args.input_channels) != 2:
        raise SystemExit("FastMNMF2 backend currently expects exactly 2 input channels.")
    if bool(args.use_azimuth_cue):
        print("[Azimuth cue] ignored for FastMNMF2 backend (blind separation).")
    if str(args.device).startswith("cuda"):
        print("[FastMNMF2] note: pyroomacoustics FastMNMF2 runs on CPU; --device is ignored.")

    streamer = FastMNMF2SeparatorStreamer(
        sample_rate=args.sample_rate,
        block_size=args.block_size,
        n_fft=args.fastmnmf2_nfft,
        hop_length=args.fastmnmf2_hop_length,
        win_length=args.fastmnmf2_win_length,
        context_seconds=args.fastmnmf2_context_seconds,
        n_iter=args.fastmnmf2_num_iter,
        n_components=args.fastmnmf2_n_components,
        reference_mic=args.fastmnmf2_reference_mic,
        accelerate=bool(args.fastmnmf2_accelerate),
        num_speakers=2,
    )
    print(
        "[FastMNMF2] "
        f"context={args.fastmnmf2_context_seconds:.2f}s "
        f"n_fft={args.fastmnmf2_nfft} hop={args.fastmnmf2_hop_length} "
        f"iter={args.fastmnmf2_num_iter} components={args.fastmnmf2_n_components} "
        f"ref_mic={args.fastmnmf2_reference_mic}"
    )
    return streamer, {
        "backend": backend,
        "name": "fastmnmf2",
        "num_speakers": int(streamer.num_speakers),
        "hop_length": int(streamer.hop_length),
    }


def _write_wav(path: str, audio: np.ndarray, sample_rate: int) -> None:
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1 and audio.shape[1] == 1:
        audio = audio[:, 0]

    try:
        import soundfile as sf

        sf.write(path, audio, sample_rate, subtype="PCM_16")
        return
    except Exception:
        pass

    import wave

    clipped = np.clip(audio, -1.0, 1.0)
    pcm16 = (clipped * 32767.0).astype(np.int16)
    channels = 1 if pcm16.ndim == 1 else pcm16.shape[1]
    with wave.open(path, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16.tobytes())


def _align_length(wav: np.ndarray, target_len: int) -> np.ndarray:
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    if wav.shape[0] < target_len:
        wav = np.pad(wav, (0, target_len - wav.shape[0]))
    elif wav.shape[0] > target_len:
        wav = wav[:target_len]
    return wav


def _ensure_stereo(chunk: np.ndarray) -> np.ndarray:
    chunk = np.asarray(chunk, dtype=np.float32)
    if chunk.ndim == 1:
        chunk = chunk[:, None]
    if chunk.shape[1] == 1:
        chunk = np.repeat(chunk, 2, axis=1)
    elif chunk.shape[1] > 2:
        chunk = chunk[:, :2]
    return np.ascontiguousarray(chunk, dtype=np.float32)


def _resolve_target_azimuths(args: argparse.Namespace, num_speakers: int) -> np.ndarray:
    if not bool(args.use_azimuth_cue):
        return None

    text = str(args.target_azimuths_deg).strip()
    if text:
        vals = [float(x.strip()) for x in text.split(",") if x.strip()]
        if len(vals) != int(num_speakers):
            raise ValueError(
                f"--target-azimuths-deg expects {num_speakers} values, got {len(vals)}"
            )
        return np.asarray(vals, dtype=np.float32)

    if int(num_speakers) == 2:
        dist = max(0.05, float(args.source_distance_m))
        half = 0.5 * max(0.01, float(args.speaker_spacing_m))
        az = math.degrees(math.atan2(half, dist))
        return np.asarray([-az, az], dtype=np.float32)

    return np.linspace(-45.0, 45.0, int(num_speakers), dtype=np.float32)


def _save_stream_outputs(
    save_dir: str,
    save_prefix: str,
    sample_rate: int,
    raw_chunks: list,
    sep1_chunks: list,
    sep2_chunks: list,
) -> None:
    if len(raw_chunks) == 0:
        print("No audio captured; skip saving.")
        return

    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"{save_prefix}_{timestamp}"

    raw_audio = np.concatenate(raw_chunks, axis=0).astype(np.float32)
    raw_len = int(raw_audio.shape[0])

    sep1 = np.concatenate(sep1_chunks, axis=0).astype(np.float32) if sep1_chunks else np.zeros(raw_len, dtype=np.float32)
    sep2 = np.concatenate(sep2_chunks, axis=0).astype(np.float32) if sep2_chunks else np.zeros(raw_len, dtype=np.float32)

    sep1 = _align_length(sep1, raw_len)
    sep2 = _align_length(sep2, raw_len)

    raw_path = os.path.join(save_dir, f"{stem}_raw.wav")
    sep1_path = os.path.join(save_dir, f"{stem}_sep1.wav")
    sep2_path = os.path.join(save_dir, f"{stem}_sep2.wav")

    _write_wav(raw_path, raw_audio, sample_rate)
    _write_wav(sep1_path, sep1, sample_rate)
    _write_wav(sep2_path, sep2, sample_rate)

    print("Saved recordings:")
    print(f"  raw : {raw_path}")
    print(f"  sep1: {sep1_path}")
    print(f"  sep2: {sep2_path}")


def _pop_output_frames(worker: AsyncSeparationWorker, frames: int, state: dict) -> np.ndarray:
    buf = state["carry"]
    while buf.shape[0] < frames:
        nxt = worker.dequeue_output()
        if nxt is None:
            break
        nxt = _ensure_stereo(nxt)
        buf = np.concatenate([buf, nxt], axis=0) if buf.size else nxt

    if buf.shape[0] < frames:
        state["carry"] = buf
        return None

    out = buf[:frames]
    state["carry"] = buf[frames:]
    return out


def main():
    args = parse_args()
    try:
        import sounddevice as sd
    except ImportError as exc:
        raise SystemExit("Please install sounddevice first: pip install sounddevice") from exc

    streamer, backend_info = build_separator_backend(args)
    worker = AsyncSeparationWorker(
        streamer=streamer,
        block_size=args.block_size,
        queue_size=args.async_queue_size,
        process_chunks_per_step=args.process_chunks_per_step,
        enable_permutation_align=bool(args.enable_permutation_align),
        perm_align_window=args.perm_align_window,
        perm_switch_margin=args.perm_switch_margin,
        enable_output_loudness_norm=bool(args.enable_output_loudness_norm),
        target_rms=args.target_rms,
        gain_smooth=args.gain_smooth,
        min_gain_db=args.min_gain_db,
        max_gain_db=args.max_gain_db,
        peak_limit=args.peak_limit,
    )
    input_comp = InputChannelCompensator(
        enabled=bool(args.enable_input_channel_compensation),
        remove_dc=bool(args.input_remove_dc),
        power_ema=args.input_power_ema,
        gain_smooth=args.input_gain_smooth,
        balance_strength=args.input_balance_strength,
        max_balance_db=args.input_max_balance_db,
        update_corr_threshold=args.input_update_corr_threshold,
    )
    worker.start()
    print(
        "[Separator] "
        f"backend={backend_info['name']} "
        f"block_size={args.block_size} "
        f"output_sources={backend_info['num_speakers']}"
    )
    print(
        "[Inference controls] "
        f"perm_align={bool(args.enable_permutation_align)} "
        f"window={args.perm_align_window} margin={args.perm_switch_margin:.3f} | "
        f"loudness_norm={bool(args.enable_output_loudness_norm)} "
        f"target_rms={args.target_rms:.3f} gain_db=[{args.min_gain_db:.1f},{args.max_gain_db:.1f}] "
        f"peak_limit={args.peak_limit:.2f}"
    )
    print(
        "[Input compensation] "
        f"enabled={bool(args.enable_input_channel_compensation)} "
        f"remove_dc={bool(args.input_remove_dc)} "
        f"strength={args.input_balance_strength:.2f} "
        f"power_ema={args.input_power_ema:.3f} "
        f"gain_smooth={args.input_gain_smooth:.3f} "
        f"max_balance_db={args.input_max_balance_db:.1f} "
        f"corr_th={args.input_update_corr_threshold:.2f}"
    )

    raw_chunks = []
    sep1_chunks = []
    sep2_chunks = []

    play_state = {
        "started": args.startup_latency_blocks <= 0,
        "carry": np.zeros((0, 2), dtype=np.float32),
    }

    def callback(indata, outdata, frames, _time, status):
        if status:
            print(status, file=sys.stderr)

        raw = np.array(indata, dtype=np.float32, copy=True)
        raw_chunks.append(raw)
        input_stereo = _ensure_stereo(raw)
        input_model = input_comp.process(input_stereo)
        worker.enqueue_input(input_model)

        if (not play_state["started"]) and worker.output_queue.qsize() >= args.startup_latency_blocks:
            play_state["started"] = True

        out_chunk = None
        if play_state["started"]:
            out_chunk = _pop_output_frames(worker, frames, play_state)

        if out_chunk is None:
            if args.fallback_output == "passthrough":
                out_chunk = input_model
            else:
                out_chunk = np.zeros((frames, 2), dtype=np.float32)

        if out_chunk.shape[0] < frames:
            out_chunk = np.pad(out_chunk, ((0, frames - out_chunk.shape[0]), (0, 0)))
        elif out_chunk.shape[0] > frames:
            tail = out_chunk[frames:]
            carry = play_state["carry"]
            play_state["carry"] = np.concatenate([tail, carry], axis=0) if carry.size else tail
            out_chunk = out_chunk[:frames]

        out_chunk = np.ascontiguousarray(out_chunk[:, :2], dtype=np.float32)
        outdata[:] = out_chunk

        sep1_chunks.append(np.array(out_chunk[:, 0], dtype=np.float32, copy=True))
        sep2_chunks.append(np.array(out_chunk[:, 1], dtype=np.float32, copy=True))

    last_stat = time.time()
    try:
        with sd.Stream(
            samplerate=args.sample_rate,
            blocksize=args.block_size,
            dtype="float32",
            channels=(args.input_channels, 2),
            device=(args.input_device, args.output_device),
            callback=callback,
        ):
            print(f"Quasi-real-time streaming started with backend={backend_info['name']}. Press Ctrl+C to stop.")
            while True:
                sd.sleep(200)
                now = time.time()
                if args.stats_interval_sec > 0 and (now - last_stat) >= args.stats_interval_sec:
                    comp_stat = input_comp.status()
                    print(
                        "[Async] "
                        f"in_q={worker.input_queue.qsize()} "
                        f"out_q={worker.output_queue.qsize()} "
                        f"dropped_in={worker.dropped_input_chunks} "
                        f"dropped_out={worker.dropped_output_chunks} "
                        f"processed={worker.processed_input_chunks} | "
                        f"comp_updates={comp_stat['updates']} "
                        f"mic_lr_db={comp_stat['power_ratio_db']:.2f} "
                        f"gain_l_db={comp_stat['gain_l_db']:.2f} "
                        f"gain_r_db={comp_stat['gain_r_db']:.2f}"
                    )
                    last_stat = now
    except KeyboardInterrupt:
        print("\nStopping stream...")
    finally:
        worker.stop()

        # Collect any pending generated outputs to maximize saved separation coverage.
        if play_state["carry"].shape[0] > 0:
            carry = _ensure_stereo(play_state["carry"])
            sep1_chunks.append(carry[:, 0].copy())
            sep2_chunks.append(carry[:, 1].copy())
            play_state["carry"] = np.zeros((0, 2), dtype=np.float32)

        for pending in worker.drain_outputs():
            pending = _ensure_stereo(pending)
            sep1_chunks.append(pending[:, 0].copy())
            sep2_chunks.append(pending[:, 1].copy())

        print(
            "[Async] final stats: "
            f"dropped_in={worker.dropped_input_chunks}, "
            f"dropped_out={worker.dropped_output_chunks}, "
            f"processed={worker.processed_input_chunks}, "
            f"perm_swaps={worker.perm_swaps}"
        )

        _save_stream_outputs(
            save_dir=args.save_dir,
            save_prefix=args.save_prefix,
            sample_rate=args.sample_rate,
            raw_chunks=raw_chunks,
            sep1_chunks=sep1_chunks,
            sep2_chunks=sep2_chunks,
        )


if __name__ == "__main__":
    main()
