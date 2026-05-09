import argparse
import os
import queue
import sys
import threading
import time
from datetime import datetime
from typing import Any, Optional

import numpy as np
import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

from lightweight_online_sep.lightweight_sep_model import build_separator_from_config  # noqa: E402
from lightweight_online_sep.streamer import OnlineSeparatorStreamer  # noqa: E402


DEFAULT_CKPT = "./checkpoint/lightweight_sep_lrs2vox2_best.pt"


class AsyncSeparationWorker:
    """Runs separator inference in a background thread."""

    def __init__(
        self,
        streamer: Any,
        queue_size: int = 128,
        process_chunks_per_step: int = 4,
    ):
        self.streamer = streamer
        self.process_chunks_per_step = max(1, int(process_chunks_per_step))
        self.input_queue: queue.Queue = queue.Queue(maxsize=max(1, int(queue_size)))
        self.output_queue: queue.Queue = queue.Queue(maxsize=max(1, int(queue_size)))
        self._stop_token = object()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self.dropped_input_chunks = 0
        self.dropped_output_chunks = 0
        self.processed_input_chunks = 0

    def start(self) -> None:
        self._thread.start()

    def enqueue_input(self, audio_chunk: np.ndarray) -> None:
        chunk = _ensure_stereo(audio_chunk)
        try:
            self.input_queue.put_nowait(chunk)
            return
        except queue.Full:
            pass

        try:
            self.input_queue.get_nowait()
        except queue.Empty:
            pass
        self.dropped_input_chunks += 1

        try:
            self.input_queue.put_nowait(chunk)
        except queue.Full:
            self.dropped_input_chunks += 1

    def dequeue_output(self) -> Optional[np.ndarray]:
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
            self.output_queue.get_nowait()
        except queue.Empty:
            pass
        self.dropped_output_chunks += 1

        try:
            self.output_queue.put_nowait(out_chunk)
        except queue.Full:
            self.dropped_output_chunks += 1

    def _process_pending(self, pending_chunks: list[np.ndarray]) -> None:
        if not pending_chunks:
            return
        lengths = [int(chunk.shape[0]) for chunk in pending_chunks]
        concat = np.concatenate(pending_chunks, axis=0)
        sep = self.streamer.process_chunk(concat)
        sep = np.ascontiguousarray(sep.T, dtype=np.float32)

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
            try:
                item = self.input_queue.get(timeout=0.05)
            except queue.Empty:
                item = None

            if item is self._stop_token:
                stop_requested = True
            elif item is not None:
                pending_chunks.append(item)

            if len(pending_chunks) >= self.process_chunks_per_step or (stop_requested and pending_chunks):
                self._process_pending(pending_chunks)
                pending_chunks = []

            if stop_requested and not pending_chunks and self.input_queue.empty():
                break

        tail = self.streamer.flush()
        if tail is not None and tail.size > 0 and tail.shape[1] > 0:
            self._enqueue_output(np.ascontiguousarray(tail.T, dtype=np.float32))

    def stop(self, join_timeout_sec: float = 30.0) -> None:
        while True:
            try:
                self.input_queue.put(self._stop_token, timeout=0.1)
                break
            except queue.Full:
                continue
        self._thread.join(timeout=join_timeout_sec)

    def drain_outputs(self) -> list[np.ndarray]:
        outputs = []
        while True:
            try:
                outputs.append(self.output_queue.get_nowait())
            except queue.Empty:
                break
        return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Quasi-real-time streaming speech separation")
    parser.add_argument("--checkpoint", default=DEFAULT_CKPT, help="Path to the learned separator checkpoint.")
    parser.add_argument("--device", default="mps" if torch.cuda.is_available() else "cpu", help="Torch inference device.")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--block-size", type=int, default=512, help="Audio callback block size.")
    parser.add_argument("--input-channels", type=int, default=2)
    parser.add_argument("--input-device", type=int, default=None)
    parser.add_argument("--output-device", type=int, default=None)
    parser.add_argument("--async-queue-size", type=int, default=128)
    parser.add_argument("--process-chunks-per-step", type=int, default=4)
    parser.add_argument("--startup-latency-blocks", type=int, default=6)
    parser.add_argument("--fallback-output", choices=["zeros", "passthrough"], default="passthrough")
    parser.add_argument("--stats-interval-sec", type=float, default=2.0)
    parser.add_argument("--target-azimuths-deg", type=str, default="15,-15")
    parser.add_argument("--save-dir", default="./recordings")
    parser.add_argument("--save-prefix", default="streaming_sep")
    return parser.parse_args()


def load_model(checkpoint_path: str, device: str):
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = build_separator_from_config(ckpt["model_config"])
    model_keys = set(model.state_dict())
    state = {key: value for key, value in ckpt["model_state_dict"].items() if key in model_keys}
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def build_streamer(args: argparse.Namespace) -> OnlineSeparatorStreamer:
    model = load_model(args.checkpoint, args.device)
    if args.block_size % model.hop_length != 0:
        raise SystemExit(f"block-size ({args.block_size}) must be multiple of hop_length ({model.hop_length})")

    target_azimuths = _parse_target_azimuths(args.target_azimuths_deg)
    print(f"[Azimuth cue] {target_azimuths.tolist()} deg")

    return OnlineSeparatorStreamer(
        model=model,
        device=args.device,
        target_azimuth_deg=target_azimuths,
    )


def _ensure_stereo(chunk: np.ndarray) -> np.ndarray:
    chunk = np.asarray(chunk, dtype=np.float32)
    if chunk.ndim == 1:
        chunk = chunk[:, None]
    if chunk.shape[1] == 1:
        chunk = np.repeat(chunk, 2, axis=1)
    elif chunk.shape[1] > 2:
        chunk = chunk[:, :2]
    return np.ascontiguousarray(chunk, dtype=np.float32)


def _parse_target_azimuths(text: str) -> np.ndarray:
    vals = [float(item.strip()) for item in str(text).split(",") if item.strip()]
    if len(vals) != 2:
        raise ValueError(f"--target-azimuths-deg expects 2 values, got {len(vals)}")
    return np.asarray(vals, dtype=np.float32)


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
    with wave.open(path, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm16.tobytes())


def _align_length(wav: np.ndarray, target_len: int) -> np.ndarray:
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    if wav.shape[0] < target_len:
        return np.pad(wav, (0, target_len - wav.shape[0]))
    if wav.shape[0] > target_len:
        return wav[:target_len]
    return wav


def _save_stream_outputs(
    save_dir: str,
    save_prefix: str,
    sample_rate: int,
    raw_chunks: list[np.ndarray],
    sep1_chunks: list[np.ndarray],
    sep2_chunks: list[np.ndarray],
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


def _pop_output_frames(worker: AsyncSeparationWorker, frames: int, state: dict) -> Optional[np.ndarray]:
    buffer = state["carry"]
    while buffer.shape[0] < frames:
        next_chunk = worker.dequeue_output()
        if next_chunk is None:
            break
        next_chunk = _ensure_stereo(next_chunk)
        buffer = np.concatenate([buffer, next_chunk], axis=0) if buffer.size else next_chunk

    if buffer.shape[0] < frames:
        state["carry"] = buffer
        return None

    out = buffer[:frames]
    state["carry"] = buffer[frames:]
    return out


def main() -> None:
    args = parse_args()
    try:
        import sounddevice as sd
    except ImportError as exc:
        raise SystemExit("Please install sounddevice first: pip install sounddevice") from exc

    streamer = build_streamer(args)
    worker = AsyncSeparationWorker(
        streamer=streamer,
        queue_size=args.async_queue_size,
        process_chunks_per_step=args.process_chunks_per_step,
    )
    worker.start()

    print(
        "[Separator] "
        f"backend=learned block_size={args.block_size} output_sources={streamer.num_speakers}"
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
        input_model = _ensure_stereo(raw)
        worker.enqueue_input(input_model)

        if not play_state["started"] and worker.output_queue.qsize() >= args.startup_latency_blocks:
            play_state["started"] = True

        out_chunk = _pop_output_frames(worker, frames, play_state) if play_state["started"] else None
        if out_chunk is None:
            out_chunk = input_model if args.fallback_output == "passthrough" else np.zeros((frames, 2), dtype=np.float32)

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
            print("Quasi-real-time streaming started. Press Ctrl+C to stop.")
            while True:
                sd.sleep(200)
                now = time.time()
                if args.stats_interval_sec > 0 and (now - last_stat) >= args.stats_interval_sec:
                    print(
                        "[Async] "
                        f"in_q={worker.input_queue.qsize()} "
                        f"out_q={worker.output_queue.qsize()} "
                        f"dropped_in={worker.dropped_input_chunks} "
                        f"dropped_out={worker.dropped_output_chunks} "
                        f"processed={worker.processed_input_chunks}"
                    )
                    last_stat = now
    except KeyboardInterrupt:
        print("\nStopping stream...")
    finally:
        worker.stop()
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
            f"processed={worker.processed_input_chunks}"
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
