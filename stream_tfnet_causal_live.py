#!/usr/bin/env python3
"""
Live streaming: stereo mic input -> TFNet Causal model -> audio output.

Input shape: (B=1, C=2, T)
Output: selected source streamed to output device.
"""

from __future__ import annotations

import argparse
import signal
import sys
from types import SimpleNamespace
from typing import Any, Dict, Iterable, Tuple

import numpy as np
import sounddevice as sd
import soundfile as sf
import torch
import yaml

from TFNet_causal import CausalTFNetSeparator


def dict_to_ns(d: Dict[str, Any]) -> SimpleNamespace:
    out: Dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out[k] = dict_to_ns(v)
        else:
            out[k] = v
    return SimpleNamespace(**out)


def list_devices() -> None:
    devices = sd.query_devices()
    for idx, dev in enumerate(devices):
        if dev.get("max_input_channels", 0) > 0 or dev.get("max_output_channels", 0) > 0:
            name = dev.get("name", "unknown")
            host = dev.get("hostapi", None)
            inputs = dev.get("max_input_channels", 0)
            outputs = dev.get("max_output_channels", 0)
            print(f"{idx}: {name} (in={inputs}, out={outputs}, hostapi={host})")


def resolve_device(selector: str) -> int:
    if selector.isdigit():
        return int(selector)

    devices = sd.query_devices()
    selector_lower = selector.lower()
    matches = []
    for idx, dev in enumerate(devices):
        name = dev.get("name", "")
        if selector_lower in name.lower():
            matches.append((idx, name))
    if not matches:
        raise ValueError(f"No device matching '{selector}'")
    if len(matches) > 1:
        names = ", ".join([f"{idx}:{name}" for idx, name in matches])
        raise ValueError(f"Multiple matches for '{selector}': {names}")
    return matches[0][0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live TFNet Causal streaming.")
    parser.add_argument("--config", type=str, default="config_tfnetcasual.yaml", help="YAML config.")
    parser.add_argument("--checkpoint", type=str, default="./checkpoint/soundBubble_tfnet_causal_updated_best.pt", help="Model weights.")
    parser.add_argument("--samplerate", type=int, default=16000, help="Sample rate in Hz.")
    parser.add_argument("--blocksize", type=int, default=None, help="Block size in frames.")
    parser.add_argument("--stereo-device", type=str, help="Stereo input device index or name.")
    parser.add_argument("--output-device", type=str, help="Output device index or name.")
    parser.add_argument("--device", type=str, default="mps", help="Torch device (cpu/cuda).")
    parser.add_argument(
        "--vec",
        type=float,
        nargs=6,
        default=[-0.5, 0.5, 0.0, 0.5, 0.5, 0.0],
        help="Direction vectors for two sources: 6 floats (x1 y1 z1 x2 y2 z2).",
    )
    parser.add_argument(
        "--source-index",
        type=int,
        default=1,
        choices=[0, 1],
        help="Which separated source to play (0 or 1).",
    )
    parser.add_argument(
        "--output-mode",
        type=str,
        default="stereo",
        choices=["mono", "stereo"],
        help="If model outputs mono, duplicate to stereo; if stereo, pass through.",
    )
    parser.add_argument("--list-devices", action="store_true", help="List audio devices and exit.")
    return parser.parse_args()


def load_checkpoint(path: str, device: torch.device) -> Dict[str, Any]:
    ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt:
            return ckpt["state_dict"]
        if "model" in ckpt:
            return ckpt["model"]
    return ckpt


def main() -> int:
    args = parse_args()

    if args.list_devices:
        list_devices()
        return 0

    if not args.stereo_device:
        print("Please provide --stereo-device (aggregate 2-channel input).", file=sys.stderr)
        return 2

    cfg_dict = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    if not isinstance(cfg_dict, dict):
        print("Invalid config file.", file=sys.stderr)
        return 2

    emit_len = int(cfg_dict["network_audio"]["streaming_emit_len"])
    left_len = int(cfg_dict["network_audio"].get("streaming_left_len", 0))

    blocksize = args.blocksize or emit_len
    if blocksize != emit_len:
        print(
            f"blocksize ({blocksize}) must match streaming_emit_len ({emit_len}) for stable streaming.",
            file=sys.stderr,
        )
        return 2

    cfg = dict_to_ns(cfg_dict)
    model = CausalTFNetSeparator(cfg)
    device = torch.device(args.device)
    model.to(device)
    model.eval()

    state_dict = load_checkpoint(args.checkpoint, device)
    model.load_state_dict(state_dict, strict=False)

    stereo_idx = resolve_device(args.stereo_device)
    output_idx = resolve_device(args.output_device) if args.output_device else None

    stereo_info = sd.query_devices(stereo_idx, "input")
    if stereo_info["max_input_channels"] < 2:
        print("Selected stereo input has fewer than 2 channels.", file=sys.stderr)
        return 2

    vec = torch.tensor(args.vec, dtype=torch.float32, device=device).view(1, 6)

    ctx_len = max(0, left_len)
    ctx_buf = np.zeros((2, ctx_len), dtype=np.float32)
    channel0_audio = []
    channel1_audio = []
    mix_audio = []

    stopped = False
    stop_requested = False

    def _handle_sigint(_signum: int, _frame: object) -> None:
        nonlocal stop_requested
        stop_requested = True
        print("\nStopping... finishing current block.")

    old_handler = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, _handle_sigint)
    with sd.InputStream(
        device=stereo_idx,
        channels=2,
        samplerate=args.samplerate,
        blocksize=blocksize,
        dtype="float32",
    ) as in_stream, sd.OutputStream(
        device=output_idx,
        channels=2,
        samplerate=args.samplerate,
        blocksize=blocksize,
        dtype="float32",
    ) as out_stream:
        print("Streaming... press Ctrl+C to stop.")
        try:
            while True:
                if stop_requested:
                    stopped = True
                    break
                block, _ = in_stream.read(blocksize)
                block = block.astype(np.float32, copy=False)
                mix_audio.append(block.copy())
                if ctx_len > 0:
                    audio = np.concatenate([ctx_buf, block.T], axis=1)
                    ctx_buf = audio[:, -ctx_len:]
                else:
                    audio = block.T

                audio_t = torch.from_numpy(audio).unsqueeze(0).to(device)
                with torch.no_grad():
                    out = model(audio_t, vec)

                out_np = out.detach().cpu().numpy()
                if out_np.ndim == 4:
                    # (B, 2, 2, T) -> choose source and keep stereo
                    src = out_np[0, args.source_index]  # (2, T)
                    out_block = src[:, -blocksize:].T
                else:
                    # (B, 2, T) -> choose source, mono
                    src = out_np[0, args.source_index]  # (T,)
                    mono = src[-blocksize:]
                    if args.output_mode == "stereo":
                        out_block = np.column_stack([mono, mono])
                    else:
                        out_block = np.column_stack([mono, mono])

                out_block = out_block.astype(np.float32, copy=False)
                channel0_audio.append(out_block[:, 0].copy())
                channel1_audio.append(out_block[:, 1].copy())
                out_stream.write(out_block)
        except KeyboardInterrupt:
            stopped = True
            print("\nStopped.")
        finally:
            signal.signal(signal.SIGINT, old_handler)

    if stopped and channel0_audio and channel1_audio:
        ch0 = np.concatenate(channel0_audio, axis=0)
        ch1 = np.concatenate(channel1_audio, axis=0)
        sf.write("SS_01.wav", ch0, args.samplerate)
        sf.write("SS_02.wav", ch1, args.samplerate)
        if mix_audio:
            mix = np.concatenate(mix_audio, axis=0)
            sf.write("mix.wav", mix, args.samplerate)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
