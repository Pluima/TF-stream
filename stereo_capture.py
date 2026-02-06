#!/usr/bin/env python3
"""
Capture stereo audio from two separate input devices on macOS.

Two modes:
1) Dual-device mix: left=MacBook mic, right=USB mic (software alignment)
2) Stereo device: record from a single 2-channel aggregate device

Usage:
  python3 stereo_capture.py --list-devices
  python3 stereo_capture.py --left-device 0 --right-device 2 --duration 10 --outfile stereo.wav
  python3 stereo_capture.py --stereo-device "Aggregate Device" --duration 10 --outfile stereo.wav
"""

from __future__ import annotations

import argparse
import queue
import sys
import time
from typing import Optional, Tuple

import numpy as np
import sounddevice as sd
import soundfile as sf


def list_devices() -> None:
    devices = sd.query_devices()
    for idx, dev in enumerate(devices):
        if dev.get("max_input_channels", 0) > 0:
            name = dev.get("name", "unknown")
            host = dev.get("hostapi", None)
            print(f"{idx}: {name} (hostapi={host})")


def resolve_device(selector: str) -> int:
    if selector.isdigit():
        return int(selector)

    devices = sd.query_devices()
    selector_lower = selector.lower()
    matches = []
    for idx, dev in enumerate(devices):
        if dev.get("max_input_channels", 0) > 0:
            name = dev.get("name", "")
            if selector_lower in name.lower():
                matches.append((idx, name))
    if not matches:
        raise ValueError(f"No input device matching '{selector}'")
    if len(matches) > 1:
        names = ", ".join([f"{idx}:{name}" for idx, name in matches])
        raise ValueError(f"Multiple matches for '{selector}': {names}")
    return matches[0][0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture stereo from two mics.")
    parser.add_argument("--list-devices", action="store_true", help="List input devices and exit.")
    parser.add_argument("--left-device", type=str, help="Device index or name for LEFT channel.")
    parser.add_argument("--right-device", type=str, help="Device index or name for RIGHT channel.")
    parser.add_argument(
        "--stereo-device",
        type=str,
        help="Device index or name for a 2-channel (aggregate) input.",
    )
    parser.add_argument("--samplerate", type=int, default=48000, help="Sample rate in Hz.")
    parser.add_argument("--blocksize", type=int, default=1024, help="Block size in frames.")
    parser.add_argument("--duration", type=float, default=10.0, help="Duration in seconds.")
    parser.add_argument("--outfile", type=str, default="stereo.wav", help="Output WAV file.")
    return parser.parse_args()


def open_input_stream(
    device_index: int,
    samplerate: int,
    blocksize: int,
    q: queue.Queue,
    label: str,
) -> sd.InputStream:
    def callback(indata, frames, time_info, status):
        if status:
            print(f"[{label}] {status}", file=sys.stderr)
        q.put(indata.copy(), block=False)

    return sd.InputStream(
        device=device_index,
        channels=1,
        samplerate=samplerate,
        blocksize=blocksize,
        dtype="float32",
        callback=callback,
    )


def main() -> int:
    args = parse_args()

    if args.list_devices:
        list_devices()
        return 0

    if args.stereo_device:
        if args.left_device or args.right_device:
            print("Use either --stereo-device or --left/right-device, not both.", file=sys.stderr)
            return 2
    else:
        if not args.left_device or not args.right_device:
            print("Both --left-device and --right-device are required.", file=sys.stderr)
            return 2

    try:
        stereo_idx = resolve_device(args.stereo_device) if args.stereo_device else None
        left_idx = resolve_device(args.left_device) if args.left_device else None
        right_idx = resolve_device(args.right_device) if args.right_device else None
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    if stereo_idx is not None:
        stereo_info = sd.query_devices(stereo_idx, "input")
        if stereo_info["max_input_channels"] < 2:
            print("Selected stereo device has fewer than 2 input channels.", file=sys.stderr)
            return 2
    else:
        left_info = sd.query_devices(left_idx, "input")
        right_info = sd.query_devices(right_idx, "input")

        if left_info["default_samplerate"] != right_info["default_samplerate"]:
            print(
                "Warning: devices have different default samplerates. "
                "Consider setting --samplerate explicitly.",
                file=sys.stderr,
            )

    total_frames = int(args.duration * args.samplerate)
    written_frames = 0

    if stereo_idx is not None:
        with sd.InputStream(
            device=stereo_idx,
            channels=2,
            samplerate=args.samplerate,
            blocksize=args.blocksize,
            dtype="float32",
        ) as stream, sf.SoundFile(
            args.outfile, mode="w", samplerate=args.samplerate, channels=2, subtype="PCM_16"
        ) as outfile:
            print("Recording (stereo device)...")
            start_time = time.time()
            while written_frames < total_frames:
                block, _ = stream.read(args.blocksize)
                frames = len(block)
                remaining = total_frames - written_frames
                if frames > remaining:
                    block = block[:remaining, :]
                    frames = remaining
                outfile.write(block)
                written_frames += frames
            elapsed = time.time() - start_time
            print(f"Done. Wrote {written_frames} frames in {elapsed:.2f}s to {args.outfile}")
    else:
        left_q: queue.Queue = queue.Queue(maxsize=32)
        right_q: queue.Queue = queue.Queue(maxsize=32)

        with open_input_stream(left_idx, args.samplerate, args.blocksize, left_q, "left"), open_input_stream(
            right_idx, args.samplerate, args.blocksize, right_q, "right"
        ), sf.SoundFile(
            args.outfile, mode="w", samplerate=args.samplerate, channels=2, subtype="PCM_16"
        ) as outfile:
            print("Recording (dual devices)...")
            start_time = time.time()
            while written_frames < total_frames:
                left_block = left_q.get()
                right_block = right_q.get()

                frames = min(len(left_block), len(right_block))
                if frames == 0:
                    continue

                stereo = np.column_stack((left_block[:frames], right_block[:frames]))

                remaining = total_frames - written_frames
                if frames > remaining:
                    stereo = stereo[:remaining, :]
                    frames = remaining

                outfile.write(stereo)
                written_frames += frames

            elapsed = time.time() - start_time
            print(f"Done. Wrote {written_frames} frames in {elapsed:.2f}s to {args.outfile}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
