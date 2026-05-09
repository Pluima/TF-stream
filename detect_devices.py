#!/usr/bin/env python3
"""List audio input/output devices for run_streaming.py."""

from __future__ import annotations

import argparse
from typing import Any


def _device_value(device: dict[str, Any], key: str, default: Any = "-") -> Any:
    value = device.get(key, default)
    return default if value is None else value


def _print_devices(devices, default_input, default_output) -> None:
    print("Input devices:")
    for index, device in enumerate(devices):
        channels = int(_device_value(device, "max_input_channels", 0))
        if channels <= 0:
            continue
        marker = " *default*" if index == default_input else ""
        name = _device_value(device, "name", "unknown")
        sample_rate = _device_value(device, "default_samplerate", "-")
        print(f"  {index}: {name} | inputs={channels} | default_sr={sample_rate}{marker}")

    print("\nOutput devices:")
    for index, device in enumerate(devices):
        channels = int(_device_value(device, "max_output_channels", 0))
        if channels <= 0:
            continue
        marker = " *default*" if index == default_output else ""
        name = _device_value(device, "name", "unknown")
        sample_rate = _device_value(device, "default_samplerate", "-")
        print(f"  {index}: {name} | outputs={channels} | default_sr={sample_rate}{marker}")


def _print_recommendation(default_input, default_output) -> None:
    if default_input is None or default_output is None:
        print("\nNo complete default input/output pair detected.")
        return
    print("\nDefault run_streaming command:")
    print(f"  python run_streaming.py --input-device {default_input} --output-device {default_output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect sounddevice input/output device IDs.")
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="Kept for compatibility; listing is the default behavior.",
    )
    return parser.parse_args()


def main() -> int:
    parse_args()
    try:
        import sounddevice as sd
    except ImportError as exc:
        raise SystemExit("Please install sounddevice first: pip install sounddevice") from exc

    devices = sd.query_devices()
    default_input, default_output = sd.default.device
    default_input = None if default_input is None or default_input < 0 else int(default_input)
    default_output = None if default_output is None or default_output < 0 else int(default_output)
    _print_devices(devices, default_input, default_output)
    _print_recommendation(default_input, default_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
