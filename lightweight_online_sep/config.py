"""Runtime configuration for the default separator checkpoint."""

from typing import Any, Mapping


DEFAULT_SEPARATOR_CONFIG = {
    "n_fft": 512,
    "hop_length": 128,
    "win_length": 512,
    "dropout": 0.08,
    "hidden_size": 128,
    "num_layers": 3,
}


def validate_model_hparams(hparams: Mapping[str, Any]) -> None:
    if int(hparams["hop_length"]) > int(hparams["win_length"]):
        raise ValueError("hop_length must be <= win_length")
    if int(hparams["n_fft"]) < int(hparams["win_length"]):
        raise ValueError("n_fft should be >= win_length")
    if int(hparams["hidden_size"]) < 32:
        raise ValueError("hidden_size must be >= 32")
    if int(hparams["num_layers"]) < 1:
        raise ValueError("num_layers must be >= 1")


def separator_kwargs_from_config(cfg: Mapping[str, Any]) -> dict:
    cfg = cfg or {}
    merged = dict(DEFAULT_SEPARATOR_CONFIG)
    for key in DEFAULT_SEPARATOR_CONFIG:
        if key in cfg and cfg[key] is not None:
            merged[key] = cfg[key]

    return {
        "n_fft": int(merged["n_fft"]),
        "hop_length": int(merged["hop_length"]),
        "win_length": int(merged["win_length"]),
        "dropout": float(merged["dropout"]),
        "hidden_size": int(merged["hidden_size"]),
        "num_layers": int(merged["num_layers"]),
    }
