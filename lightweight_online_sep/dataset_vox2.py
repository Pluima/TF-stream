import csv
import math
import os
import random
from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset

try:
    import soundfile as sf
except Exception:
    sf = None


@dataclass(frozen=True)
class MixPair:
    partition: str
    source1_subset: str
    source1_speaker: str
    source1_utt: str
    source1_db: float
    source2_subset: str
    source2_speaker: str
    source2_utt: str
    source2_db: float
    duration: float


@dataclass(frozen=True)
class RenderedMixRow:
    partition: str
    mix: str
    source1: str
    source1_start: int
    source1_db: float
    source2: str
    source2_start: int
    source2_db: float
    az1_deg: float
    az2_deg: float
    duration: float
    target1_ref: str = ""
    target2_ref: str = ""
    target_ref_channel: str = ""


RowType = Union[MixPair, RenderedMixRow]


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_int(value, default=0):
    try:
        return int(float(value))
    except Exception:
        return int(default)


def _parse_vector_text(value) -> Optional[np.ndarray]:
    text = str(value).strip()
    if not text:
        return None
    text = text.strip("[]()")
    parts = [p.strip() for p in text.replace(";", ",").split(",") if p.strip()]
    if len(parts) < 2:
        return None
    vals: List[float] = []
    for p in parts[:3]:
        try:
            vals.append(float(p))
        except Exception:
            return None
    return np.asarray(vals, dtype=np.float32)


def _vector_to_azimuth_deg(v: np.ndarray) -> float:
    if v.size < 2:
        return float("nan")
    x = float(v[0])
    y = float(v[1])
    if (abs(x) + abs(y)) < 1e-8:
        return float("nan")
    return float(math.degrees(math.atan2(x, y)))


def _infer_azimuth_deg(row: dict, az_key: str, vec_key: str) -> float:
    az = _safe_float(row.get(az_key, "nan"), float("nan"))
    if np.isfinite(az):
        return float(az)
    vec = _parse_vector_text(row.get(vec_key, ""))
    if vec is None:
        return float("nan")
    return _vector_to_azimuth_deg(vec)


def _build_vox_path(root: str, subset: str, speaker: str, utt: str) -> str:
    return os.path.join(
        os.path.expanduser(root),
        "audio_clean",
        str(subset),
        str(speaker),
        f"{str(utt)}.wav",
    )


def _resample_linear(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return np.asarray(audio, dtype=np.float32)

    ratio = float(dst_sr) / float(src_sr)
    out_len = max(1, int(round(audio.shape[0] * ratio)))
    old_x = np.arange(audio.shape[0], dtype=np.float32)
    new_x = np.linspace(0, max(audio.shape[0] - 1, 0), out_len, dtype=np.float32)

    if audio.ndim == 1:
        return np.interp(new_x, old_x, audio).astype(np.float32)

    out = np.empty((out_len, audio.shape[1]), dtype=np.float32)
    for ch in range(audio.shape[1]):
        out[:, ch] = np.interp(new_x, old_x, audio[:, ch]).astype(np.float32)
    return out


def _ensure_mono(audio: np.ndarray, frames: int) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2:
        # Expect time-first from sf with always_2d=True.
        audio = np.mean(audio, axis=1)
    audio = audio.reshape(-1)
    if audio.shape[0] < frames:
        audio = np.pad(audio, (0, frames - audio.shape[0]))
    elif audio.shape[0] > frames:
        audio = audio[:frames]
    return np.asarray(audio, dtype=np.float32)


def _ensure_stereo(audio: np.ndarray, frames: int) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 1:
        audio = np.stack([audio, audio], axis=1)
    elif audio.ndim == 2:
        if audio.shape[1] == 1:
            audio = np.repeat(audio, 2, axis=1)
        elif audio.shape[1] > 2:
            audio = audio[:, :2]
    else:
        raise ValueError(f"Unsupported audio ndim for stereo conversion: {audio.ndim}")

    if audio.shape[0] < frames:
        audio = np.pad(audio, ((0, frames - audio.shape[0]), (0, 0)))
    elif audio.shape[0] > frames:
        audio = audio[:frames]
    return np.asarray(audio, dtype=np.float32)


def _load_segment(path: str, start: int, frames: int, target_sr: int, keep_channels: bool) -> np.ndarray:
    if not path or (not os.path.exists(path)):
        if keep_channels:
            return np.zeros((frames, 2), dtype=np.float32)
        return np.zeros(frames, dtype=np.float32)

    if sf is not None:
        with sf.SoundFile(path) as f:
            src_sr = int(f.samplerate)
            start_src = int(round(float(start) * float(src_sr) / float(target_sr)))
            start_src = max(0, int(start_src))
            if start_src > len(f):
                if keep_channels:
                    return np.zeros((frames, 2), dtype=np.float32)
                return np.zeros(frames, dtype=np.float32)
            f.seek(start_src)
            frames_src = max(1, int(round(float(frames) * float(src_sr) / float(target_sr))))
            audio = f.read(frames=frames_src, dtype="float32", always_2d=bool(keep_channels))
    else:
        info = torchaudio.info(path)
        src_sr = int(info.sample_rate)
        start_src = int(round(float(start) * float(src_sr) / float(target_sr)))
        start_src = max(0, int(start_src))
        if start_src > int(info.num_frames):
            if keep_channels:
                return np.zeros((frames, 2), dtype=np.float32)
            return np.zeros(frames, dtype=np.float32)
        frames_src = max(1, int(round(float(frames) * float(src_sr) / float(target_sr))))
        wav, _ = torchaudio.load(path, frame_offset=start_src, num_frames=frames_src)
        # torchaudio returns [C, T]
        audio = wav.transpose(0, 1).cpu().numpy().astype(np.float32, copy=False)
        if not keep_channels:
            if audio.ndim == 2 and audio.shape[1] >= 1:
                audio = np.mean(audio, axis=1)

    audio = np.asarray(audio, dtype=np.float32)
    if src_sr != target_sr:
        audio = _resample_linear(audio, src_sr, target_sr)

    if keep_channels:
        return _ensure_stereo(audio, frames)
    return _ensure_mono(audio, frames)


def _apply_relative_db(source1: np.ndarray, source2: np.ndarray, source1_db: float, source2_db: float) -> np.ndarray:
    rel_db = float(source2_db) - float(source1_db)
    p1 = float(np.mean(np.square(source1, dtype=np.float64)))
    p2 = float(np.mean(np.square(source2, dtype=np.float64)))
    if p1 <= 1e-12 or p2 <= 1e-12:
        return source2
    scalar = (10.0 ** (rel_db / 20.0)) * np.sqrt(p1 / p2)
    return source2 * float(scalar)


def _is_rendered_csv(csv_path: str) -> bool:
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        try:
            first = next(reader)
        except StopIteration:
            return False
    header = {str(x).strip().lower() for x in first}
    return {"mix", "source1", "source2"}.issubset(header)


def _read_vox2_pair_csv(csv_path: str) -> List[MixPair]:
    rows: List[MixPair] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for line_id, row in enumerate(reader, start=1):
            if len(row) < 10:
                continue
            try:
                rows.append(
                    MixPair(
                        partition=str(row[0]).strip().lower(),
                        source1_subset=str(row[1]).strip(),
                        source1_speaker=str(row[2]).strip(),
                        source1_utt=str(row[3]).strip(),
                        source1_db=_safe_float(row[4], 0.0),
                        source2_subset=str(row[5]).strip(),
                        source2_speaker=str(row[6]).strip(),
                        source2_utt=str(row[7]).strip(),
                        source2_db=_safe_float(row[8], 0.0),
                        duration=max(_safe_float(row[9], 0.0), 0.0),
                    )
                )
            except Exception as exc:
                raise ValueError(f"Invalid csv row at line {line_id}: {row}") from exc
    if not rows:
        raise ValueError(f"No valid samples found in csv: {csv_path}")
    return rows


def _read_rendered_mix_csv(csv_path: str) -> List[RenderedMixRow]:
    rows: List[RenderedMixRow] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"mix", "source1", "source2"}
        if not required.issubset({x.lower() for x in (reader.fieldnames or [])}):
            raise ValueError(f"Rendered mix csv missing required columns: {required}")

        for line_id, row in enumerate(reader, start=2):
            mix = str(row.get("mix", "")).strip()
            source1 = str(row.get("source1", "")).strip()
            source2 = str(row.get("source2", "")).strip()
            if (not mix) or (not source1) or (not source2):
                continue

            partition = (
                str(row.get("partition", "")).strip().lower()
                or str(row.get("split", "")).strip().lower()
                or "train"
            )
            rows.append(
                RenderedMixRow(
                    partition=partition,
                    mix=mix,
                    source1=source1,
                    source1_start=_safe_int(row.get("source1_start", 0), 0),
                    source1_db=_safe_float(row.get("source1_db", 0.0), 0.0),
                    source2=source2,
                    source2_start=_safe_int(row.get("source2_start", 0), 0),
                    source2_db=_safe_float(row.get("source2_db", 0.0), 0.0),
                    az1_deg=_infer_azimuth_deg(row, "az1_deg", "source1_vector"),
                    az2_deg=_infer_azimuth_deg(row, "az2_deg", "source2_vector"),
                    duration=max(_safe_float(row.get("duration", 0.0), 0.0), 0.0),
                    target1_ref=str(row.get("target1_ref", "")).strip(),
                    target2_ref=str(row.get("target2_ref", "")).strip(),
                    target_ref_channel=str(row.get("target_ref_channel", "")).strip(),
                )
            )
    if not rows:
        raise ValueError(f"No valid samples found in rendered csv: {csv_path}")
    return rows


def _row_partition(row: RowType) -> str:
    p = str(getattr(row, "partition", "train")).strip().lower()
    return p if p else "train"


def _split_rows(rows: List[RowType], split: str, valid_ratio: float, seed: int) -> List[RowType]:
    split = str(split).lower()
    train_rows = [r for r in rows if _row_partition(r) == "train"]
    valid_rows = [r for r in rows if _row_partition(r) in {"val", "valid", "validation", "dev"}]
    test_rows = [r for r in rows if _row_partition(r) == "test"]

    if split == "test":
        if not test_rows:
            raise ValueError("CSV does not contain test partition")
        return test_rows

    if not train_rows:
        # Fallback for csvs without partition labels: treat all as train.
        train_rows = list(rows)

    if not valid_rows:
        rng = random.Random(seed)
        idx = list(range(len(train_rows)))
        rng.shuffle(idx)
        valid_count = max(1, int(round(len(train_rows) * valid_ratio)))
        valid_count = min(valid_count, max(1, len(train_rows) - 1)) if len(train_rows) > 1 else 1
        valid_idx = set(idx[:valid_count])
        valid_rows = [train_rows[i] for i in idx[:valid_count]]
        train_rows = [train_rows[i] for i in idx if i not in valid_idx]

    if split == "train":
        return train_rows
    if split in {"valid", "val"}:
        return valid_rows
    raise ValueError(f"Unknown split: {split}")


class VoxCeleb2MixDataset(Dataset):
    """
    Supports two csv formats:
    1) Legacy VoxCeleb2 pair csv (headerless, 10 cols) -> online mono mix construction.
    2) Rendered mix csv (headered with mix/source1/source2) -> load pre-rendered (e.g., HRIR stereo) mixes.
    """

    def __init__(
        self,
        csv_path: str,
        data_root: str,
        split: str = "train",
        sample_rate: int = 16000,
        segment_seconds: float = 2.0,
        valid_ratio: float = 0.1,
        seed: int = 42,
        train_random_offset: bool = True,
    ):
        self.csv_path = os.path.expanduser(csv_path)
        self.data_root = os.path.expanduser(data_root)
        self.sample_rate = int(sample_rate)
        self.segment_samples = max(1, int(round(float(segment_seconds) * self.sample_rate)))
        self.split = str(split).lower()
        self.train_random_offset = bool(train_random_offset)

        self.mode = "rendered" if _is_rendered_csv(self.csv_path) else "vox2_pairs"
        if self.mode == "rendered":
            rows: List[RowType] = _read_rendered_mix_csv(self.csv_path)
        else:
            rows = _read_vox2_pair_csv(self.csv_path)

        self.rows = _split_rows(rows, self.split, valid_ratio=valid_ratio, seed=seed)
        self.rng = random.Random(seed)
        self.row_lengths = [self._estimate_row_len_samples(r) for r in self.rows]

    def __len__(self) -> int:
        return len(self.rows)

    def _resolve_path(self, path: str) -> str:
        p = os.path.expanduser(str(path))
        if os.path.isabs(p):
            return p
        return os.path.join(self.data_root, p)

    def _estimate_row_len_samples(self, row: RowType) -> int:
        if isinstance(row, RenderedMixRow):
            mix_path = self._resolve_path(row.mix)
            if sf is not None:
                try:
                    info = sf.info(mix_path)
                    return max(1, int(round(float(info.frames) * self.sample_rate / float(info.samplerate))))
                except Exception:
                    pass
            else:
                try:
                    info = torchaudio.info(mix_path)
                    return max(1, int(round(float(info.num_frames) * self.sample_rate / float(info.sample_rate))))
                except Exception:
                    pass
            if row.duration > 0:
                return max(1, int(round(float(row.duration) * self.sample_rate)))
            return self.segment_samples

        # Legacy row: duration from csv.
        duration_samples = int(round(float(row.duration) * self.sample_rate))
        return max(self.segment_samples, max(1, duration_samples))

    def _get_start_sample(self, index: int) -> int:
        available = int(self.row_lengths[index])
        max_start = max(0, available - self.segment_samples)
        if self.split == "train" and self.train_random_offset and max_start > 0:
            return self.rng.randint(0, max_start)
        if max_start <= 0:
            return 0
        # deterministic center crop for validation/test
        return max_start // 2

    def __getitem__(self, index: int):
        row = self.rows[index]
        start = self._get_start_sample(index)

        if isinstance(row, RenderedMixRow):
            s1_path = self._resolve_path(row.source1)
            s2_path = self._resolve_path(row.source2)
            mix_path = self._resolve_path(row.mix)
            target1_ref_path = self._resolve_path(row.target1_ref) if str(row.target1_ref).strip() else ""
            target2_ref_path = self._resolve_path(row.target2_ref) if str(row.target2_ref).strip() else ""
            use_rendered_targets = bool(target1_ref_path) and bool(target2_ref_path)

            try:
                if use_rendered_targets:
                    source1 = _load_segment(
                        target1_ref_path,
                        start=start,
                        frames=self.segment_samples,
                        target_sr=self.sample_rate,
                        keep_channels=False,
                    )
                    source2 = _load_segment(
                        target2_ref_path,
                        start=start,
                        frames=self.segment_samples,
                        target_sr=self.sample_rate,
                        keep_channels=False,
                    )
                else:
                    source1 = _load_segment(
                        s1_path,
                        start=start + int(row.source1_start),
                        frames=self.segment_samples,
                        target_sr=self.sample_rate,
                        keep_channels=False,
                    )
                    source2 = _load_segment(
                        s2_path,
                        start=start + int(row.source2_start),
                        frames=self.segment_samples,
                        target_sr=self.sample_rate,
                        keep_channels=False,
                    )
                    source2 = _apply_relative_db(source1, source2, row.source1_db, row.source2_db)
            except Exception:
                source1 = np.zeros(self.segment_samples, dtype=np.float32)
                source2 = np.zeros(self.segment_samples, dtype=np.float32)
                use_rendered_targets = False

            try:
                mix = _load_segment(
                    mix_path,
                    start=start,
                    frames=self.segment_samples,
                    target_sr=self.sample_rate,
                    keep_channels=True,
                )  # [T, 2]
            except Exception:
                mix = np.stack([source1 + source2, source1 + source2], axis=1)
            az1_deg = float(row.az1_deg) if np.isfinite(row.az1_deg) else -15.0
            az2_deg = float(row.az2_deg) if np.isfinite(row.az2_deg) else 15.0
            if use_rendered_targets:
                s1_path = target1_ref_path
                s2_path = target2_ref_path

        else:
            s1_path = _build_vox_path(
                self.data_root,
                row.source1_subset,
                row.source1_speaker,
                row.source1_utt,
            )
            s2_path = _build_vox_path(
                self.data_root,
                row.source2_subset,
                row.source2_speaker,
                row.source2_utt,
            )

            try:
                source1 = _load_segment(
                    s1_path,
                    start=start,
                    frames=self.segment_samples,
                    target_sr=self.sample_rate,
                    keep_channels=False,
                )
                source2 = _load_segment(
                    s2_path,
                    start=start,
                    frames=self.segment_samples,
                    target_sr=self.sample_rate,
                    keep_channels=False,
                )
            except Exception:
                source1 = np.zeros(self.segment_samples, dtype=np.float32)
                source2 = np.zeros(self.segment_samples, dtype=np.float32)

            source2 = _apply_relative_db(source1, source2, row.source1_db, row.source2_db)
            mono_mix = source1 + source2
            mix = np.asarray(mono_mix, dtype=np.float32)  # [T]
            az1_deg = -15.0
            az2_deg = 15.0

        if mix.ndim == 2:
            mix_peak = float(np.max(np.abs(mix))) if mix.size else 0.0
        else:
            mix_peak = float(np.max(np.abs(mix))) if mix.size else 0.0
        peak = max(
            float(np.max(np.abs(source1))) if source1.size else 0.0,
            float(np.max(np.abs(source2))) if source2.size else 0.0,
            mix_peak,
            1e-7,
        )
        if peak > 1.0:
            scale = 0.98 / peak
            source1 = source1 * scale
            source2 = source2 * scale
            mix = mix * scale

        if mix.ndim == 2:
            mix_t = torch.from_numpy(mix.T.copy())  # [2, T]
        else:
            mix_t = torch.from_numpy(mix.copy())  # [T]

        return {
            "mix": mix_t,
            "sources": torch.stack(
                [
                    torch.from_numpy(source1),
                    torch.from_numpy(source2),
                ],
                dim=0,
            ),
            "azimuth_deg": torch.tensor([az1_deg, az2_deg], dtype=torch.float32),
            "source1_path": s1_path,
            "source2_path": s2_path,
            "mode": self.mode,
        }
