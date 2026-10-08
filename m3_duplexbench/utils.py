import csv
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio

def get_available_path(path: str | Path) -> Path:
    path = Path(path)
    if not path.exists():
        return path

    i = 1
    while True:
        candidate = path.parent / f"{path.stem}_{i}{path.suffix}"
        if not candidate.exists():
            return candidate
        i += 1


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at line {line_idx}: {path}") from e


def write_jsonl(path: Path, records: Iterable[dict[str, Any]], avoid_overwrite: bool = False) -> None:
    if avoid_overwrite:
        path = get_available_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, obj: Any, avoid_overwrite: bool = False) -> None:
    if avoid_overwrite:
        path = get_available_path(path)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_csv(path: str | Path, rows: list[dict[str, Any]], avoid_overwrite: bool = False) -> None:
    if avoid_overwrite:
        path = get_available_path(path)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        with path.open("w", encoding="utf-8") as f:
            f.write("")
        return

    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_audio_info(audio_path: Path) -> tuple[int, float]:
    info = sf.info(str(audio_path))
    sample_rate = int(info.samplerate)
    duration = float(info.frames) / float(info.samplerate)
    return sample_rate, duration


def load_audio(audio_path: Path):
    audio, sample_rate = sf.read(str(audio_path), always_2d=True)
    audio = audio.astype(np.float32)
    return audio, int(sample_rate)


def load_audio_channel_first(audio_path: str | Path):
    audio, sample_rate = sf.read(str(audio_path))
    audio = audio.astype(np.float32)

    if audio.ndim == 1:
        audio = audio[None, :]
    elif audio.ndim == 2:
        audio = audio.T
    else:
        raise ValueError(f"Unsupported audio shape: {audio.shape}")

    return audio, int(sample_rate)


def resample_mono_np_lin(
    audio: np.ndarray,
    orig_sr: int,
    target_sr: int,
) -> np.ndarray:
    """Resample mono audio using torch interpolation."""
    if orig_sr == target_sr:
        return audio.astype(np.float32)

    if audio.ndim != 1:
        raise ValueError(f"Expected mono audio, got shape {audio.shape}")

    if len(audio) == 0:
        return audio.astype(np.float32)

    x = torch.from_numpy(audio.astype(np.float32))[None, None, :]
    target_len = int(round(len(audio) * float(target_sr) / float(orig_sr)))

    if target_len <= 0:
        return np.zeros(0, dtype=np.float32)

    y = F.interpolate(
        x,
        size=target_len,
        mode="linear",
        align_corners=False,
    )
    return y[0, 0].cpu().numpy().astype(np.float32)


def resample_mono_np(
    audio: np.ndarray,
    orig_sr: int,
    target_sr: int,
) -> np.ndarray:
    if orig_sr == target_sr:
        return audio.astype(np.float32)

    if len(audio) == 0:
        return audio.astype(np.float32)

    x = torch.from_numpy(audio).clone()
    y = torchaudio.transforms.Resample(
        orig_freq=orig_sr,
        new_freq=target_sr,
    )(x)

    return y.cpu().numpy().astype(np.float32)


def pad_or_trim(audio: np.ndarray, target_len: int) -> np.ndarray:
    if len(audio) == target_len:
        return audio.astype(np.float32)

    if len(audio) > target_len:
        return audio[:target_len].astype(np.float32)

    out = np.zeros(target_len, dtype=np.float32)
    out[: len(audio)] = audio.astype(np.float32)
    return out



def detect_non_silent_region(
    audio: np.ndarray,
    sr: int,
    frame_ms: float = 30.0,
    hop_ms: float = 10.0,
    threshold_db: float = -40.0,
    pad_sec: float = 0.2,
) -> tuple[float, float]:
    """Detect a non-silent region from mono audio by relative RMS energy."""
    if audio.ndim != 1:
        raise ValueError(f"Expected mono audio, got shape {audio.shape}")

    dur = len(audio) / float(sr)
    if len(audio) == 0:
        return 0.0, dur

    frame_len = max(1, int(round(frame_ms / 1000.0 * sr)))
    hop_len = max(1, int(round(hop_ms / 1000.0 * sr)))

    if len(audio) < frame_len:
        return 0.0, dur

    rms_list = []
    starts = []

    for start in range(0, len(audio) - frame_len + 1, hop_len):
        frame = audio[start : start + frame_len]
        rms = np.sqrt(np.mean(frame ** 2) + 1e-12)
        rms_list.append(rms)
        starts.append(start)

    rms_arr = np.asarray(rms_list, dtype=np.float32)
    starts_arr = np.asarray(starts, dtype=np.int64)

    peak = float(np.max(rms_arr))
    if peak <= 1e-8:
        return 0.0, dur

    rms_db = 20.0 * np.log10(rms_arr / peak + 1e-12)
    active = rms_db >= threshold_db

    if not np.any(active):
        return 0.0, dur

    active_idx = np.where(active)[0]
    first = int(active_idx[0])
    last = int(active_idx[-1])

    start_sec = starts_arr[first] / float(sr)
    end_sec = (starts_arr[last] + frame_len) / float(sr)

    start_sec = max(0.0, start_sec - pad_sec)
    end_sec = min(dur, end_sec + pad_sec)

    if end_sec <= start_sec:
        return 0.0, dur

    return start_sec, end_sec
