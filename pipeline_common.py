"""Shared local audio preparation and result writing for the three notebooks."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
import subprocess
import wave
from datetime import datetime, timezone
from pathlib import Path


COLUMNS = ("start_seconds", "end_seconds", "model_speaker", "text", "review_reason")


def _run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def audio_info(path: Path) -> dict:
    with wave.open(str(path), "rb") as handle:
        info = {
            "sample_rate": handle.getframerate(),
            "channels": handle.getnchannels(),
            "sample_width_bytes": handle.getsampwidth(),
            "duration_seconds": handle.getnframes() / handle.getframerate(),
        }
    if (info["sample_rate"], info["channels"], info["sample_width_bytes"]) != (16000, 1, 2):
        raise ValueError(f"Expected 16 kHz mono 16-bit PCM WAV: {path}: {info}")
    if info["duration_seconds"] <= 0:
        raise ValueError(f"Empty audio: {path}")
    return info


def prepare_audio(source: str, output_root: str, route: str,
                  denoise: bool = False, audio_stream: int = 0,
                  noise_reduction_db: int = 12) -> tuple[Path, Path, dict]:
    """Extract one stream, optionally apply FFmpeg afftdn, and save provenance."""
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise RuntimeError("ffmpeg and ffprobe must be on PATH")
    source_path = Path(source).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    if audio_stream < 0 or not 1 <= noise_reduction_db <= 30:
        raise ValueError("audio_stream must be >= 0; noise_reduction_db must be 1..30")
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=index,codec_type",
         "-of", "json", str(source_path)],
        capture_output=True, text=True, check=True,
    )
    streams = [s for s in json.loads(probe.stdout).get("streams", [])
               if s.get("codec_type") == "audio"]
    if audio_stream >= len(streams):
        raise ValueError(f"audio_stream={audio_stream}; found {len(streams)} audio stream(s)")

    safe_stem = re.sub(r"[^A-Za-z0-9_-]+", "_", source_path.stem).strip("_")[:48] or "audio"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = Path(output_root).expanduser().resolve() / f"{safe_stem}_{route}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    original_wav = run_dir / "audio_16k_mono_original.wav"
    _run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
          "-i", str(source_path), "-map", f"0:a:{audio_stream}", "-vn",
          "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(original_wav)])
    info = audio_info(original_wav)
    model_wav = run_dir / "audio_16k_mono_model.wav"
    if denoise:
        _run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
              "-i", str(original_wav), "-af", f"afftdn=nr={noise_reduction_db}",
              "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(model_wav)])
    else:
        shutil.copy2(original_wav, model_wav)
    processed_info = audio_info(model_wav)
    if abs(processed_info["duration_seconds"] - info["duration_seconds"]) > 0.1:
        raise ValueError("Denoising changed audio duration by more than 0.1 s")
    digest = hashlib.sha256()
    with source_path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    sha256 = digest.hexdigest()
    manifest = {
        "source_path": str(source_path), "source_sha256": sha256,
        "route": route, "audio_stream": audio_stream,
        "denoise": denoise, "denoise_filter": f"afftdn=nr={noise_reduction_db}" if denoise else None,
        "original_audio": original_wav.name, "model_audio": model_wav.name,
        **info,
    }
    (run_dir / "input_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return run_dir, model_wav, manifest


def parse_time(value: object) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    fields = str(value).strip().removesuffix("s").split(":")
    if len(fields) > 3:
        raise ValueError(value)
    return sum(float(part) * 60 ** power for power, part in enumerate(reversed(fields)))


def save_turns(run_dir: Path, segments: list[dict], duration: float) -> list[dict]:
    """Save even empty results, flag questionable timestamps, never invent speakers."""
    rows = []
    for segment in segments:
        try:
            start = parse_time(segment["start_seconds"])
            end = parse_time(segment["end_seconds"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Unparseable segment: {segment}") from exc
        speaker = str(segment.get("model_speaker") or "UNKNOWN")
        body = str(segment.get("text") or "").strip()
        reasons = list(filter(None, str(segment.get("review_reason") or "").split(";")))
        if start < 0 or end > duration + 1:
            reasons.append("timestamp_outside_audio")
        if end <= start:
            reasons.append("zero_duration")
        if not body:
            reasons.append("empty_text")
        if speaker == "UNKNOWN":
            reasons.append("missing_speaker")
        rows.append(dict(zip(COLUMNS, (start, end, speaker, body, ";".join(dict.fromkeys(reasons))))))
    rows.sort(key=lambda row: (row["start_seconds"], row["end_seconds"]))
    with (run_dir / "speaker_attributed_turns.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    with (run_dir / "speaker_attributed_transcript.txt").open("w", encoding="utf-8") as handle:
        for row in rows:
            flag = f" [REVIEW: {row['review_reason']}]" if row["review_reason"] else ""
            handle.write(f"[{row['start_seconds']:.2f}–{row['end_seconds']:.2f}] "
                         f"{row['model_speaker']}{flag}: {row['text']}\n")
    return rows


def save_summary(run_dir: Path, **details: object) -> None:
    (run_dir / "run_summary.json").write_text(
        json.dumps(details, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )


def gpu_preflight() -> tuple[object, dict]:
    """Actually execute a small CUDA kernel to catch unsupported 5090 wheels."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Check NVIDIA driver and PyTorch CUDA wheel.")
    device = torch.device("cuda:0")
    try:
        probe = torch.ones((32, 32), device=device) @ torch.ones((32, 32), device=device)
        torch.cuda.synchronize()
        assert float(probe[0, 0]) == 32.0
    except Exception as exc:
        raise RuntimeError("CUDA kernel failed; install a PyTorch build supporting this GPU (5090: CUDA 12.8+).") from exc
    props = torch.cuda.get_device_properties(device)
    return device, {
        "gpu": props.name, "compute_capability": torch.cuda.get_device_capability(device),
        "vram_gib": round(props.total_memory / 2**30, 2),
        "torch": torch.__version__, "torch_cuda": torch.version.cuda,
    }
