"""Shared local audio preparation and result writing for the three notebooks."""

from __future__ import annotations

import csv
import hashlib
import html
import json
import math
import re
import shutil
import subprocess
import wave
from datetime import datetime, timezone
from pathlib import Path


COLUMNS = ("start_seconds", "end_seconds", "model_speaker", "text", "review_reason")


def _run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def ffmpeg_tools() -> tuple[str, str | None]:
    """Prefer a system install; fall back to imageio-ffmpeg's bundled binary."""
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if ffmpeg:
        return ffmpeg, ffprobe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe(), None
    except (ImportError, RuntimeError) as exc:
        raise RuntimeError(
            "FFmpeg unavailable. Install the route's requirements (imageio-ffmpeg), "
            "or install system FFmpeg; then select that venv as the notebook kernel."
        ) from exc


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
                  noise_reduction_db: int = 12, denoise_method: str = "noisereduce",
                  denoise_strength: float = 0.8) -> tuple[Path, Path, dict]:
    """Extract one stream, optionally denoise, and save provenance."""
    ffmpeg, ffprobe = ffmpeg_tools()
    source_path = Path(source).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    if audio_stream < 0 or not 1 <= noise_reduction_db <= 30:
        raise ValueError("audio_stream must be >= 0; noise_reduction_db must be 1..30")
    if denoise_method not in {"noisereduce", "afftdn"}:
        raise ValueError("denoise_method must be 'noisereduce' or 'afftdn'")
    if not 0 <= denoise_strength <= 1:
        raise ValueError("denoise_strength must be 0..1")
    if ffprobe:
        probe = subprocess.run(
            [ffprobe, "-v", "error", "-show_entries", "stream=index,codec_type",
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
    _run([ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
          "-i", str(source_path), "-map", f"0:a:{audio_stream}", "-vn",
          "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(original_wav)])
    info = audio_info(original_wav)
    model_wav = run_dir / "audio_16k_mono_model.wav"
    denoise_filter = None
    if denoise and denoise_method == "afftdn":
        denoise_filter = f"afftdn=nr={noise_reduction_db}"
        _run([ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
              "-i", str(original_wav), "-af", denoise_filter,
              "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(model_wav)])
    elif denoise and denoise_method == "noisereduce":
        try:
            import noisereduce as nr
            import numpy as np
        except ImportError as exc:
            raise RuntimeError("Install requirements-prep.txt in the selected notebook kernel") from exc
        denoise_filter = f"noisereduce nonstationary prop_decrease={denoise_strength} n_fft=512"
        with wave.open(str(original_wav), "rb") as reader:
            samples = np.frombuffer(reader.readframes(reader.getnframes()), dtype="<i2")
        signal = samples.astype(np.float32) / 32768.0
        processed = nr.reduce_noise(
            y=signal, sr=16000, stationary=False, prop_decrease=denoise_strength,
            n_fft=512, chunk_size=16000 * 10, padding=16000 * 2, n_jobs=1,
        )
        pcm = np.rint(np.clip(processed, -1.0, 32767 / 32768) * 32768).astype("<i2")
        with wave.open(str(model_wav), "wb") as writer:
            writer.setnchannels(1)
            writer.setsampwidth(2)
            writer.setframerate(16000)
            writer.writeframes(pcm.tobytes())
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
        "ffmpeg_executable": ffmpeg,
        "denoise": denoise, "denoise_method": denoise_method if denoise else None,
        "denoise_strength": denoise_strength if denoise and denoise_method == "noisereduce" else None,
        "denoise_filter": denoise_filter,
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


def write_audio_preview(run_dir: Path, *, denoise: bool, method: str = "noisereduce") -> Path:
    """Create an offline comparison page without embedding audio in the notebook."""
    original = run_dir / "audio_16k_mono_original.wav"
    processed = run_dir / "audio_16k_mono_model.wav"
    audio_info(original)
    audio_info(processed)
    manifest = json.loads((run_dir / "input_manifest.json").read_text(encoding="utf-8"))
    label = (manifest.get("denoise_filter") or f"{method} enabled") if denoise else "Denoising off (same audio)"
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Audio comparison</title>
<style>body{{font:16px system-ui;max-width:800px;margin:2rem auto;padding:0 1rem}}
audio{{width:100%}}section{{margin:2rem 0}}</style></head><body>
<h1>Audio extraction and denoising check</h1>
<p>{html.escape(label)}. Listen to the same time position in both players.</p>
<section><h2>Extracted original</h2><audio controls preload="metadata" src="{original.name}"></audio></section>
<section><h2>Model input</h2><audio controls preload="metadata" src="{processed.name}"></audio></section>
<p>Both WAV files are stored beside this page. Do not add this run folder to Git.</p>
</body></html>"""
    destination = run_dir / "audio_preview.html"
    destination.write_text(page, encoding="utf-8")
    return destination


def make_inline_audio_clips(run_dir: Path, start_seconds: float = 0,
                            duration_seconds: float = 20) -> tuple[Path, Path]:
    """Bound notebook-embedded audio to a short selected excerpt."""
    if start_seconds < 0 or not 0 < duration_seconds <= 60:
        raise ValueError("Preview start must be >= 0 and duration must be 0..60 seconds")
    paths = []
    for source_name, target_name in (
        ("audio_16k_mono_original.wav", "preview_original.wav"),
        ("audio_16k_mono_model.wav", "preview_model.wav"),
    ):
        source = run_dir / source_name
        info = audio_info(source)
        if start_seconds >= info["duration_seconds"]:
            raise ValueError(f"Preview start exceeds audio duration: {info['duration_seconds']:.2f}s")
        target = run_dir / target_name
        with wave.open(str(source), "rb") as reader, wave.open(str(target), "wb") as writer:
            writer.setparams(reader.getparams())
            reader.setpos(int(start_seconds * info["sample_rate"]))
            remaining = min(math.ceil(duration_seconds * info["sample_rate"]),
                            reader.getnframes() - reader.tell())
            writer.writeframes(reader.readframes(remaining))
        paths.append(target)
    return paths[0], paths[1]


def split_wav_chunks(wav_path: Path, out_dir: Path, chunk_seconds: float = 180,
                     overlap_seconds: float = 10) -> list[dict]:
    """Cut a PCM WAV into consecutive chunks for models that loop on long inputs.

    Chunk ``i`` covers ``[i * chunk_seconds, (i + 1) * chunk_seconds + overlap_seconds)``.
    Its "owned" window is ``[i * chunk_seconds, (i + 1) * chunk_seconds)``: when merging,
    keep only turns that start inside the owned window, so speech crossing a boundary is
    transcribed whole by the earlier chunk and not duplicated by the next one.
    """
    if chunk_seconds <= 0 or overlap_seconds < 0:
        raise ValueError("chunk_seconds must be > 0 and overlap_seconds >= 0")
    out_dir.mkdir(parents=True, exist_ok=True)
    chunks = []
    with wave.open(str(wav_path), "rb") as reader:
        rate, total = reader.getframerate(), reader.getnframes()
        step, extra = round(chunk_seconds * rate), round(overlap_seconds * rate)
        for index, first in enumerate(range(0, total, step)):
            count = min(step + extra, total - first)
            reader.setpos(first)
            target = out_dir / f"chunk_{index:03d}.wav"
            with wave.open(str(target), "wb") as writer:
                writer.setparams(reader.getparams())
                writer.writeframes(reader.readframes(count))
            chunks.append(dict(index=index, path=target, offset=first / rate,
                               owned_end=min(first + step, total) / rate,
                               end=(first + count) / rate))
    return chunks


def render_speaker_timeline(run_dir: Path, rows: list[dict], route: str,
                            source_name: str, diarization: list[dict] | None = None):
    """Build the same three-panel Plotly timeline for all model routes (shown inline, not saved)."""
    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    original = run_dir / "audio_16k_mono_original.wav"
    model_audio = run_dir / "audio_16k_mono_model.wav"
    info = audio_info(original)
    audio_info(model_audio)
    duration = info["duration_seconds"]
    total_frames = round(duration * info["sample_rate"])
    bin_frames = max(1, math.ceil(total_frames / 6000))
    times, low, high = [], [], []
    with wave.open(str(original), "rb") as reader:
        frame_index = 0
        while chunk := reader.readframes(bin_frames):
            values = np.frombuffer(chunk, dtype="<i2")
            if values.size:
                times.append((frame_index + values.size / 2) / info["sample_rate"])
                low.append(float(values.min()) / 32768)
                high.append(float(values.max()) / 32768)
                frame_index += values.size

    activities = diarization if diarization is not None else rows
    speakers = sorted({str(row.get("speaker", row.get("model_speaker", "UNKNOWN")))
                       for row in activities} |
                      {str(row.get("model_speaker", "UNKNOWN")) for row in rows})
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
               "#9467bd", "#8c564b", "#e377c2", "#17becf"]
    colors = {speaker: palette[index % len(palette)]
              for index, speaker in enumerate(speakers)}
    colors.update({"UNKNOWN": "#777777", "AMBIGUOUS": "#d89019"})
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.20, 0.45, 0.35], vertical_spacing=0.07,
        subplot_titles=("Original audio waveform", f"{route} speaker activity",
                        "Speaker-attributed transcript"),
    )
    fig.add_trace(go.Scatter(
        x=times, y=high, mode="lines", line=dict(color="#666666", width=0.7),
        name="Audio", hoverinfo="skip",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=times, y=low, mode="lines", fill="tonexty",
        fillcolor="rgba(100,100,100,0.25)", line=dict(color="#666666", width=0.7),
        showlegend=False, hovertemplate="%{x:.2f} s<extra></extra>",
    ), row=1, col=1)

    for speaker in speakers:
        activity = [row for row in activities
                    if str(row.get("speaker", row.get("model_speaker", "UNKNOWN"))) == speaker]
        if activity:
            custom = [[float(row["start_seconds"]), float(row["end_seconds"])]
                      for row in activity]
            fig.add_trace(go.Bar(
                x=[max(0.0, end - start) for start, end in custom],
                base=[start for start, _ in custom], y=[speaker] * len(activity),
                orientation="h", width=0.7, marker_color=colors[speaker],
                name=speaker, customdata=custom,
                hovertemplate=(f"<b>{html.escape(speaker)}</b><br>"
                               "%{customdata[0]:.2f}–%{customdata[1]:.2f} s<extra></extra>"),
            ), row=2, col=1)
        turns = [row for row in rows if str(row.get("model_speaker", "UNKNOWN")) == speaker]
        if turns:
            custom = [[float(row["start_seconds"]), float(row["end_seconds"]),
                       html.escape(str(row.get("text", ""))),
                       html.escape(str(row.get("review_reason", "")))] for row in turns]
            fig.add_trace(go.Bar(
                x=[max(0.0, item[1] - item[0]) for item in custom],
                base=[item[0] for item in custom], y=["Transcript"] * len(turns),
                orientation="h", width=0.7,
                marker=dict(color=colors[speaker],
                            line=dict(color=["#e6a23c" if item[3] else "white" for item in custom],
                                      width=2)),
                name=f"Text: {speaker}", showlegend=False, customdata=custom,
                hovertemplate=(f"<b>{html.escape(speaker)}</b><br>"
                               "%{customdata[0]:.2f}–%{customdata[1]:.2f} s"
                               "<br>%{customdata[2]}<br>Review: %{customdata[3]}<extra></extra>"),
            ), row=3, col=1)
    fig.update_layout(
        title=f"{html.escape(source_name)}: {html.escape(route)} transcript + diarization",
        height=850, barmode="overlay", hovermode="closest", legend_title="Model speaker",
    )
    fig.update_yaxes(title_text="Amplitude", row=1, col=1)
    fig.update_yaxes(title_text="Speaker ID", row=2, col=1)
    fig.update_xaxes(title_text="Time (seconds)", range=[0, duration], row=3, col=1)
    if not rows:
        fig.add_annotation(text="No parsed transcript turns; inspect raw model output",
                           x=0.5, y=0.12, xref="paper", yref="paper", showarrow=False)
    return fig


def show_figure(fig, height: int | None = None) -> None:
    """Display a Plotly figure inline in any Jupyter front end, without writing files.

    The figure goes into an iframe ``srcdoc`` that loads plotly.js from the CDN, so it
    does not need the jupyterlab-plotly extension in the Jupyter server environment.
    """
    from IPython.display import HTML, display
    page = fig.to_html(full_html=True, include_plotlyjs="cdn",
                       config={"scrollZoom": True, "responsive": True})
    height = height or int(fig.layout.height or 600) + 30
    import warnings
    with warnings.catch_warnings():  # IPython suggests IFrame, which needs a file/URL
        warnings.simplefilter("ignore", UserWarning)
        display(HTML(f'<iframe srcdoc="{html.escape(page, quote=True)}" '
                     f'style="width:100%;height:{height}px;border:0"></iframe>'))


def local_model_dir(repo_id: str, models_root: str = "models") -> Path:
    """Return a local copy of a Hugging Face model, downloading it once if missing.

    Loading from this folder makes no Hub requests, so runs work offline and
    without HF_TOKEN. The folder is ignored by Git.
    """
    target = Path(models_root) / repo_id.split("/")[-1]
    if not (target / "config.json").is_file():
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id, local_dir=target)
    return target.resolve()


def gpu_preflight() -> tuple[object, dict]:
    """Actually execute a small CUDA kernel to catch unsupported 5090 wheels."""
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "PyTorch is missing from this notebook kernel. Run setup_env.py for this route "
            "and select its Trauma kernel in Jupyter/VS Code."
        ) from exc

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
