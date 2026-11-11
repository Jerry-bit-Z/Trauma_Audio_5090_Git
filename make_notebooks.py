"""Rebuild the three local Jupyter notebooks from inspectable cell sources."""

import json
import hashlib
from pathlib import Path
from textwrap import indent


ROOT = Path(__file__).resolve().parent


def md(source):
    cell_id = "md-" + hashlib.sha1(source.encode("utf-8")).hexdigest()[:12]
    return {"cell_type": "markdown", "id": cell_id,
            "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source):
    cell_id = "code-" + hashlib.sha1(source.encode("utf-8")).hexdigest()[:12]
    return {"cell_type": "code", "id": cell_id,
            "execution_count": None, "metadata": {},
            "outputs": [], "source": source.splitlines(keepends=True)}


def write(name, cells):
    notebook = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3",
                 "language": "python", "name": "python3"},
                 "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    (ROOT / name).write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


INTRO = """# {title}

Run top to bottom in a **fresh local Jupyter kernel** with the matching route environment from README. Put one extracted audio/video file path in ignored `local_config.json` or set `TRAUMA_INPUT_FILE`. `DENOISE=True` applies FFmpeg `afftdn`; `False` copies the normalized WAV unchanged. Both versions and the input hash are saved under `runs/`. Section 2 provides short inline audio players and a full-recording HTML comparison.

Outputs are draft anonymous speaker labels, not verified identities or clinical roles. Check the audio before interpreting them. No input recordings or outputs belong in Git.
"""

SETTINGS = """from pathlib import Path
import sys
import json
import os
from pipeline_common import (prepare_audio, gpu_preflight, save_turns, save_summary,
                             write_audio_preview, make_inline_audio_clips)

# Prefer an environment variable or ignored local_config.json to keep private paths out of Git.
INPUT_FILE = os.environ.get("TRAUMA_INPUT_FILE", "")
if not INPUT_FILE and Path("local_config.json").is_file():
    INPUT_FILE = json.loads(Path("local_config.json").read_text(encoding="utf-8")).get("input_file", "")
OUTPUT_ROOT = "runs"
AUDIO_STREAM = 0
DENOISE = False  # True / False
NOISE_REDUCTION_DB = 12  # FFmpeg afftdn nr; 1..30; only used if DENOISE=True
PREVIEW_START_SECONDS = 0  # choose where to listen in the recording
PREVIEW_DURATION_SECONDS = 20  # 1..60; bounded notebook audio output

if not INPUT_FILE:
    raise ValueError("Set TRAUMA_INPUT_FILE or create ignored local_config.json with input_file")
print("Notebook Python:", sys.executable)
print("Route:", ROUTE, "| denoise:", DENOISE)
"""

PREP = """run_dir, model_wav, manifest = prepare_audio(
    INPUT_FILE, OUTPUT_ROOT, ROUTE, denoise=DENOISE,
    audio_stream=AUDIO_STREAM, noise_reduction_db=NOISE_REDUCTION_DB,
)
duration_seconds = manifest["duration_seconds"]
print(f"Model input: {model_wav} ({duration_seconds:.2f}s)")
print(f"Denoising: {manifest['denoise_filter'] or 'off'}")
print(f"Output: {run_dir}")
preview_path = write_audio_preview(run_dir, denoise=DENOISE)
print(f"Open this file in a browser to compare both audio players: {preview_path}")
preview_original, preview_model = make_inline_audio_clips(
    run_dir, start_seconds=PREVIEW_START_SECONDS,
    duration_seconds=PREVIEW_DURATION_SECONDS,
)
from IPython.display import Audio, Markdown, display
display(Markdown(f"**Original audio** — {PREVIEW_START_SECONDS}s to "
                 f"{PREVIEW_START_SECONDS + PREVIEW_DURATION_SECONDS}s"))
display(Audio(filename=str(preview_original)))
display(Markdown(f"**Model input** — denoise {'on' if DENOISE else 'off'}"))
display(Audio(filename=str(preview_model)))
"""

MOSS_INFER = """import json
import time
device, gpu_info = gpu_preflight()
print(gpu_info)
import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from moss_transcribe_diarize import parse_transcript
from moss_transcribe_diarize.inference_utils import build_transcription_messages, generate_transcription

MODEL_ID = "OpenMOSS-Team/MOSS-Transcribe-Diarize"
MAX_NEW_TOKENS = 8192
dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, trust_remote_code=True, dtype="auto", attn_implementation="sdpa",
).to(dtype=dtype).to(device).eval()
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
started = time.perf_counter()
result = generate_transcription(
    model, processor, build_transcription_messages(model_wav),
    max_new_tokens=MAX_NEW_TOKENS, do_sample=False, device=device, dtype=dtype,
)
elapsed = time.perf_counter() - started
raw_text = result["text"]
(run_dir / "moss_raw.txt").write_text(raw_text, encoding="utf-8")
(run_dir / "moss_raw.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
segments = [dict(start_seconds=s.start, end_seconds=s.end,
                 model_speaker=s.speaker, text=s.text)
            for s in parse_transcript(raw_text)]
rows = save_turns(run_dir, segments, duration_seconds)
save_summary(run_dir, model=MODEL_ID, **gpu_info, denoise=DENOISE,
             generated_tokens=result.get("generated_tokens"), max_new_tokens=MAX_NEW_TOKENS,
             inference_seconds=round(elapsed, 2), parsed_turns=len(rows),
             needs_review=not rows or result.get("generated_tokens", 0) >= MAX_NEW_TOKENS)
print(f"{len(rows)} turns saved in {run_dir}; inspect moss_raw.txt if empty or truncated")
rows[:10]
"""

VIBE_INFER = """import json
import time
device, gpu_info = gpu_preflight()
print(gpu_info)
import torch
from vibevoice.modular.modeling_vibevoice_asr import VibeVoiceASRForConditionalGeneration
from vibevoice.processor.vibevoice_asr_processor import VibeVoiceASRProcessor
from pipeline_common import parse_time

MODEL_ID = "microsoft/VibeVoice-ASR"
MAX_NEW_TOKENS = 8192
dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
processor = VibeVoiceASRProcessor.from_pretrained(MODEL_ID)
model = VibeVoiceASRForConditionalGeneration.from_pretrained(
    MODEL_ID, dtype=dtype, attn_implementation="sdpa", trust_remote_code=True,
).to(device).eval()
inputs = processor(audio=[str(model_wav)], sampling_rate=None,
                   return_tensors="pt", padding=True, add_generation_prompt=True)
inputs = {key: value.to(device) if isinstance(value, torch.Tensor) else value
          for key, value in inputs.items()}
prompt_length = inputs["input_ids"].shape[1]
started = time.perf_counter()
with torch.inference_mode():
    output_ids = model.generate(
        **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
        pad_token_id=processor.pad_id,
        eos_token_id=processor.tokenizer.eos_token_id,
    )
elapsed = time.perf_counter() - started
generated_ids = output_ids[0, prompt_length:]
raw_text = processor.decode(generated_ids, skip_special_tokens=True)
(run_dir / "vibevoice_raw.txt").write_text(raw_text, encoding="utf-8")
parsed = processor.post_process_transcription(raw_text)
(run_dir / "vibevoice_raw.json").write_text(json.dumps(parsed, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
segments = []
parse_failures = []
for index, segment in enumerate(parsed):
    try:
        segments.append(dict(start_seconds=parse_time(segment["start_time"]),
                             end_seconds=parse_time(segment["end_time"]),
                             model_speaker=str(segment.get("speaker_id") or "UNKNOWN"),
                             text=str(segment.get("text") or "")))
    except (KeyError, ValueError, TypeError) as exc:
        parse_failures.append({"index": index, "error": str(exc), "raw": segment})
(run_dir / "parse_failures.json").write_text(json.dumps(parse_failures, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
rows = save_turns(run_dir, segments, duration_seconds)
save_summary(run_dir, model=MODEL_ID, **gpu_info, denoise=DENOISE,
             generated_tokens=len(generated_ids), max_new_tokens=MAX_NEW_TOKENS,
             inference_seconds=round(elapsed, 2), parsed_turns=len(rows),
             parse_failures=len(parse_failures),
             needs_review=not rows or bool(parse_failures) or len(generated_ids) >= MAX_NEW_TOKENS)
print(f"{len(rows)} turns saved in {run_dir}; parse failures: {len(parse_failures)}")
rows[:10]
"""

NEMO_INFER = """import gc
import json
import os
import subprocess
import sys
import time
device, gpu_info = gpu_preflight()
print(gpu_info)
import torch
from nemo.collections.asr.models import SortformerEncLabelModel

MODEL_ID = "nvidia/Nemotron-3-Diarization"
diar_model = SortformerEncLabelModel.from_pretrained(MODEL_ID).to(device).eval()
diar_model.sortformer_modules.spkcache_len = 264
diar_model.sortformer_modules.fifo_len = 40
diar_model.sortformer_modules.chunk_len = 340
diar_model.sortformer_modules.chunk_right_context = 40
diar_model.sortformer_modules.spkcache_update_period = 300
diar_model._check_streaming_parameters()
started = time.perf_counter()
predicted = diar_model.diarize(audio=[str(model_wav)], batch_size=1)[0]
diar_seconds = time.perf_counter() - started
diar_rows = []
for item in predicted:
    start, end, speaker = item.split()
    diar_rows.append(dict(start_seconds=float(start), end_seconds=float(end), speaker=speaker))
import csv
with (run_dir / "diarization.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=["start_seconds", "end_seconds", "speaker"])
    writer.writeheader()
    writer.writerows(diar_rows)
with (run_dir / "diarization.rttm").open("w", encoding="utf-8") as handle:
    for row in diar_rows:
        handle.write(f"SPEAKER audio 1 {row['start_seconds']:.3f} "
                     f"{row['end_seconds'] - row['start_seconds']:.3f} <NA> <NA> "
                     f"{row['speaker']} <NA> <NA>\\n")
del diar_model
gc.collect()
torch.cuda.empty_cache()
print(f"Saved {len(diar_rows)} diarization intervals in {diar_seconds:.1f}s")
"""

ASR_INFER = """# Run CTranslate2 in a child process so CUDA libraries are visible before import.
ASR_MODEL = "large-v3"
ASR_LANGUAGE = "en"  # use None for automatic language detection
config = dict(wav_path=str(model_wav), output_json=str(run_dir / "asr_raw.json"),
              model=ASR_MODEL, language=ASR_LANGUAGE)
config_path = run_dir / "asr_config.json"
config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
env = os.environ.copy()
if sys.platform == "linux":
    try:
        import nvidia.cublas
        import nvidia.cudnn
        from pathlib import Path
        lib_dirs = [Path(nvidia.cublas.__path__[0]) / "lib",
                    Path(nvidia.cudnn.__path__[0]) / "lib"]
        env["LD_LIBRARY_PATH"] = ":".join(map(str, lib_dirs)) + ":" + env.get("LD_LIBRARY_PATH", "")
    except ImportError:
        print("Using system cuBLAS/cuDNN; install the NVIDIA pip wheels if CTranslate2 cannot find them")
started = time.perf_counter()
subprocess.run([sys.executable, str(Path("run_faster_whisper.py").resolve()), str(config_path)],
               env=env, check=True)
asr_seconds = time.perf_counter() - started
records = json.loads((run_dir / "asr_raw.json").read_text(encoding="utf-8"))
print(f"Saved {len(records)} ASR segments in {asr_seconds:.1f}s")
"""

ALIGN = """# Assign each ASR word by time overlap. Ambiguous/unknown labels stay explicit.
from pipeline_common import save_turns
word_rows = []
for segment_index, record in enumerate(records):
    for word in record.get("words", []):
        start, end = word.get("start"), word.get("end")
        if start is None or end is None or end <= start:
            continue
        overlap = {}
        for interval in diar_rows:
            amount = max(0.0, min(end, interval["end_seconds"]) - max(start, interval["start_seconds"]))
            if amount:
                name = interval["speaker"]
                overlap[name] = overlap.get(name, 0.0) + amount
        ranked = sorted(overlap.items(), key=lambda pair: pair[1], reverse=True)
        length = end - start
        if not ranked or ranked[0][1] < 0.5 * length:
            speaker, reason = "UNKNOWN", "no_clear_time_match"
        elif len(ranked) > 1 and ranked[1][1] >= 0.25 * length:
            speaker, reason = "AMBIGUOUS", "multiple_active_speakers"
        else:
            speaker, reason = ranked[0][0], ""
        word_rows.append(dict(asr_segment=segment_index, start_seconds=start,
                              end_seconds=end, word=word["word"],
                              model_speaker=speaker, review_reason=reason))
with (run_dir / "speaker_attributed_words.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=["asr_segment", "start_seconds", "end_seconds",
                                                "word", "model_speaker", "review_reason"])
    writer.writeheader()
    writer.writerows(word_rows)
turns = []
for word in word_rows:
    if (turns and turns[-1]["asr_segment"] == word["asr_segment"]
            and turns[-1]["model_speaker"] == word["model_speaker"]
            and turns[-1]["review_reason"] == word["review_reason"]):
        turns[-1]["end_seconds"] = word["end_seconds"]
        turns[-1]["text"] += word["word"]
    else:
        turns.append(dict(asr_segment=word["asr_segment"], start_seconds=word["start_seconds"],
                          end_seconds=word["end_seconds"], model_speaker=word["model_speaker"],
                          text=word["word"], review_reason=word["review_reason"]))
rows = save_turns(run_dir, turns, duration_seconds)
save_summary(run_dir, diarization_model=MODEL_ID, asr_model=ASR_MODEL, **gpu_info,
             denoise=DENOISE, diarization_seconds=round(diar_seconds, 2),
             asr_seconds=round(asr_seconds, 2), diarization_intervals=len(diar_rows),
             asr_segments=len(records), attributed_turns=len(rows),
             needs_review=not rows or any(row["review_reason"] for row in rows))
print(f"{len(rows)} turns saved in {run_dir}; review flags are in the CSV")
rows[:10]
"""

TIMELINE_MOSS = """if RUN_MODEL:
    from pipeline_common import render_speaker_timeline
    fig, timeline_path = render_speaker_timeline(
        run_dir, rows, route="moss", source_name=Path(INPUT_FILE).name,
    )
    fig.show(config={"scrollZoom": True})
    print("Interactive timeline:", timeline_path)
else:
    print("Timeline skipped: set RUN_MODEL=True after installing MOSS and rerun the notebook.")
"""

TIMELINE_VIBE = """from pipeline_common import render_speaker_timeline
fig, timeline_path = render_speaker_timeline(
    run_dir, rows, route="vibevoice", source_name=Path(INPUT_FILE).name,
)
fig.show(config={"scrollZoom": True})
print("Interactive timeline:", timeline_path)
"""

TIMELINE_NEMO = """from pipeline_common import render_speaker_timeline
fig, timeline_path = render_speaker_timeline(
    run_dir, rows, route="nemo_fw", source_name=Path(INPUT_FILE).name,
    diarization=diar_rows,
)
fig.show(config={"scrollZoom": True})
print("Interactive timeline:", timeline_path)
"""


MOSS_SETTINGS = SETTINGS.replace(
    "DENOISE = False  # True / False",
    "DENOISE = False  # True / False\nRUN_MODEL = False  # leave False for video/audio/denoising checks",
)

write("01_MOSS.ipynb", [
    md("""# MOSS video/audio preparation and optional inference

Use the local `Trauma MOSS prep` kernel or `.venv-moss/Scripts/python.exe` on Windows. With `RUN_MODEL=False` (default), **Run All** extracts audio, optionally denoises it, and shows short original/processed players in section 2. It also creates `audio_preview.html` with full-recording players. This path needs `imageio-ffmpeg`, not PyTorch or a GPU. Set `RUN_MODEL=True` later when the full MOSS environment is installed; the last cell then renders and saves the three-panel timeline.

Put a local absolute MP4 path in ignored `local_config.json` as `{"input_file": "..."}` or set `TRAUMA_INPUT_FILE`. Keep source recordings and `runs/` out of Git. Speaker labels from later inference remain anonymous drafts.
"""),
    md("## 1. Select local input and preprocessing options"), code('ROUTE = "moss"\n' + MOSS_SETTINGS),
    md("## 2. Convert video/audio to WAV and create a listening preview"), code(PREP),
    md("## 3. Optional MOSS inference"),
    code("if RUN_MODEL:\n" + indent(MOSS_INFER, "    ") +
         "else:\n    print('Model inference skipped. Open audio_preview.html to listen first.')\n"),
    md("## 4. Three-panel waveform, speaker activity, and transcript timeline"),
    code(TIMELINE_MOSS),
])
write("02_VibeVoice.ipynb", [
    md(INTRO.format(title="VibeVoice ASR and speaker turns")),
    md("## 1. Settings and GPU check"), code('ROUTE = "vibevoice"\n' + SETTINGS),
    md("## 2. Convert video/audio to WAV; optionally denoise"), code(PREP),
    md("## 3. Infer and save results"), code(VIBE_INFER),
    md("## 4. Three-panel waveform, speaker activity, and transcript timeline"),
    code(TIMELINE_VIBE),
])
write("03_NeMo_FasterWhisper.ipynb", [
    md(INTRO.format(title="NeMo diarization and Faster Whisper ASR")),
    md("## 1. Settings and GPU check"), code('ROUTE = "nemo_fw"\n' + SETTINGS),
    md("## 2. Convert video/audio to WAV; optionally denoise"), code(PREP),
    md("## 3. NeMo diarization"), code(NEMO_INFER),
    md("## 4. Faster Whisper in a clean process"), code(ASR_INFER),
    md("## 5. Time align words and save results"), code(ALIGN),
    md("## 6. Three-panel waveform, diarization, and transcript timeline"),
    code(TIMELINE_NEMO),
])
