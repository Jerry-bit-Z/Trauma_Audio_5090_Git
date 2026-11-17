"""Run CTranslate2 in a fresh process after NeMo releases the GPU."""

import json
import sys
import wave
from pathlib import Path

import numpy as np
from faster_whisper import WhisperModel


config = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
with wave.open(config["wav_path"], "rb") as handle:
    assert (handle.getframerate(), handle.getnchannels(), handle.getsampwidth()) == (16000, 1, 2)
    audio = np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2").astype(np.float32) / 32768
model = WhisperModel(config["model"], device="cuda", compute_type="float16")
segments, info = model.transcribe(
    audio, language=config.get("language") or None, beam_size=5,
    word_timestamps=True,
    # Conditioning on earlier text makes Whisper loop through long silences
    # (one phrase repeated for minutes). Silero VAD drops most quiet, distant
    # trauma-bay speech, so keep it off.
    vad_filter=False, condition_on_previous_text=False,
)
records = [
    {"start": segment.start, "end": segment.end, "text": segment.text,
     "words": [{"start": word.start, "end": word.end, "word": word.word}
               for word in (segment.words or [])]}
    for segment in segments
]
Path(config["output_json"]).write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"Saved {len(records)} ASR segments; language={info.language}")
