# Trauma audio pipelines for a local RTX 5090

Three independent Jupyter paths: **MOSS**, **VibeVoice ASR**, and **NeMo Nemotron-3-Diarization + Faster Whisper**. Each accepts a local audio/video file, extracts one audio stream to 16 kHz mono PCM, optionally applies FFmpeg `afftdn` denoising, and writes a common `speaker_attributed_turns.csv` format. The notebooks are new local editions based on the corresponding Colab notebooks in `../NoteBooks/`; the original upload cell and other Colab files are untouched.

## First: local video extraction and listening check (no GPU)

In Windows Git Bash, from this repository root:

```bash
source .venv-moss/Scripts/activate
python setup_env.py moss --prep-only
jupyter lab
```

The setup command installs only `requirements-prep.txt` (Jupyter, kernel, and bundled FFmpeg). On a fresh Windows clone without `.venv-moss`, install Python 3.12 first and run `py -3.12 setup_env.py moss --prep-only`; use the full Python path if the `py` launcher is unavailable. In VS Code, choose the **Trauma MOSS prep** kernel or `.venv-moss/Scripts/python.exe`.

Extract a video from its ZIP archive before setting it as input. Put the MP4's absolute path in an ignored `local_config.json` beside the notebooks, using forward slashes in JSON even on Windows:

```json
{"input_file": "D:/path/to/extracted/video.mp4"}
```

The current checkout already has an ignored local config pointing to the extracted selected clip. `01_MOSS.ipynb` defaults to `RUN_MODEL=False`. Set `DENOISE=False` or `True`, then **Run All**. Section 2 plays both the original and processed audio **inside the notebook**. `PREVIEW_START_SECONDS` and `PREVIEW_DURATION_SECONDS` choose a listening excerpt of up to 60 seconds; the full-recording players remain in the printed `audio_preview.html`. The HTML and WAV files are in `runs/`, which Git ignores. Change `DENOISE` and rerun to compare two separate run folders. This step needs no PyTorch or GPU.

## Later: MOSS model inference on Alex's GPU

Install Python 3.12, Git, and an NVIDIA driver on Alex's 5090 machine. System FFmpeg is optional: `requirements-common.txt` installs `imageio-ffmpeg`, whose bundled binary is used when `ffmpeg` is absent from `PATH`. Do not place recordings in this Git directory. RTX 5090 needs a CUDA 12.8+ PyTorch wheel; the notebook runs a real CUDA matrix multiplication before model loading. If the PyTorch command changes, use the [official installer](https://pytorch.org/get-started/locally/).

From this folder, create and install the **MOSS** environment (downloads are large):

```bash
# Linux / WSL2 on Alex's machine
python3.12 setup_env.py moss
source .venv-moss/bin/activate
jupyter lab
```

```powershell
# Windows PowerShell, if Alex uses native Windows and Python 3.12 is installed
py -3.12 setup_env.py moss
& .\.venv-moss\Scripts\Activate.ps1
jupyter lab
```

For **Git Bash on Windows**, use `Scripts`, not Linux's `bin`:

```bash
source .venv-moss/Scripts/activate
python --version                 # must say Python 3.12
python setup_env.py moss         # installs the full MOSS/CUDA environment
python -m pip check
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
jupyter lab
```

Run each line only after the previous installation/check succeeds. The final CUDA value must be `True` for GPU inference. On a fresh Windows clone with no `.venv-moss`, install Python 3.12 first and run `py -3.12 setup_env.py moss` (or use the full path to that Python executable). A Windows venv cannot be moved to Linux; recreate it on Alex's machine.

After full setup, choose the **Trauma MOSS** kernel and change `RUN_MODEL=True` to run inference. The last cell then renders a three-panel waveform, speaker activity, and transcript timeline inline in the notebook (no HTML file is written). VibeVoice and NeMo notebooks use the same visualization helper; NeMo's middle panel uses its raw diarization intervals. The current Windows checkout has the preparation dependencies and Plotly for visualization checks, but PyTorch and model dependencies have **not** been installed. A fresh clone on Alex's computer must run `setup_env.py moss` again; virtual environments are intentionally excluded from Git.

The root `requirements.txt` defaults to MOSS and includes `requirements-moss.txt`. The other routes need **separate** environments because [MOSS requires Transformers 5.x](https://github.com/OpenMOSS/MOSS-Transcribe-Diarize/blob/main/pyproject.toml) and [VibeVoice declares Transformers 4.x](https://github.com/microsoft/VibeVoice/blob/main/pyproject.toml). When ready, run `python3.12 setup_env.py vibe` or `python3.12 setup_env.py nemo` on Linux, or `py -3.12 setup_env.py vibe` on Windows. Their dependency lists are `requirements-vibe.txt` and `requirements-nemo.txt`; shared notebook dependencies are in `requirements-common.txt`. The setup script creates the venv, installs CUDA 12.8 PyTorch and the route requirements, checks dependency consistency, and registers a named Jupyter kernel. Upstream Git heads can change; after a successful run, save `pip freeze` outside this repository to record exact versions. NeMo may require Hugging Face access; use `hf auth login` on Alex's machine if loading returns 401/403.

Open the notebook with its matching kernel, put an extracted local MP4 path in ignored `local_config.json`, set `DENOISE` to `True` or `False`, and use **Run All**. `AUDIO_STREAM` selects the video audio track. NeMo defaults to English Faster Whisper ASR; edit `ASR_LANGUAGE` in its fourth section if needed. MOSS transcribes in `CHUNK_SECONDS` pieces (default 60 s, 5 s overlap) because it loops on long or silent audio; speaker IDs are per chunk (`C03:S01`), and turns from a chunk that hit `MAX_NEW_TOKENS` are flagged `chunk_hit_token_limit`. VibeVoice can hit `MAX_NEW_TOKENS` on long clips; check `run_summary.json` and `*_raw.txt`, then increase the limit or split the recording if truncated. Start with a short, authorized public clip to verify a new model environment.

## Denoising and outputs

**Git safety:** `runs/`, `models/`, media, transcripts and weights are ignored. Notebook outputs can embed recording audio and transcript text, so `setup_env.py` configures a Git filter (`strip_notebook_outputs.py`, see `.gitattributes`) that removes outputs whenever a notebook is staged; the copy on disk keeps them. On a clone where `setup_env.py` has not run, use **Edit → Clear Outputs of All Cells** before committing.

`DENOISE=False` is the baseline. `DENOISE=True` runs FFmpeg's [afftdn](https://ffmpeg.org/ffmpeg-filters.html#afftdn) FFT denoiser with `nr=12` dB by default. `NOISE_REDUCTION_DB` accepts 1–30; switch denoising off with `DENOISE=False`. This is a configurable baseline filter, not a validated trauma-room denoising method; compare both branches by listening and checking word and speaker accuracy. `audio_16k_mono_original.wav`, `audio_16k_mono_model.wav`, and `input_manifest.json` preserve the comparison and input hash. Other files include raw model output, a common CSV and readable transcript, and `run_summary.json`. NeMo also writes diarization CSV/RTTM, ASR JSON, and word alignment CSV.

Notebook audio players embed only the selected short excerpt in cell output. **Clear all notebook outputs before committing or uploading** a notebook that you ran on Aimbient data. Full audio, HTML timelines, transcripts, and model outputs stay under Git-ignored `runs/`.

Speaker IDs are anonymous and local to a recording. NeMo assigns words to speaker intervals by time overlap; overlap or weak matching is flagged `AMBIGUOUS`/`UNKNOWN`. These outputs do not establish a clinician's identity or role. The `.gitignore` excludes `runs/`, media, model weights, and environment files; check `git status` before uploading, and do not upload patient recordings or derived transcripts without project authorization.

## Collected failures and remedies

| Earlier failure / risk | Handling here |
| --- | --- |
| Colab Drive path and `google.colab` imports prevent local execution | Each notebook takes a local absolute `INPUT_FILE`; the Colab source remains unchanged. |
| NeMo import needed a newer source tree and `lhotse==2.0.0a6` | Separate NeMo environment with those installs before kernel launch. |
| Installing Faster Whisper after NeMo imports can replace loaded PyTorch | Install everything before starting the kernel, then reinstall the matching CUDA PyTorch wheel; ASR runs in a fresh child process. |
| CTranslate2 cannot find CUDA 12 cuBLAS/cuDNN 9 | Install NVIDIA pip wheels in the NeMo environment and set `LD_LIBRARY_PATH` for its child process. |
| PyAV `metadata_errors` incompatibility during Faster Whisper decoding | Pass validated WAV samples as a NumPy array to Faster Whisper, bypassing PyAV file decoding. |
| 5090 `sm_120` unsupported by older PyTorch wheels | Install CUDA 12.8+ PyTorch and execute a real CUDA kernel at notebook start. |
| Multiple audio streams or unsupported input format | Select `AUDIO_STREAM`; system `ffprobe` checks the track count when available, and FFmpeg normalizes to validated 16 kHz mono PCM. |
| Noisy/overlapping speech and empty or truncated parses | Optional reversible denoise branch; retain raw output and flag empty/ambiguous/truncated results for listening review. |

## Validation status

The JSON notebooks and Python sources are syntax checked locally. The selected 19-minute-59-second Aimbient MP4 was extracted from its ZIP and converted with the bundled FFmpeg both with denoising off and on. Both runs produced 16 kHz mono WAVs and local audio comparison pages. The revised MOSS notebook completed Run All with two inline audio players and model inference skipped. All three timeline routes generated HTML from synthetic speaker turns; actual model timelines and listening quality still need review after inference.

Upstream interfaces: [MOSS](https://github.com/OpenMOSS/MOSS-Transcribe-Diarize), [VibeVoice](https://github.com/microsoft/VibeVoice), [Nemotron-3 model card](https://huggingface.co/nvidia/Nemotron-3-Diarization), [Faster Whisper CUDA notes](https://github.com/SYSTRAN/faster-whisper).
