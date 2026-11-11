# Trauma audio pipelines for a local RTX 5090

Three independent Jupyter paths: **MOSS**, **VibeVoice ASR**, and **NeMo Nemotron-3-Diarization + Faster Whisper**. Each accepts a local audio/video file, extracts one audio stream to 16 kHz mono PCM, optionally applies FFmpeg `afftdn` denoising, and writes a common `speaker_attributed_turns.csv` format. The notebooks are new local editions based on the corresponding Colab notebooks in `../NoteBooks/`; the original upload cell and other Colab files are untouched.

## Before running on Alex's machine

Use Linux, Python 3.12, Git, FFmpeg (`ffmpeg` and `ffprobe` on `PATH`), a recent NVIDIA driver, and a separate virtual environment **for each route**. Do not place recordings in this Git directory. PyTorch for RTX 5090 needs a CUDA 12.8+ build; the notebook runs a real CUDA matrix multiplication before loading a model. Consult the [PyTorch installer](https://pytorch.org/get-started/locally/) if the wheel command changes.

From this directory, create one environment and kernel per route. `python3.12 -m venv` can be replaced with `uv venv --python 3.12` if available.

```bash
sudo apt-get update && sudo apt-get install -y ffmpeg git
# Install Python 3.12 and its venv package with the system package manager if absent.
python3.12 -m venv .venv-moss
source .venv-moss/bin/activate
python -m pip install -U pip
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
python -m pip install 'git+https://github.com/OpenMOSS/MOSS-Transcribe-Diarize.git' jupyterlab ipykernel
python -m ipykernel install --user --name trauma-moss --display-name 'Trauma MOSS'
deactivate
```

```bash
python3.12 -m venv .venv-vibe
source .venv-vibe/bin/activate
python -m pip install -U pip
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
python -m pip install 'git+https://github.com/microsoft/VibeVoice.git' jupyterlab ipykernel
python -m ipykernel install --user --name trauma-vibe --display-name 'Trauma VibeVoice'
deactivate
```

```bash
python3.12 -m venv .venv-nemo
source .venv-nemo/bin/activate
python -m pip install -U pip Cython packaging
python -m pip install 'nemo_toolkit[asr]'
python -m pip install --no-deps --force-reinstall 'nemo_toolkit @ git+https://github.com/NVIDIA-NeMo/Speech.git@main'
python -m pip install 'lhotse==2.0.0a6' faster-whisper 'nvidia-cublas-cu12' 'nvidia-cudnn-cu12==9.*' jupyterlab ipykernel
python -m pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
python -m ipykernel install --user --name trauma-nemo --display-name 'Trauma NeMo + FW'
deactivate
```

These commands use upstream Git heads because the source notebooks used them; dependency APIs may change. Run `python -m pip check` in each environment before starting its kernel. Save `python -m pip freeze > environment.txt` **outside this repo** after a successful run to capture the exact environment. If the NeMo install reports incompatible requirements after the final torch install, resolve those in the NeMo environment before running the notebook. The Nemotron checkpoint may require Hugging Face access; authenticate with `hf auth login` on Alex's machine if model loading returns 401/403, without putting a token in the notebook. Start Jupyter from this directory so it can import `pipeline_common.py` and find `run_faster_whisper.py`:

```bash
source .venv-moss/bin/activate  # or the route's environment
jupyter lab
```

Open `01_MOSS.ipynb`, `02_VibeVoice.ipynb`, or `03_NeMo_FasterWhisper.ipynb` with its matching kernel. Set `INPUT_FILE` to an absolute path outside the repo, `DENOISE` to `True` or `False`, and use **Run All**. NeMo defaults to English Faster Whisper ASR; edit `ASR_LANGUAGE` in its fourth section if needed. MOSS and VibeVoice can hit `MAX_NEW_TOKENS` on long clips; check `run_summary.json` and `*_raw.txt`, then increase the limit or split the recording if truncated. Start with a short, authorized public clip to verify the environment.

## Denoising and outputs

`DENOISE=False` is the baseline. `DENOISE=True` runs FFmpeg's [afftdn](https://ffmpeg.org/ffmpeg-filters.html#afftdn) FFT denoiser with `nr=12` dB by default. `NOISE_REDUCTION_DB` accepts 1–30; switch denoising off with `DENOISE=False`. This is a configurable baseline filter, not a validated trauma-room denoising method; compare both branches by listening and checking word and speaker accuracy. `audio_16k_mono_original.wav`, `audio_16k_mono_model.wav`, and `input_manifest.json` preserve the comparison and input hash. Other files include raw model output, a common CSV and readable transcript, and `run_summary.json`. NeMo also writes diarization CSV/RTTM, ASR JSON, and word alignment CSV.

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
| Multiple audio streams or unsupported input format | Probe streams, select `AUDIO_STREAM`, and let FFmpeg normalize to validated 16 kHz mono PCM. |
| Noisy/overlapping speech and empty or truncated parses | Optional reversible denoise branch; retain raw output and flag empty/ambiguous/truncated results for listening review. |

## Validation status

The JSON notebooks and Python sources are generated and syntax checked locally. FFmpeg is unavailable in this workspace, so video extraction and denoising have not been executed here. Model downloads and 5090 inference require Alex's machine; until a short clip completes through all three kernels, these are prepared handoff notebooks, not confirmed one-pass GPU runs.

Upstream interfaces: [MOSS](https://github.com/OpenMOSS/MOSS-Transcribe-Diarize), [VibeVoice](https://github.com/microsoft/VibeVoice), [Nemotron-3 model card](https://huggingface.co/nvidia/Nemotron-3-Diarization), [Faster Whisper CUDA notes](https://github.com/SYSTRAN/faster-whisper).
