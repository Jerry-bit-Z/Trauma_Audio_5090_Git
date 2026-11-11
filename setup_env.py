"""Create one isolated notebook environment per model route.

Run with Python 3.12: python setup_env.py vibe|moss|nemo [--create-only]
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import venv
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ROUTES = {
    "moss": (".venv-moss", "requirements-moss.txt", "Trauma MOSS"),
    "vibe": (".venv-vibe", "requirements-vibe.txt", "Trauma VibeVoice"),
    "nemo": (".venv-nemo", "requirements-nemo.txt", "Trauma NeMo + FW"),
}


def run(args: list[str]) -> None:
    print("+", " ".join(args), flush=True)
    subprocess.run(args, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("route", choices=ROUTES)
    parser.add_argument("--create-only", action="store_true",
                        help="Create the venv without downloading large GPU/model dependencies")
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 12):
        raise SystemExit("Use Python 3.12 to create these environments")
    env_name, requirements, display = ROUTES[args.route]
    env_dir = ROOT / env_name
    env_python = env_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    # Some restricted Windows hosts deny writes to their default TEMP directory.
    local_tmp = ROOT / ".venv-tmp"
    local_tmp.mkdir(exist_ok=True)
    for name in ("TMP", "TEMP", "TMPDIR"):
        os.environ[name] = str(local_tmp)
    if not env_python.exists():
        venv.EnvBuilder(with_pip=True).create(env_dir)
    try:
        run([str(env_python), "-m", "pip", "--version"])
    except subprocess.CalledProcessError:
        run([str(env_python), "-m", "ensurepip", "--upgrade", "--default-pip"])
    if args.create_only:
        print(f"Created {env_dir}; model dependencies and kernel were not installed by this command")
        return
    run([str(env_python), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])
    # RTX 5090 needs a CUDA 12.8+ PyTorch wheel. Install it before other packages.
    torch_packages = ["torch", "torchvision", "torchaudio"]
    run([str(env_python), "-m", "pip", "install", "--upgrade", *torch_packages,
         "--index-url", "https://download.pytorch.org/whl/cu128"])
    run([str(env_python), "-m", "pip", "install", "-r", str(ROOT / requirements)])
    if args.route == "nemo":
        run([str(env_python), "-m", "pip", "install", "--no-deps", "--force-reinstall",
             "nemo_toolkit @ git+https://github.com/NVIDIA-NeMo/Speech.git@main"])
        # NeMo's resolver may replace torch; restore the matching CUDA build.
        run([str(env_python), "-m", "pip", "install", "--upgrade", *torch_packages,
             "--index-url", "https://download.pytorch.org/whl/cu128"])
    run([str(env_python), "-m", "pip", "check"])
    run([str(env_python), "-m", "ipykernel", "install", "--user",
         "--name", f"trauma-{args.route}", "--display-name", display])
    print(f"Ready: select the '{display}' kernel and run the matching notebook")


if __name__ == "__main__":
    main()
