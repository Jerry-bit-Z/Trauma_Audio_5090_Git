"""Git clean filter: remove notebook outputs before they are staged.

Notebook outputs here can embed patient audio (Audio players) and transcript text
(timeline figures), so they must never reach Git. Git pipes each staged .ipynb
through this script (stdin -> stdout); the file on disk keeps its outputs.
Configured by ``python setup_env.py`` (see .gitattributes); standard library only.
"""

import json
import sys


def strip(notebook: dict) -> dict:
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
        cell.get("metadata", {}).pop("execution", None)
        cell.pop("attachments", None)  # pasted images in markdown cells
    return notebook


def main() -> None:
    text = sys.stdin.buffer.read().decode("utf-8")
    if not text.strip():
        return
    notebook = strip(json.loads(text))
    # Match Jupyter's own formatting so diffs stay small.
    sys.stdout.buffer.write(
        (json.dumps(notebook, indent=1, ensure_ascii=False) + "\n").encode("utf-8")
    )


if __name__ == "__main__":
    main()
