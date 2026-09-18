#!/usr/bin/env python3
"""Build a self-contained devmem ZIP for temporary use in another repository."""

from __future__ import annotations

import argparse
import shutil
import tempfile
import zipapp
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOL_DIR = ".devmem-tool"

RUNNER = '''#!/usr/bin/env python3
"""Launch devmem without installing it into the target project."""
from __future__ import annotations

import os
from pathlib import Path
import sys

tool = Path(__file__).resolve().parent
environment = os.environ.copy()
environment["DEVMEM_DB"] = str(tool / "state" / "devmem.db")
os.execvpe(sys.executable, [sys.executable, str(tool / "devmem.pyz"), *sys.argv[1:]], environment)
'''

README = '''# Portable devmem

Run this from the root of the target project:

```bash
python3 .devmem-tool/run.py init
python3 .devmem-tool/run.py session start "describe the task"
python3 .devmem-tool/run.py run -- pytest -q
python3 .devmem-tool/run.py session end
python3 .devmem-tool/run.py handoff
```

No package is installed and all records are stored in
`.devmem-tool/state/`. To remove the tool and all its data:

```bash
rm -rf .devmem-tool
```

`run` and `verify` execute only commands supplied explicitly after `--`.
'''


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def build(output: Path) -> Path:
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="devmem-portable-") as temporary:
        temporary_root = Path(temporary)
        app = temporary_root / "app"
        shutil.copytree(ROOT / "src" / "devmem", app / "devmem", ignore=shutil.ignore_patterns("__pycache__"))
        rag = app / "vietnamese_rag"
        rag.mkdir()
        for name in ("__init__.py", "retrieval.py"):
            shutil.copy2(ROOT / "src" / "vietnamese_rag" / name, rag / name)
        _write(app / "__main__.py", "from devmem.cli import main\nraise SystemExit(main())\n")

        bundle = temporary_root / TOOL_DIR
        bundle.mkdir()
        zipapp.create_archive(app, bundle / "devmem.pyz", interpreter="/usr/bin/env python3")
        _write(bundle / "run.py", RUNNER)
        _write(bundle / "README.md", README)
        _write(bundle / "state" / ".gitignore", "*\n!.gitignore\n")

        with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for file in sorted(bundle.rglob("*")):
                if file.is_file():
                    archive.write(file, (Path(TOOL_DIR) / file.relative_to(bundle)).as_posix())
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "dist" / "devmem-portable.zip")
    args = parser.parse_args()
    print(f"Portable bundle created: {build(args.output)}")
    print(f"Extract {TOOL_DIR}/ into the target project, then run python3 {TOOL_DIR}/run.py")


if __name__ == "__main__":
    main()
