"""Local folder discovery; no browser uploads or directory publication."""

import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile


VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v", ".mpeg", ".mpg"}
GENERATED_NAME = re.compile(r"_[0-9a-f]{8}\.mkv$", re.IGNORECASE)


def discover_videos(folder_text: str) -> tuple[Path, ...]:
    """Recursively snapshot files once; deduplicate overlapping roots.

    Junctions/symlinks are not traversed outside a selected root. Generated
    folder-batch names are excluded so repeat runs do not enhance outputs.
    Permission and enumeration errors abort rather than silently omit videos.
    """
    lines = [line.strip().strip('"') for line in folder_text.splitlines() if line.strip()]
    if not lines:
        raise ValueError("Select or drop at least one local folder first.")
    roots = []
    for line in lines:
        root = Path(line).expanduser()
        if not root.is_absolute():
            raise ValueError(f"Folder paths must be absolute: {line}")
        root = root.resolve(strict=True)
        if not root.is_dir():
            raise ValueError(f"Not a folder: {root}")
        roots.append(root)
    files: set[Path] = set()
    visited: set[Path] = set()

    def fail(error: OSError) -> None:
        raise error

    for root in roots:
        for current, dirs, names in os.walk(root, followlinks=False, onerror=fail):
            directory = Path(current).resolve(strict=True)
            if directory in visited or not directory.is_relative_to(root):
                dirs[:] = []
                continue
            visited.add(directory)
            dirs[:] = sorted(name for name in dirs
                             if not (Path(current) / name).is_symlink()
                             and (Path(current) / name).resolve().is_relative_to(root))
            for name in names:
                path = Path(current) / name
                if (path.suffix.lower() not in VIDEO_EXTENSIONS or GENERATED_NAME.search(name)
                    or name.lower().endswith(".partial.mkv")):
                    continue
                if path.is_symlink():
                    continue
                resolved = path.resolve(strict=True)
                if resolved.is_relative_to(root) and resolved.is_file():
                    files.add(resolved)
    if not files:
        raise ValueError("No supported source videos found (previous *_xxxxxxxx.mkv outputs are excluded).")
    return tuple(sorted(files, key=lambda path: str(path).casefold()))


def select_local_folders() -> str | None:
    """Open a native Explorer-drop window on the machine running the app."""
    if os.name != "nt":
        raise RuntimeError("Local folder drag/drop requires Windows desktop.")
    executable = shutil.which("powershell.exe")
    if executable is None:
        raise RuntimeError("Windows PowerShell is required for the local folder window.")
    script = Path(__file__).resolve().parents[1] / "scripts" / "Select-VideoFolders.ps1"
    with tempfile.TemporaryDirectory(prefix="video-folder-picker-") as temp:
        result = Path(temp) / "folders.json"
        subprocess.run(
            [executable, "-NoProfile", "-STA", "-File", str(script), "-ResultPath", str(result)],
            check=True, capture_output=True,
        )
        if not result.exists():
            return None  # Closing/cancelling preserves the current UI selection.
        paths = json.loads(result.read_text(encoding="utf-8-sig"))
        if not isinstance(paths, list) or not all(isinstance(path, str) for path in paths):
            raise ValueError("Invalid folder selection result.")
        return "\n".join(paths)