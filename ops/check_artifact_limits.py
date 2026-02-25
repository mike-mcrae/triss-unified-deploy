#!/usr/bin/env python3
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    max_file_mb = float(os.environ.get("TRISS_MAX_FILE_MB", "90"))
    max_repo_mb = float(os.environ.get("TRISS_MAX_TRACKED_MB", "500"))
    forbidden_prefixes = [
        "data/raw/",
        "data/interim/",
    ]

    try:
        output = subprocess.check_output(
            ["git", "-C", str(root), "ls-files", "-z"],
            stderr=subprocess.DEVNULL,
        )  # tracked only
        tracked = [Path(p.decode("utf-8")) for p in output.split(b"\x00") if p]
    except Exception as exc:
        print(f"WARN: git tracked-file scan unavailable ({exc}); falling back to full tree scan.")
        tracked = [p.relative_to(root) for p in root.rglob("*") if p.is_file()]
    too_big = []
    forbidden = []
    total_bytes = 0

    for rel in tracked:
        abs_path = root / rel
        if not abs_path.exists() or not abs_path.is_file():
            continue
        size = abs_path.stat().st_size
        total_bytes += size
        if size > max_file_mb * 1024 * 1024:
            too_big.append((str(rel), size))
        rel_posix = str(rel).replace("\\", "/")
        if rel_posix.endswith(".gitkeep"):
            continue
        if any(rel_posix.startswith(prefix) for prefix in forbidden_prefixes):
            forbidden.append(rel_posix)

    total_mb = total_bytes / 1024 / 1024
    failed = False

    if too_big:
        failed = True
        print(f"FAIL: tracked files exceed {max_file_mb:.1f} MB")
        for path, size in sorted(too_big, key=lambda t: t[1], reverse=True):
            print(f"  - {path}: {size / 1024 / 1024:.2f} MB")

    if forbidden:
        failed = True
        print("FAIL: forbidden tracked paths detected")
        for path in sorted(forbidden):
            print(f"  - {path}")

    if total_mb > max_repo_mb:
        failed = True
        print(f"FAIL: total tracked size {total_mb:.2f} MB exceeds {max_repo_mb:.2f} MB")

    if failed:
        return 1

    print(f"OK: tracked files={len(tracked)}, total={total_mb:.2f} MB, max_file={max_file_mb:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
