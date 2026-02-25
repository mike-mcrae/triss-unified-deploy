#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List


def _parse_simple_yaml(path: Path) -> Dict[str, str]:
    data: Dict[str, str] = {}
    if not path.exists():
        return data
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        data[key.strip()] = value.strip().strip("\"").strip("'")
    return data


def _load_settings(root_dir: Path) -> Dict[str, str]:
    defaults = {
        "TRISS_DATA_DIR": str(root_dir / "data"),
    }
    local = _parse_simple_yaml(root_dir / "config" / "settings.local.yml")
    settings = defaults.copy()
    settings.update(local)
    for key in defaults:
        env_val = os.environ.get(key)
        if env_val not in (None, ""):
            settings[key] = env_val
    return settings


def _resolve_path(value: str, root_dir: Path) -> Path:
    p = Path(value).expanduser()
    if not p.is_absolute():
        p = (root_dir / p).resolve()
    return p.resolve()


def _csv_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        reader = csv.reader(fh)
        next(reader, None)
        return sum(1 for _ in reader)


def _fmt_mtime(path: Path) -> str:
    ts = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    return ts.isoformat()


def main() -> int:
    root_dir = Path(__file__).resolve().parents[1]
    settings = _load_settings(root_dir)
    data_dir = _resolve_path(settings["TRISS_DATA_DIR"], root_dir)
    final_dir = data_dir / "final"

    required = [
        final_dir / "profiles" / "1. profiles_summary.csv",
        final_dir / "profiles" / "4. triss_researcher_summary.csv",
        final_dir / "network" / "7.researcher_similarity_matrix.csv",
        final_dir / "network" / "8. user_to_other_publication_similarity_openai.csv",
        final_dir / "publications" / "4.All measured publications.csv",
        final_dir / "analysis" / "global" / "global_cluster_descriptions.json",
        final_dir / "analysis" / "global" / "global_umap_coordinates.csv",
        final_dir / "analysis" / "global" / "global_publication_umap_coordinates.csv",
        final_dir / "analysis" / "global" / "stage3" / "policy_domains_metadata.json",
        final_dir / "report" / "report.pdf",
    ]

    missing = [str(p) for p in required if not p.exists()]

    print(f"TRISS_DATA_DIR={data_dir}")
    print(f"FINAL_DIR={final_dir}")
    print(f"Checked at: {datetime.now(timezone.utc).isoformat()}")

    build_info_path = final_dir / "_BUILD_INFO.json"
    if build_info_path.exists():
        try:
            build_info = json.loads(build_info_path.read_text(encoding="utf-8"))
            print("Build info:")
            print(f"  commit: {build_info.get('git_commit')}")
            print(f"  timestamp_utc: {build_info.get('timestamp_utc')}")
            print(f"  stages: {build_info.get('stages_run')}")
        except Exception as exc:
            print(f"Build info unreadable: {exc}")

    key_counts: List[str] = []
    for label, path in [
        ("profiles", final_dir / "profiles" / "1. profiles_summary.csv"),
        ("researcher_summaries", final_dir / "profiles" / "4. triss_researcher_summary.csv"),
        ("publications", final_dir / "publications" / "4.All measured publications.csv"),
        ("global_umap", final_dir / "analysis" / "global" / "global_umap_coordinates.csv"),
        ("global_pub_umap", final_dir / "analysis" / "global" / "global_publication_umap_coordinates.csv"),
    ]:
        if path.exists():
            try:
                count = _csv_rows(path)
                key_counts.append(f"  - {label}: {count} rows ({_fmt_mtime(path)})")
            except Exception as exc:
                key_counts.append(f"  - {label}: ERROR ({exc})")
    print("Counts:")
    for line in key_counts:
        print(line)

    if missing:
        print("Missing required artifacts:")
        for p in missing:
            print(f"  - {p}")
        return 1

    print("Healthcheck OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
