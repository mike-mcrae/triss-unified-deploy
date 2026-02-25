#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Sequence


@dataclass
class Context:
    root_dir: Path
    data_dir: Path
    final_dir: Path
    logs_dir: Path
    source_shared_data_dir: Path
    source_pipeline_dir: Path
    source_report_dir: Path
    dry_run: bool
    stages_run: List[str]
    log_fp: object


def parse_simple_yaml(path: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not path.exists():
        return out
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        out[key.strip()] = value.strip().strip("\"").strip("'")
    return out


def load_settings(root_dir: Path) -> Dict[str, str]:
    parent_2 = root_dir.parent.parent
    defaults = {
        "TRISS_DATA_DIR": str(root_dir / "data"),
        "TRISS_ENV": "local",
        "TRISS_SOURCE_SHARED_DATA_DIR": str((parent_2 / "1. data").resolve()),
        "TRISS_SOURCE_PIPELINE_DIR": str((parent_2 / "triss-pipeline-2026").resolve()),
        "TRISS_SOURCE_REPORT_DIR": str((parent_2 / "triss-report-app-v3").resolve()),
    }
    local = parse_simple_yaml(root_dir / "config" / "settings.local.yml")
    merged = defaults.copy()
    merged.update(local)
    for key in defaults:
        env_val = os.environ.get(key)
        if env_val not in (None, ""):
            merged[key] = env_val
    return merged


def resolve_path(value: str, root_dir: Path) -> Path:
    p = Path(value).expanduser()
    if not p.is_absolute():
        p = (root_dir / p).resolve()
    return p.resolve()


def log(ctx: Context, msg: str) -> None:
    line = f"[{datetime.now(timezone.utc).isoformat()}] {msg}"
    print(line)
    ctx.log_fp.write(line + "\n")
    ctx.log_fp.flush()


def ensure_dir(path: Path, dry_run: bool) -> None:
    if dry_run:
        return
    path.mkdir(parents=True, exist_ok=True)


def copy_file(src: Path, dst: Path, ctx: Context) -> None:
    if not src.exists():
        log(ctx, f"WARN missing source: {src}")
        return
    if dst.exists():
        try:
            if src.stat().st_size == dst.stat().st_size and int(src.stat().st_mtime) <= int(dst.stat().st_mtime):
                log(ctx, f"SKIP up-to-date: {dst}")
                return
        except OSError:
            pass
    log(ctx, f"COPY {src} -> {dst}")
    if not ctx.dry_run:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def copy_tree(src_dir: Path, dst_dir: Path, ctx: Context) -> None:
    if not src_dir.exists():
        log(ctx, f"WARN missing source dir: {src_dir}")
        return
    for src in src_dir.rglob("*"):
        if not src.is_file():
            continue
        if src.name == ".DS_Store":
            continue
        rel = src.relative_to(src_dir)
        copy_file(src, dst_dir / rel, ctx)


def stage_sync_profiles(ctx: Context) -> None:
    src = ctx.source_shared_data_dir / "3. final"
    dst = ctx.final_dir / "profiles"
    for name in [
        "1. profiles_summary.csv",
        "4. triss_researcher_summary.csv",
    ]:
        copy_file(src / name, dst / name, ctx)


def stage_sync_network(ctx: Context) -> None:
    src = ctx.source_shared_data_dir / "3. final"
    dst = ctx.final_dir / "network"
    for name in [
        "7.researcher_similarity_matrix.csv",
        "8. user_to_other_publication_similarity_openai.csv",
    ]:
        copy_file(src / name, dst / name, ctx)


def stage_sync_publications(ctx: Context) -> None:
    src = ctx.source_pipeline_dir / "1. data" / "3. final"
    dst = ctx.final_dir / "publications"
    for name in [
        "4.All measured publications.csv",
        "3. All listed publications 2019 +.csv",
    ]:
        copy_file(src / name, dst / name, ctx)


def stage_sync_analysis(ctx: Context) -> None:
    src_base = ctx.source_pipeline_dir / "1. data" / "5. analysis"
    dst_base = ctx.final_dir / "analysis"
    for sub in ["global", "schools", "global_mpnet", "schools_mpnet"]:
        copy_tree(src_base / sub, dst_base / sub, ctx)


def stage_sync_embeddings(ctx: Context) -> None:
    src_base = ctx.source_pipeline_dir / "1. data" / "4. embeddings" / "v3"
    dst_base = ctx.final_dir / "embeddings" / "v3"
    copy_tree(src_base / "mpnet" / "by_researcher", dst_base / "mpnet" / "by_researcher", ctx)
    copy_tree(src_base / "mpnet" / "by_publication", dst_base / "mpnet" / "by_publication", ctx)
    copy_tree(src_base / "openai" / "by_researcher", dst_base / "openai" / "by_researcher", ctx)


def stage_sync_report(ctx: Context) -> None:
    copy_file(
        ctx.source_report_dir / "latex" / "report.pdf",
        ctx.final_dir / "report" / "report.pdf",
        ctx,
    )


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _csv_count(path: Path) -> int:
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        reader = csv.reader(fh)
        next(reader, None)
        return sum(1 for _ in reader)


def _git_commit(root_dir: Path) -> str:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(root_dir), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        return out
    except Exception:
        return "unknown"


def stage_build_info(ctx: Context) -> None:
    if ctx.dry_run:
        log(ctx, "DRY-RUN build-info skipped")
        return

    artifact_hashes: Dict[str, str] = {}
    for file_path in sorted(p for p in ctx.final_dir.rglob("*") if p.is_file() and p.name != "_BUILD_INFO.json"):
        artifact_hashes[str(file_path.relative_to(ctx.final_dir))] = _sha256(file_path)

    key_counts = {}
    for rel in [
        Path("profiles/1. profiles_summary.csv"),
        Path("profiles/4. triss_researcher_summary.csv"),
        Path("publications/4.All measured publications.csv"),
        Path("analysis/global/global_umap_coordinates.csv"),
        Path("analysis/global/global_publication_umap_coordinates.csv"),
    ]:
        p = ctx.final_dir / rel
        if p.exists():
            try:
                key_counts[str(rel)] = _csv_count(p)
            except Exception:
                key_counts[str(rel)] = -1

    embedding_meta = {
        "query_model": os.environ.get("TRISS_QUERY_EMBED_MODEL", "all-mpnet-base-v2"),
        "runtime_embeddings": [
            "embeddings/v3/mpnet/by_researcher",
            "embeddings/v3/mpnet/by_publication",
            "embeddings/v3/openai/by_researcher",
        ],
    }

    payload = {
        "git_commit": _git_commit(ctx.root_dir),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "stages_run": ctx.stages_run,
        "key_table_counts": key_counts,
        "embedding_models": embedding_meta,
        "artifact_sha256": artifact_hashes,
    }
    build_info = ctx.final_dir / "_BUILD_INFO.json"
    build_info.parent.mkdir(parents=True, exist_ok=True)
    build_info.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log(ctx, f"WROTE {build_info}")


STAGES: List[tuple[str, Callable[[Context], None]]] = [
    ("sync_profiles", stage_sync_profiles),
    ("sync_network", stage_sync_network),
    ("sync_publications", stage_sync_publications),
    ("sync_analysis", stage_sync_analysis),
    ("sync_embeddings", stage_sync_embeddings),
    ("sync_report", stage_sync_report),
    ("build_info", stage_build_info),
]
STAGE_INDEX = {name: idx for idx, (name, _) in enumerate(STAGES)}


def resolve_stage_list(args: argparse.Namespace) -> List[str]:
    all_names = [s for s, _ in STAGES]
    if args.only:
        requested = [p.strip() for p in args.only.split(",") if p.strip()]
        unknown = [p for p in requested if p not in STAGE_INDEX]
        if unknown:
            raise ValueError(f"Unknown --only stages: {unknown}")
        return requested

    selected: List[str] = all_names[:]
    if args.stage:
        unknown = [s for s in args.stage if s not in STAGE_INDEX]
        if unknown:
            raise ValueError(f"Unknown --stage values: {unknown}")
        selected = args.stage

    if args.from_stage or args.to_stage:
        start = STAGE_INDEX.get(args.from_stage, 0)
        end = STAGE_INDEX.get(args.to_stage, len(all_names) - 1)
        if end < start:
            raise ValueError("--to-stage must be after --from-stage")
        selected = all_names[start : end + 1]
    return selected


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="TRISS deploy pipeline runner")
    parser.add_argument("--stage", action="append", help="Run specific stage(s) (repeatable)")
    parser.add_argument("--from-stage", dest="from_stage", help="Start stage name")
    parser.add_argument("--to-stage", dest="to_stage", help="End stage name")
    parser.add_argument("--only", help="Comma-separated stage list")
    parser.add_argument("--dry-run", action="store_true", help="Print actions without writing files")
    args = parser.parse_args(argv)

    root_dir = Path(__file__).resolve().parents[1]
    settings = load_settings(root_dir)
    data_dir = resolve_path(settings["TRISS_DATA_DIR"], root_dir)
    final_dir = data_dir / "final"
    logs_dir = data_dir / "interim" / "logs"

    if not args.dry_run:
        logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"pipeline_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("w", encoding="utf-8") as log_fp:
        ctx = Context(
            root_dir=root_dir,
            data_dir=data_dir,
            final_dir=final_dir,
            logs_dir=logs_dir,
            source_shared_data_dir=resolve_path(settings["TRISS_SOURCE_SHARED_DATA_DIR"], root_dir),
            source_pipeline_dir=resolve_path(settings["TRISS_SOURCE_PIPELINE_DIR"], root_dir),
            source_report_dir=resolve_path(settings["TRISS_SOURCE_REPORT_DIR"], root_dir),
            dry_run=args.dry_run,
            stages_run=[],
            log_fp=log_fp,
        )
        log(ctx, f"TRISS_DATA_DIR={ctx.data_dir}")
        log(ctx, f"TRISS_SOURCE_SHARED_DATA_DIR={ctx.source_shared_data_dir}")
        log(ctx, f"TRISS_SOURCE_PIPELINE_DIR={ctx.source_pipeline_dir}")
        log(ctx, f"TRISS_SOURCE_REPORT_DIR={ctx.source_report_dir}")

        selected = resolve_stage_list(args)
        log(ctx, f"Selected stages: {selected}")

        for stage_name, fn in STAGES:
            if stage_name not in selected:
                continue
            log(ctx, f"=== START {stage_name} ===")
            fn(ctx)
            ctx.stages_run.append(stage_name)
            log(ctx, f"=== DONE {stage_name} ===")

        log(ctx, f"Completed stages: {ctx.stages_run}")
        log(ctx, f"Log file: {log_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
