#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Sequence

import pandas as pd


REQUIRED_RUNTIME_RELATIVE: List[Path] = [
    Path("profiles/1. profiles_summary.csv"),
    Path("profiles/4. triss_researcher_summary.csv"),
    Path("network/7.researcher_similarity_matrix.csv"),
    Path("network/8. user_to_other_publication_similarity_openai.csv"),
    Path("publications/4.All measured publications.csv"),
    Path("analysis/global/global_cluster_descriptions.json"),
    Path("analysis/global/global_umap_coordinates.csv"),
    Path("analysis/global/global_publication_umap_coordinates.csv"),
    Path("analysis/global/stage3/policy_domains_metadata.json"),
    Path("report/report.pdf"),
]


@dataclass
class Context:
    root_dir: Path
    data_dir: Path
    final_dir: Path
    logs_dir: Path
    interim_dir: Path
    settings: Dict[str, str]
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
    defaults = {
        "TRISS_DATA_DIR": str(root_dir / "data"),
        "TRISS_ENV": "local",
        "TRISS_BASELINE_URL": "",
        "TRISS_BASELINE_SHA256": "",
        "TRISS_BASELINE_TIMEOUT_SECONDS": "1200",
        "TRISS_BASELINE_OUTPUT_PATH": "",
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
        if not src.is_file() or src.name == ".DS_Store":
            continue
        rel = src.relative_to(src_dir)
        copy_file(src, dst_dir / rel, ctx)


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


def _missing_required_files(final_dir: Path) -> List[str]:
    missing: List[str] = []
    for rel in REQUIRED_RUNTIME_RELATIVE:
        if not (final_dir / rel).exists():
            missing.append(str(rel))
    return missing


def _write_runtime_status(ctx: Context, status: str, message: str, extra: Dict[str, object] | None = None) -> None:
    payload: Dict[str, object] = {
        "status": status,
        "message": message,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "data_dir": str(ctx.data_dir),
        "final_dir": str(ctx.final_dir),
    }
    if extra:
        payload.update(extra)
    status_path = ctx.interim_dir / "runtime_init_status.json"
    if ctx.dry_run:
        log(ctx, f"DRY-RUN status update: {status_path} -> {status}")
        return
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _download_file(url: str, dst: Path, timeout_seconds: int, ctx: Context) -> None:
    log(ctx, f"Downloading baseline archive: {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "triss-unified-deploy/1.0"})
    with urllib.request.urlopen(req, timeout=timeout_seconds) as response, dst.open("wb") as out_fh:
        shutil.copyfileobj(response, out_fh)


def _verify_sha256(path: Path, expected_sha256: str, ctx: Context) -> bool:
    if not expected_sha256.strip():
        return True
    actual = _sha256(path)
    ok = actual.lower() == expected_sha256.strip().lower()
    if ok:
        log(ctx, f"SHA256 verified: {actual}")
    else:
        log(ctx, f"ERROR SHA256 mismatch expected={expected_sha256} actual={actual}")
    return ok


def _detect_extracted_final_dir(extract_root: Path) -> Path | None:
    candidates: List[Path] = [
        extract_root / "final",
        extract_root / "data" / "final",
    ]
    for child in extract_root.iterdir():
        if not child.is_dir():
            continue
        candidates.append(child / "final")
        candidates.append(child / "data" / "final")
    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate
    return None


def _initialize_from_remote_baseline(ctx: Context) -> bool:
    baseline_url = ctx.settings.get("TRISS_BASELINE_URL", "").strip()
    if not baseline_url:
        log(ctx, "WARN TRISS_BASELINE_URL not set; cannot bootstrap empty runtime data")
        _write_runtime_status(
            ctx,
            status="waiting_for_baseline",
            message="TRISS_BASELINE_URL is not configured.",
            extra={"missing_required_files": _missing_required_files(ctx.final_dir)},
        )
        return False

    timeout_seconds = int(ctx.settings.get("TRISS_BASELINE_TIMEOUT_SECONDS", "1200") or "1200")
    expected_sha = ctx.settings.get("TRISS_BASELINE_SHA256", "")

    if ctx.dry_run:
        log(ctx, f"DRY-RUN would bootstrap runtime data from {baseline_url}")
        return True

    tmp_parent = ctx.interim_dir / "tmp"
    tmp_parent.mkdir(parents=True, exist_ok=True)
    tmp_root = Path(tempfile.mkdtemp(prefix="triss-baseline-", dir=str(tmp_parent)))
    archive_path = tmp_root / "baseline.tar.gz"
    extract_dir = tmp_root / "extract"
    extract_dir.mkdir(parents=True, exist_ok=True)

    try:
        _download_file(baseline_url, archive_path, timeout_seconds, ctx)
        if not _verify_sha256(archive_path, expected_sha, ctx):
            return False
        with tarfile.open(archive_path, mode="r:*") as tar:
            tar.extractall(path=extract_dir)

        src_final_dir = _detect_extracted_final_dir(extract_dir)
        if src_final_dir is None:
            log(ctx, "ERROR baseline archive does not contain final/ or data/final/")
            return False

        log(ctx, f"Applying baseline into runtime contract from {src_final_dir}")
        copy_tree(src_final_dir, ctx.final_dir, ctx)

        missing_after = _missing_required_files(ctx.final_dir)
        if missing_after:
            log(ctx, f"WARN baseline applied but still missing required files: {missing_after}")
            _write_runtime_status(
                ctx,
                status="partial_baseline",
                message="Baseline fetched, but required files are still missing.",
                extra={"missing_required_files": missing_after},
            )
            return False

        _write_runtime_status(
            ctx,
            status="ready",
            message="Runtime data initialized from remote baseline.",
            extra={
                "baseline_url": baseline_url,
                "baseline_sha256": expected_sha,
            },
        )
        return True
    except (urllib.error.URLError, TimeoutError) as exc:
        log(ctx, f"ERROR failed downloading baseline: {exc}")
        _write_runtime_status(
            ctx,
            status="baseline_download_failed",
            message=str(exc),
            extra={"baseline_url": baseline_url},
        )
        return False
    except Exception as exc:
        log(ctx, f"ERROR baseline initialization failed: {exc}")
        _write_runtime_status(
            ctx,
            status="baseline_init_failed",
            message=str(exc),
            extra={"baseline_url": baseline_url},
        )
        return False
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


def stage_ensure_runtime_data(ctx: Context) -> None:
    ensure_dir(ctx.data_dir / "raw", ctx.dry_run)
    ensure_dir(ctx.interim_dir, ctx.dry_run)
    ensure_dir(ctx.final_dir, ctx.dry_run)

    missing_before = _missing_required_files(ctx.final_dir)
    if not missing_before:
        log(ctx, "Runtime data already initialized; no bootstrap needed")
        _write_runtime_status(
            ctx,
            status="ready",
            message="Runtime data is already present.",
            extra={"missing_required_files": []},
        )
        return

    log(ctx, f"Runtime data missing required files: {missing_before}")
    ok = _initialize_from_remote_baseline(ctx)
    missing_after = _missing_required_files(ctx.final_dir)
    if ok and not missing_after:
        log(ctx, "Baseline initialization completed")
    else:
        log(ctx, f"Runtime data still incomplete after initialization attempt: {missing_after}")


def _upsert_csv(base_path: Path, patch_path: Path, key: str, ctx: Context) -> tuple[int, int, int]:
    if not base_path.exists():
        log(ctx, f"WARN base file missing, cannot apply patch: {base_path}")
        return (0, 0, 0)
    if not patch_path.exists():
        log(ctx, f"SKIP missing patch file: {patch_path}")
        return (0, 0, 0)

    base_df = pd.read_csv(base_path)
    patch_df = pd.read_csv(patch_path)

    if key not in base_df.columns or key not in patch_df.columns:
        log(ctx, f"WARN key '{key}' missing in base or patch: {base_path}, {patch_path}")
        return (0, 0, 0)

    if patch_df.empty:
        log(ctx, f"SKIP empty patch file: {patch_path}")
        return (0, 0, len(base_df))

    for col in patch_df.columns:
        if col not in base_df.columns:
            base_df[col] = pd.NA

    patch_df = patch_df[[c for c in base_df.columns if c in patch_df.columns]]

    base_df[key] = base_df[key].astype(str)
    patch_df[key] = patch_df[key].astype(str)
    patch_df = patch_df.dropna(subset=[key])
    patch_df = patch_df[patch_df[key].str.strip() != ""]
    patch_df = patch_df.drop_duplicates(subset=[key], keep="last")

    base_key_to_idx = {k: idx for idx, k in enumerate(base_df[key].tolist())}
    updated = 0
    inserted = 0

    for _, patch_row in patch_df.iterrows():
        k = str(patch_row.get(key, "")).strip()
        if not k:
            continue
        if k in base_key_to_idx:
            idx = base_key_to_idx[k]
            for col, value in patch_row.items():
                if pd.isna(value):
                    continue
                if isinstance(value, str) and not value.strip():
                    continue
                base_df.at[idx, col] = value
            updated += 1
            continue

        row_data = {col: pd.NA for col in base_df.columns}
        for col, value in patch_row.items():
            if col not in row_data:
                continue
            if pd.isna(value):
                continue
            row_data[col] = value
        base_df = pd.concat([base_df, pd.DataFrame([row_data])], ignore_index=True)
        base_key_to_idx[k] = len(base_df) - 1
        inserted += 1

    if key in base_df.columns:
        numeric_key = pd.to_numeric(base_df[key], errors="coerce")
        if numeric_key.notna().any():
            base_df = base_df.assign(_sort_key=numeric_key)
            base_df = base_df.sort_values(by="_sort_key", kind="stable").drop(columns=["_sort_key"])

    if not ctx.dry_run:
        base_path.parent.mkdir(parents=True, exist_ok=True)
        base_df.to_csv(base_path, index=False)

    return (updated, inserted, len(base_df))


def stage_incremental_researcher_update(ctx: Context) -> None:
    updates_dir = ctx.data_dir / "raw" / "updates" / "researchers"
    profiles_base = ctx.final_dir / "profiles" / "1. profiles_summary.csv"
    summaries_base = ctx.final_dir / "profiles" / "4. triss_researcher_summary.csv"
    profiles_patch = updates_dir / "profiles_patch.csv"
    summaries_patch = updates_dir / "summaries_patch.csv"

    if not updates_dir.exists():
        log(ctx, f"No updates dir at {updates_dir}; incremental update is a no-op")
        return

    p_updated, p_inserted, p_total = _upsert_csv(profiles_base, profiles_patch, "n_id", ctx)
    s_updated, s_inserted, s_total = _upsert_csv(summaries_base, summaries_patch, "n_id", ctx)

    log(
        ctx,
        (
            "Incremental researcher update complete: "
            f"profiles(updated={p_updated}, inserted={p_inserted}, total={p_total}); "
            f"summaries(updated={s_updated}, inserted={s_inserted}, total={s_total})"
        ),
    )


def stage_baseline_build(ctx: Context) -> None:
    if ctx.settings.get("TRISS_ENV", "local").strip().lower() == "render":
        raise RuntimeError("baseline_build is offline-only and disabled when TRISS_ENV=render")

    ensure_dir(ctx.interim_dir / "baseline", ctx.dry_run)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    configured_output = ctx.settings.get("TRISS_BASELINE_OUTPUT_PATH", "").strip()
    if configured_output:
        artifact_path = resolve_path(configured_output, ctx.root_dir)
    else:
        artifact_path = ctx.interim_dir / "baseline" / f"triss_baseline_{ts}.tar.gz"
    manifest_path = artifact_path.with_suffix(artifact_path.suffix + ".manifest.json")

    missing = _missing_required_files(ctx.final_dir)
    if missing:
        log(ctx, f"WARN building baseline with missing required files: {missing}")

    if ctx.dry_run:
        log(ctx, f"DRY-RUN would create baseline artifact at {artifact_path}")
        return

    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(artifact_path, mode="w:gz") as tar:
        tar.add(ctx.final_dir, arcname="final")

    checksum = _sha256(artifact_path)
    file_hashes: Dict[str, str] = {}
    for path in sorted(p for p in ctx.final_dir.rglob("*") if p.is_file()):
        file_hashes[str(path.relative_to(ctx.final_dir))] = _sha256(path)

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifact": str(artifact_path),
        "artifact_sha256": checksum,
        "required_missing": missing,
        "file_count": len(file_hashes),
        "file_sha256": file_hashes,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    log(ctx, f"Baseline artifact written: {artifact_path}")
    log(ctx, f"Baseline manifest written: {manifest_path}")


def stage_build_info(ctx: Context) -> None:
    if ctx.dry_run:
        log(ctx, "DRY-RUN build-info skipped")
        return

    artifact_hashes: Dict[str, str] = {}
    if ctx.final_dir.exists():
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

    payload = {
        "git_commit": _git_commit(ctx.root_dir),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "stages_run": ctx.stages_run,
        "key_table_counts": key_counts,
        "required_runtime_missing": _missing_required_files(ctx.final_dir),
        "artifact_sha256": artifact_hashes,
    }
    build_info = ctx.final_dir / "_BUILD_INFO.json"
    build_info.parent.mkdir(parents=True, exist_ok=True)
    build_info.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log(ctx, f"WROTE {build_info}")


STAGES: List[tuple[str, Callable[[Context], None]]] = [
    ("ensure_runtime_data", stage_ensure_runtime_data),
    ("incremental_researcher_update", stage_incremental_researcher_update),
    ("baseline_build", stage_baseline_build),
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

    selected: List[str] = ["ensure_runtime_data", "build_info"]
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
    parser = argparse.ArgumentParser(description="TRISS deploy runtime pipeline")
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
    interim_dir = data_dir / "interim"
    logs_dir = interim_dir / "logs"

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
            interim_dir=interim_dir,
            settings=settings,
            dry_run=args.dry_run,
            stages_run=[],
            log_fp=log_fp,
        )
        log(ctx, f"TRISS_ENV={settings.get('TRISS_ENV', 'local')}")
        log(ctx, f"TRISS_DATA_DIR={ctx.data_dir}")
        log(ctx, f"TRISS_BASELINE_URL={'set' if settings.get('TRISS_BASELINE_URL', '').strip() else 'unset'}")

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
