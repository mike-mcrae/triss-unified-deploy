#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
TRISS Unified Research Web Application
Backend API using FastAPI
"""

import json
import html
import re
import unicodedata
import os
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import numpy as np
import pandas as pd
from pydantic import BaseModel
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None  # type: ignore[assignment]

app = FastAPI(title="TRISS Unified App API")

def _parse_simple_yaml(path: Path) -> Dict[str, str]:
    data: Dict[str, str] = {}
    if not path.exists():
        return data
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        data[key.strip()] = value.strip().strip("'").strip('"')
    return data


def _load_settings() -> Dict[str, str]:
    root_dir = Path(__file__).resolve().parents[2]
    local_settings = _parse_simple_yaml(root_dir / "config" / "settings.local.yml")
    defaults = {
        "TRISS_DATA_DIR": str(root_dir / "data"),
        "TRISS_ENV": "local",
        "TRISS_CORS_ORIGINS": (
            "http://localhost:5173,http://localhost:5174,http://localhost:5175,"
            "http://127.0.0.1:5173,http://127.0.0.1:5174,http://127.0.0.1:5175"
        ),
        "TRISS_QUERY_EMBED_MODEL": "all-mpnet-base-v2",
    }
    settings = defaults.copy()
    settings.update(local_settings)
    for key in defaults.keys():
        env_val = os.environ.get(key)
        if env_val not in (None, ""):
            settings[key] = env_val
    return settings


def _resolve_path(value: str, root_dir: Path) -> Path:
    p = Path(value).expanduser()
    if not p.is_absolute():
        p = (root_dir / p).resolve()
    return p.resolve()


SETTINGS = _load_settings()

# Setup CORS (env/config driven)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in SETTINGS["TRISS_CORS_ORIGINS"].split(",") if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[2]
TRISS_DATA_DIR = _resolve_path(SETTINGS["TRISS_DATA_DIR"], ROOT_DIR)
FINAL_DIR = TRISS_DATA_DIR / "final"

# App runtime contract (all reads from data/final/*)
PROFILES_DIR = FINAL_DIR / "profiles"
NETWORK_DIR = FINAL_DIR / "network"
PUBLICATIONS_DIR = FINAL_DIR / "publications"
ANALYSIS_DIR = FINAL_DIR / "analysis"
ANALYSIS_GLOBAL_DIR = ANALYSIS_DIR / "global"
ANALYSIS_SCHOOLS_DIR = ANALYSIS_DIR / "schools"
ANALYSIS_GLOBAL_MPNET_DIR = ANALYSIS_DIR / "global_mpnet"
ANALYSIS_SCHOOLS_MPNET_DIR = ANALYSIS_DIR / "schools_mpnet"
ANALYSIS_STAGE2_DIR = ANALYSIS_GLOBAL_DIR / "stage2"
ANALYSIS_STAGE3_DIR = ANALYSIS_GLOBAL_DIR / "stage3"
EMBED_ROOT_DIR = FINAL_DIR / "embeddings" / "v3" / "mpnet"
RESEARCHER_EMBED_DIR = EMBED_ROOT_DIR / "by_researcher"
PUBLICATION_EMBED_DIR = EMBED_ROOT_DIR / "by_publication"
QUERY_EMBED_MODEL = SETTINGS["TRISS_QUERY_EMBED_MODEL"]
OPENAI_RESEARCHER_EMBED_DIR = FINAL_DIR / "embeddings" / "v3" / "openai" / "by_researcher"
GLOBAL_CLUSTER_ASSIGNMENTS_PATH = ANALYSIS_GLOBAL_DIR / "global_cluster_assignments.csv"
GLOBAL_CLUSTER_CENTROIDS_PATH = ANALYSIS_GLOBAL_DIR / "global_cluster_centroids.npy"
REPORT_V3_PDF_PATH = FINAL_DIR / "report" / "report.pdf"
RUNTIME_REQUIRED_RELATIVE: List[Path] = [
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

# ---------------------------------------------------------------------
# MODELS
# ---------------------------------------------------------------------
class SearchResult(BaseModel):
    n_id: int
    name: str
    department: str
    school: str

class ResearcherProfile(BaseModel):
    n_id: int
    name: str
    email: str
    department: str
    school: str
    one_line_summary: Optional[str] = None
    topics: Optional[str] = None
    research_area: Optional[str] = None
    subfields: Optional[str] = None

class NeighbourResponse(BaseModel):
    n_id: int
    name: str
    department: str
    school: str
    similarity: float
    rank: int

class PublicationMatch(BaseModel):
    title: str
    abstract: Optional[str] = None
    similarity: float
    doi: Optional[str] = None
    main_url: Optional[str] = None
    authors: Optional[str] = None

class OwnPublication(BaseModel):
    title: str
    abstract: Optional[str] = None
    year: Optional[int] = None
    doi: Optional[str] = None
    main_url: Optional[str] = None
    authors: Optional[str] = None

class EmailLookup(BaseModel):
    email: str

class LookupResponse(BaseModel):
    n_id: int
    name: str
    department: str
    school: str

class ExpertSearchRequest(BaseModel):
    query: str
    top_k_researchers: int = 12
    top_k_publications: int = 20
    school: Optional[str] = None
    department: Optional[str] = None

class ExpertResearcherResult(BaseModel):
    n_id: int
    name: str
    school: str
    department: str
    similarity: float

class ExpertPublicationResult(BaseModel):
    n_id: int
    researcher_name: Optional[str] = None
    school: Optional[str] = None
    department: Optional[str] = None
    publication_index: Optional[int] = None
    article_id: str
    title: str
    abstract_snippet: Optional[str] = None
    main_url: Optional[str] = None
    similarity: float = 0.0

class ExpertSearchResponse(BaseModel):
    query: str
    top_researchers: List[ExpertResearcherResult]
    top_publications: List[ExpertPublicationResult]

class ExpertThemeOption(BaseModel):
    option_id: str
    label: str
    scope: str
    school: Optional[str] = None
    school_label: Optional[str] = None
    topic_id: int

class ExpertThemeSearchRequest(BaseModel):
    option_id: str
    top_k_researchers: int = 12
    top_k_publications: int = 20
    school: Optional[str] = None
    department: Optional[str] = None

class ExpertSearchFiltersResponse(BaseModel):
    schools: List[str]
    departments_by_school: Dict[str, List[str]]

# ---------------------------------------------------------------------
# DATA CACHE
# ---------------------------------------------------------------------
profiles_df: Optional[pd.DataFrame] = None
researcher_summary_df: Optional[pd.DataFrame] = None
similarity_matrix_df: Optional[pd.DataFrame] = None
pub_similarity_df: Optional[pd.DataFrame] = None
publications_df: Optional[pd.DataFrame] = None
distilled_synthesis: Dict[str, Any] = {}
umap_coords_df: Optional[pd.DataFrame] = None
pub_umap_coords_df: Optional[pd.DataFrame] = None
umap_coords_mpnet_df: Optional[pd.DataFrame] = None
pub_umap_coords_mpnet_df: Optional[pd.DataFrame] = None
map_points_cache: List[Dict[str, Any]] = []
map_filters_cache: Dict[str, Any] = {}
map_topics_cache: Dict[str, Any] = {}
map_researcher_points_cache: List[Dict[str, Any]] = []
map_v2_points_cache: List[Dict[str, Any]] = []
map_v2_filters_cache: Dict[str, Any] = {}
map_v2_topics_cache: Dict[str, Any] = {}
map_v2_researcher_points_cache: List[Dict[str, Any]] = []
researcher_embedding_ids: List[int] = []
researcher_embedding_matrix: Optional[np.ndarray] = None
researcher_meta_cache: Dict[int, Dict[str, str]] = {}
abstract_records: List[Dict[str, Any]] = []
abstract_matrix: Optional[np.ndarray] = None
query_embed_cache: Dict[str, np.ndarray] = {}
mpnet_model: Optional[Any] = None
report_schools_cache: List[Dict[str, Any]] = []
report_school_detail_cache: Dict[str, Dict[str, Any]] = {}
report_statistics_cache: Dict[str, Any] = {}
report_key_areas_cache: List[Dict[str, Any]] = []
report_topic_experts_cache: Dict[int, List[Dict[str, Any]]] = {}
report_school_topics_cache: Dict[str, List[Dict[str, Any]]] = {}
report_school_topic_experts_cache: Dict[str, Dict[int, List[Dict[str, Any]]]] = {}
report_macro_themes_cache: List[Dict[str, Any]] = []
report_macro_theme_experts_cache: Dict[int, List[Dict[str, Any]]] = {}
expert_theme_options_cache: List[Dict[str, Any]] = []
expert_search_filters_cache: Dict[str, Any] = {"schools": [], "departments_by_school": {}}
startup_init_status: Dict[str, Any] = {
    "state": "not_started",
    "message": "Startup not started.",
    "loaded": False,
    "attempted_bootstrap": False,
    "missing_required_files": [],
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
}

# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------
def _norm_text(x: object) -> str:
    if pd.isna(x): return ""
    s = str(x).strip().lower()
    return " ".join(s.split())

def _clean_text(x: object) -> str:
    if pd.isna(x): return ""
    s = str(x).strip()
    # Unescape HTML entities (&amp; -> &, etc)
    s = html.unescape(s)
    markers = ("Ã", "Â", "â", "\x83", "\x82")

    def marker_count(text: str) -> int:
        return sum(text.count(marker) for marker in markers)

    def decode_once(text: str, strict: bool) -> str:
        try:
            raw = text.encode("latin1", errors="ignore")
            return raw.decode("utf-8", errors="strict" if strict else "ignore")
        except Exception:
            return text

    # Repair common UTF-8/latin-1 mojibake sequences, including double-encoded text.
    for _ in range(6):
        if marker_count(s) == 0:
            break
        strict_candidate = decode_once(s, strict=True)
        if strict_candidate != s and marker_count(strict_candidate) < marker_count(s):
            s = strict_candidate
            continue
        lossy_candidate = decode_once(s, strict=False)
        if lossy_candidate != s and marker_count(lossy_candidate) < marker_count(s):
            s = lossy_candidate
            continue
        break
    # Fix a recurring punctuation artifact where apostrophes/quotes become upside-down question marks.
    # This preserves legitimate Spanish opening questions like "¿Que ...?" when a closing "?" is present.
    fixed_chars: List[str] = []
    for idx, ch in enumerate(s):
        if ch != "¿":
            fixed_chars.append(ch)
            continue
        closing_q_idx = s.find("?", idx + 1)
        if 0 <= closing_q_idx - idx <= 120:
            fixed_chars.append(ch)
        else:
            fixed_chars.append("'")
    s = "".join(fixed_chars)
    # Normalize whitespace for display consistency.
    return " ".join(s.split())

def _titlecase_name(firstname: str, lastname: str) -> str:
    first = str(firstname).strip()
    last = str(lastname).strip()
    # Preserve accented/combining sequences in first names (e.g. fáinche -> Fáinche)
    # while still title-casing surnames like o'neill -> O'Neill.
    if first:
        first = first[0].upper() + first[1:]
    return unicodedata.normalize("NFC", f"{first} {last.title()}".strip())

def _require_loaded():
    if profiles_df is None:
        detail = startup_init_status.get("message") or "Data not loaded yet."
        raise HTTPException(status_code=503, detail=detail)


def _missing_runtime_files() -> List[str]:
    missing: List[str] = []
    for rel in RUNTIME_REQUIRED_RELATIVE:
        if not (FINAL_DIR / rel).exists():
            missing.append(str(rel))
    return missing


def _set_startup_status(
    state: str,
    message: str,
    loaded: bool,
    attempted_bootstrap: bool,
    missing_required_files: Optional[List[str]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    global startup_init_status
    payload: Dict[str, Any] = {
        "state": state,
        "message": message,
        "loaded": loaded,
        "attempted_bootstrap": attempted_bootstrap,
        "missing_required_files": missing_required_files or [],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    if extra:
        payload.update(extra)
    startup_init_status = payload
    print(f"[startup] {state}: {message}")


def _run_runtime_initializer_if_needed() -> List[str]:
    missing_before = _missing_runtime_files()
    if not missing_before:
        _set_startup_status(
            state="ready",
            message="Runtime data already present.",
            loaded=False,
            attempted_bootstrap=False,
            missing_required_files=[],
        )
        return []

    _set_startup_status(
        state="initializing",
        message="Runtime data missing. Attempting baseline initialization.",
        loaded=False,
        attempted_bootstrap=True,
        missing_required_files=missing_before,
    )
    cmd = [
        sys.executable,
        str(ROOT_DIR / "pipeline" / "run_pipeline.py"),
        "--only",
        "ensure_runtime_data,build_info",
    ]
    try:
        result = subprocess.run(
            cmd,
            cwd=str(ROOT_DIR),
            check=False,
            capture_output=True,
            text=True,
            timeout=1800,
        )
        if result.stdout.strip():
            print(result.stdout.strip())
        if result.stderr.strip():
            print(result.stderr.strip())
        if result.returncode != 0:
            _set_startup_status(
                state="init_failed",
                message=f"Runtime initializer exited with code {result.returncode}.",
                loaded=False,
                attempted_bootstrap=True,
                missing_required_files=missing_before,
                extra={"initializer_return_code": result.returncode},
            )
    except Exception as exc:
        _set_startup_status(
            state="init_failed",
            message=f"Runtime initializer error: {exc}",
            loaded=False,
            attempted_bootstrap=True,
            missing_required_files=missing_before,
        )

    missing_after = _missing_runtime_files()
    if missing_after:
        _set_startup_status(
            state="waiting_for_data",
            message="Runtime data is not ready; service started in degraded mode.",
            loaded=False,
            attempted_bootstrap=True,
            missing_required_files=missing_after,
        )
    return missing_after


def _safe_int(value: object) -> Optional[int]:
    try:
        if pd.isna(value):
            return None
        return int(value)
    except Exception:
        return None


def _extract_publication_index(article_id: str) -> Optional[int]:
    if not article_id:
        return None
    match = re.search(r"_(\d+)$", article_id)
    if not match:
        return None
    return _safe_int(match.group(1))


def _normalize_vector(vec: np.ndarray) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    if norm <= 0:
        return arr
    return arr / norm


def _topk_indices(scores: np.ndarray, k: int) -> np.ndarray:
    if scores.size == 0 or k <= 0:
        return np.array([], dtype=int)
    k = min(k, scores.size)
    idx = np.argpartition(scores, -k)[-k:]
    return idx[np.argsort(scores[idx])[::-1]]


def _load_expert_search_embeddings() -> None:
    global researcher_embedding_ids, researcher_embedding_matrix, researcher_meta_cache
    global abstract_records, abstract_matrix

    researcher_embedding_ids = []
    researcher_meta_cache = {}
    abstract_records = []
    researcher_rows: List[np.ndarray] = []
    abstract_rows: List[np.ndarray] = []

    if profiles_df is not None:
        for _, row in profiles_df.iterrows():
            n_id = _safe_int(row.get("n_id"))
            if n_id is None:
                continue
            researcher_meta_cache[n_id] = {
                "name": _clean_text(row.get("display_name") or f"Researcher {n_id}"),
                "school": _clean_text(row.get("school")),
                "department": _clean_text(row.get("department")),
            }

    pubs_by_article: Dict[str, Dict[str, Any]] = {}
    if publications_df is not None and "article_id" in publications_df.columns:
        for _, row in publications_df.iterrows():
            article_id = str(row.get("article_id", "")).strip()
            if not article_id:
                continue
            pubs_by_article[article_id] = row.to_dict()

    if RESEARCHER_EMBED_DIR.exists():
        for f in sorted(RESEARCHER_EMBED_DIR.glob("n_*.json")):
            try:
                with f.open("r", encoding="utf-8") as fh:
                    obj = json.load(fh)
                n_id = _safe_int(obj.get("n_id"))
                vec = obj.get("embeddings", {}).get("abstracts_mean")
                if n_id is None or vec is None:
                    continue
                nvec = _normalize_vector(np.asarray(vec, dtype=np.float32))
                if nvec.size == 0:
                    continue
                researcher_embedding_ids.append(n_id)
                researcher_rows.append(nvec)
                if n_id not in researcher_meta_cache:
                    researcher_meta_cache[n_id] = {
                        "name": f"Researcher {n_id}",
                        "school": "",
                        "department": "",
                    }
            except Exception:
                continue

    researcher_embedding_matrix = (
        np.vstack(researcher_rows).astype(np.float32) if researcher_rows else np.zeros((0, 0), dtype=np.float32)
    )

    if PUBLICATION_EMBED_DIR.exists():
        for f in sorted(PUBLICATION_EMBED_DIR.glob("article_*.json")):
            try:
                with f.open("r", encoding="utf-8") as fh:
                    obj = json.load(fh)
                article_id = str(obj.get("article_id", "")).strip()
                n_id = _safe_int(obj.get("n_id"))
                vec = obj.get("embeddings", {}).get("abstract")
                if not article_id or n_id is None or vec is None:
                    continue
                nvec = _normalize_vector(np.asarray(vec, dtype=np.float32))
                if nvec.size == 0:
                    continue
                src = pubs_by_article.get(article_id, {})
                title = _clean_text(src.get("Title") or obj.get("title") or "Untitled")
                abstract_text = _clean_text(src.get("abstract") or "")
                if not abstract_text:
                    abstract_text = _clean_text(obj.get("abstract") or "")
                main_url = src.get("main_url")
                main_url_val = str(main_url).strip() if pd.notna(main_url) else None
                author_name = (
                    researcher_meta_cache.get(n_id, {}).get("name")
                    or _clean_text(src.get("firstname", "")) + " " + _clean_text(src.get("lastname", ""))
                ).strip()
                abstract_records.append({
                    "article_id": article_id,
                    "n_id": n_id,
                    "researcher_name": author_name or f"Researcher {n_id}",
                    "publication_index": _extract_publication_index(article_id),
                    "title": title,
                    "abstract": abstract_text,
                    "main_url": main_url_val if main_url_val else None,
                })
                abstract_rows.append(nvec)
            except Exception:
                continue

    abstract_matrix = np.vstack(abstract_rows).astype(np.float32) if abstract_rows else np.zeros((0, 0), dtype=np.float32)
    print(
        f"Loaded expert-search embeddings: "
        f"{len(researcher_embedding_ids)} researchers, {len(abstract_records)} publications"
    )


def _get_query_embedding(query_text: str) -> np.ndarray:
    global mpnet_model, query_embed_cache
    query = " ".join(query_text.split())
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    cache_key = query.lower()
    if cache_key in query_embed_cache:
        return query_embed_cache[cache_key]

    if SentenceTransformer is None:
        raise HTTPException(status_code=503, detail="sentence-transformers is not installed in backend environment.")

    if mpnet_model is None:
        try:
            mpnet_model = SentenceTransformer(QUERY_EMBED_MODEL)
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"Failed to load MPNet model ({QUERY_EMBED_MODEL}): {exc}") from exc

    try:
        vec = mpnet_model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        qvec = _normalize_vector(np.asarray(vec, dtype=np.float32))
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Failed to embed query with MPNet: {exc}") from exc

    if len(query_embed_cache) >= 128:
        oldest_key = next(iter(query_embed_cache))
        query_embed_cache.pop(oldest_key, None)
    query_embed_cache[cache_key] = qvec
    return qvec


def _rank_researchers_for_query(query_vec: np.ndarray, top_k: int) -> List[Dict[str, Any]]:
    if researcher_embedding_matrix is None or researcher_embedding_matrix.size == 0:
        return []
    scores = researcher_embedding_matrix @ query_vec
    top_idx = _topk_indices(scores, top_k)
    results: List[Dict[str, Any]] = []
    for idx in top_idx:
        n_id = int(researcher_embedding_ids[int(idx)])
        meta = researcher_meta_cache.get(n_id, {})
        results.append({
            "n_id": n_id,
            "name": meta.get("name") or f"Researcher {n_id}",
            "school": meta.get("school", ""),
            "department": meta.get("department", ""),
            "similarity": float(scores[int(idx)]),
        })
    return results


def _rank_publications_for_query(query_vec: np.ndarray, top_k: int) -> List[Dict[str, Any]]:
    if abstract_matrix is None or abstract_matrix.size == 0:
        return []
    scores = abstract_matrix @ query_vec
    top_idx = _topk_indices(scores, top_k)
    results: List[Dict[str, Any]] = []
    for idx in top_idx:
        rec = abstract_records[int(idx)]
        n_id = int(rec["n_id"])
        meta = researcher_meta_cache.get(n_id, {})
        abstract_text = rec.get("abstract") or ""
        snippet = abstract_text[:360] + ("..." if len(abstract_text) > 360 else "")
        results.append({
            "n_id": n_id,
            "researcher_name": rec.get("researcher_name"),
            "school": meta.get("school", ""),
            "department": meta.get("department", ""),
            "publication_index": rec.get("publication_index"),
            "article_id": rec.get("article_id", ""),
            "title": rec.get("title", "Untitled"),
            "abstract_snippet": snippet if snippet else None,
            "main_url": rec.get("main_url"),
            "similarity": float(scores[int(idx)]),
        })
    return results


def _rank_publications_for_researcher_query(n_id: int, query_vec: np.ndarray, top_k: int) -> List[Dict[str, Any]]:
    if abstract_matrix is None or abstract_matrix.size == 0:
        return []
    candidate_idx = [i for i, rec in enumerate(abstract_records) if int(rec.get("n_id", -1)) == n_id]
    if not candidate_idx:
        return []
    sub = abstract_matrix[candidate_idx]
    scores = sub @ query_vec
    top_local = _topk_indices(scores, top_k)
    results: List[Dict[str, Any]] = []
    for local_idx in top_local:
        global_idx = candidate_idx[int(local_idx)]
        rec = abstract_records[global_idx]
        meta = researcher_meta_cache.get(n_id, {})
        abstract_text = rec.get("abstract") or ""
        snippet = abstract_text[:360] + ("..." if len(abstract_text) > 360 else "")
        results.append({
            "n_id": n_id,
            "researcher_name": rec.get("researcher_name"),
            "school": meta.get("school", ""),
            "department": meta.get("department", ""),
            "publication_index": rec.get("publication_index"),
            "article_id": rec.get("article_id", ""),
            "title": rec.get("title", "Untitled"),
            "abstract_snippet": snippet if snippet else None,
            "main_url": rec.get("main_url"),
            "similarity": float(scores[int(local_idx)]),
        })
    return results


def _build_map_payloads() -> None:
    global map_points_cache, map_filters_cache, map_topics_cache
    map_points_cache, map_filters_cache, map_topics_cache = _build_map_payloads_for(
        ANALYSIS_GLOBAL_DIR, pub_umap_coords_df
    )


def _build_map_payloads_v2() -> None:
    global map_v2_points_cache, map_v2_filters_cache, map_v2_topics_cache
    map_v2_points_cache, map_v2_filters_cache, map_v2_topics_cache = _build_map_payloads_for(
        ANALYSIS_GLOBAL_MPNET_DIR, pub_umap_coords_mpnet_df
    )


def _build_map_payloads_for(
    analysis_global_dir: Path, pub_umap_df: Optional[pd.DataFrame]
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    points_cache: List[Dict[str, Any]] = []
    filters_cache: Dict[str, Any] = {
        "schools": ["All"],
        "departments_by_school": {},
        "researchers": [],
        "topics": [],
        "policies": [],
    }
    topics_cache: Dict[str, Any] = {}

    if pub_umap_df is None or publications_df is None:
        return points_cache, filters_cache, topics_cache

    cluster_desc_path = analysis_global_dir / "global_cluster_descriptions.json"
    cluster_descriptions: Dict[str, Dict[str, Any]] = {}
    if cluster_desc_path.exists():
        with cluster_desc_path.open("r", encoding="utf-8") as f:
            loaded = json.load(f)
            if isinstance(loaded, dict):
                cluster_descriptions = loaded

    cluster_to_policy: Dict[int, int] = {}
    policy_labels: Dict[int, str] = {}
    stage3_cluster_policy_path = ANALYSIS_STAGE3_DIR / "global_cluster_policy_weights.csv"
    stage3_policy_meta_path = ANALYSIS_STAGE3_DIR / "policy_domains_metadata.json"
    if stage3_cluster_policy_path.exists():
        try:
            cdf = pd.read_csv(stage3_cluster_policy_path)
            cdf["cluster_id"] = pd.to_numeric(cdf.get("cluster_id"), errors="coerce").astype("Int64")
            cdf["hard_policy_domain"] = pd.to_numeric(cdf.get("hard_policy_domain"), errors="coerce").astype("Int64")
            cdf = cdf.dropna(subset=["cluster_id", "hard_policy_domain"])
            cluster_to_policy = {
                int(r["cluster_id"]): int(r["hard_policy_domain"])
                for _, r in cdf.iterrows()
            }
        except Exception:
            cluster_to_policy = {}
    if stage3_policy_meta_path.exists():
        try:
            with stage3_policy_meta_path.open("r", encoding="utf-8") as f:
                meta = json.load(f)
            if isinstance(meta, list):
                for row in meta:
                    pid = _safe_int(row.get("policy_domain_id"))
                    if pid is None:
                        continue
                    policy_labels[pid] = _clean_text(row.get("title")) or f"Policy {pid}"
        except Exception:
            policy_labels = {}

    coords = pub_umap_df.copy()
    coords["article_id"] = coords["article_id"].astype(str)
    coords["topic"] = pd.to_numeric(coords.get("predicted_cluster"), errors="coerce").astype("Int64")
    coords["x"] = pd.to_numeric(coords.get("umap_x"), errors="coerce")
    coords["y"] = pd.to_numeric(coords.get("umap_y"), errors="coerce")
    coords = coords.dropna(subset=["x", "y", "topic"])

    pubs = publications_df.copy()
    pubs["article_id"] = pubs["article_id"].astype(str)
    pubs["n_id"] = pd.to_numeric(pubs["n_id"], errors="coerce").astype("Int64")

    if "Title" not in pubs.columns:
        pubs["Title"] = ""
    if "abstract" not in pubs.columns:
        pubs["abstract"] = ""
    if "school" not in pubs.columns:
        pubs["school"] = ""
    if "department" not in pubs.columns:
        pubs["department"] = ""
    if "firstname" not in pubs.columns:
        pubs["firstname"] = ""
    if "lastname" not in pubs.columns:
        pubs["lastname"] = ""

    pubs["name"] = (
        pubs["firstname"].fillna("").astype(str).str.strip() + " " + pubs["lastname"].fillna("").astype(str).str.strip()
    ).str.strip()
    pubs.loc[pubs["name"] == "", "name"] = None
    pubs["publication_index"] = pubs["article_id"].apply(_extract_publication_index)

    merged = coords.merge(
        pubs[["article_id", "n_id", "school", "department", "name", "publication_index", "Title", "abstract"]],
        on="article_id",
        how="left"
    )

    if profiles_df is not None:
        merged = merged.merge(
            profiles_df[["n_id", "display_name", "school", "department", "email", "biography", "research_interests"]],
            on="n_id",
            how="left",
            suffixes=("", "_profile")
        )
        # Prefer canonical profile display names to avoid mojibake from publications files.
        merged["name"] = merged["display_name"].fillna(merged["name"])
        merged["school"] = merged["school"].fillna(merged["school_profile"])
        merged["department"] = merged["department"].fillna(merged["department_profile"])
    if researcher_summary_df is not None:
        summary_cols = ["n_id", "one_line_summary", "overall_research_area", "topics", "subfields"]
        existing = [c for c in summary_cols if c in researcher_summary_df.columns]
        if len(existing) > 1:
            merged = merged.merge(
                researcher_summary_df[existing],
                on="n_id",
                how="left",
                suffixes=("", "_summary")
            )

    points: List[Dict[str, Any]] = []
    for _, row in merged.iterrows():
        topic_id = _safe_int(row.get("topic"))
        n_id = _safe_int(row.get("n_id"))
        publication_index = _safe_int(row.get("publication_index"))
        if topic_id is None:
            continue
        topic_meta = cluster_descriptions.get(str(topic_id), {})
        policy_id = cluster_to_policy.get(topic_id)
        key = f"{n_id}:{publication_index}" if n_id is not None and publication_index is not None else row.get("article_id", "")
        points.append({
            "key": key,
            "n_id": n_id,
            "publication_index": publication_index,
            "name": _clean_text(row.get("name")),
            "school": _clean_text(row.get("school")),
            "department": _clean_text(row.get("department")),
            "topic": topic_id,
            "topic_name": topic_meta.get("topic_name") or f"Topic {topic_id}",
            "policy_domain": policy_id,
            "policy_name": policy_labels.get(policy_id, f"Policy {policy_id}") if policy_id is not None else None,
            "x": float(row["x"]),
            "y": float(row["y"]),
            "title": _clean_text(row.get("Title")),
            "abstract": _clean_text(row.get("abstract")),
            "email": _clean_text(row.get("email")),
            "one_line_summary": _clean_text(row.get("one_line_summary")),
            "research_area": _clean_text(row.get("overall_research_area")),
            "topics": _clean_text(row.get("topics")),
            "subfields": _clean_text(row.get("subfields")),
            "centroid_similarity": None
        })

    points_cache = points

    topic_counts: Dict[int, int] = {}
    policy_counts: Dict[int, int] = {}
    school_set: set[str] = set()
    departments_by_school: Dict[str, set[str]] = {}
    researchers: Dict[int, Dict[str, Any]] = {}
    for point in points:
        topic_id = point["topic"]
        topic_counts[topic_id] = topic_counts.get(topic_id, 0) + 1
        policy_id = _safe_int(point.get("policy_domain"))
        if policy_id is not None:
            policy_counts[policy_id] = policy_counts.get(policy_id, 0) + 1

        school = point.get("school") or ""
        department = point.get("department") or ""
        if school:
            school_set.add(school)
            departments_by_school.setdefault(school, set())
            if department:
                departments_by_school[school].add(department)

        n_id = point.get("n_id")
        if isinstance(n_id, int) and n_id not in researchers:
            researchers[n_id] = {
                "n_id": n_id,
                "name": point.get("name") or f"Researcher {n_id}",
                "school": school,
                "department": department
            }

    topics = []
    policies = []
    map_topics: Dict[str, Any] = {}
    for topic_id in sorted(topic_counts.keys()):
        payload = cluster_descriptions.get(str(topic_id), {})
        topic_name = payload.get("topic_name") or f"Topic {topic_id}"
        topics.append({
            "topic": topic_id,
            "topic_name": topic_name,
            "count": topic_counts[topic_id]
        })
        map_topics[str(topic_id)] = {
            "topic_id": topic_id,
            "topic_name": topic_name,
            "topic_description": payload.get("description"),
            "top_words": payload.get("keywords", [])
        }

    for policy_id in sorted(policy_counts.keys()):
        policies.append({
            "policy_domain": policy_id,
            "policy_name": policy_labels.get(policy_id, f"Policy {policy_id}"),
            "count": policy_counts[policy_id],
        })

    topics_cache = map_topics
    filters_cache = {
        "schools": ["All"] + sorted(school_set),
        "departments_by_school": {
            school: sorted(departments)
            for school, departments in departments_by_school.items()
        },
        "researchers": sorted(researchers.values(), key=lambda item: item["name"].lower()),
        "topics": topics,
        "policies": policies,
    }
    return points_cache, filters_cache, topics_cache


def _build_map_researcher_points_for(
    analysis_global_dir: Path, researcher_umap_df: Optional[pd.DataFrame]
) -> List[Dict[str, Any]]:
    if researcher_umap_df is None:
        return []

    cluster_desc_path = analysis_global_dir / "global_cluster_descriptions.json"
    cluster_descriptions: Dict[str, Dict[str, Any]] = {}
    if cluster_desc_path.exists():
        try:
            with cluster_desc_path.open("r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                cluster_descriptions = loaded
        except Exception:
            cluster_descriptions = {}

    cluster_to_policy: Dict[int, int] = {}
    policy_labels: Dict[int, str] = {}
    stage3_cluster_policy_path = ANALYSIS_STAGE3_DIR / "global_cluster_policy_weights.csv"
    stage3_policy_meta_path = ANALYSIS_STAGE3_DIR / "policy_domains_metadata.json"
    if stage3_cluster_policy_path.exists():
        try:
            cdf = pd.read_csv(stage3_cluster_policy_path)
            cdf["cluster_id"] = pd.to_numeric(cdf.get("cluster_id"), errors="coerce").astype("Int64")
            cdf["hard_policy_domain"] = pd.to_numeric(cdf.get("hard_policy_domain"), errors="coerce").astype("Int64")
            cdf = cdf.dropna(subset=["cluster_id", "hard_policy_domain"])
            cluster_to_policy = {int(r["cluster_id"]): int(r["hard_policy_domain"]) for _, r in cdf.iterrows()}
        except Exception:
            cluster_to_policy = {}
    if stage3_policy_meta_path.exists():
        try:
            with stage3_policy_meta_path.open("r", encoding="utf-8") as f:
                meta = json.load(f)
            if isinstance(meta, list):
                for row in meta:
                    pid = _safe_int(row.get("policy_domain_id"))
                    if pid is None:
                        continue
                    policy_labels[pid] = _clean_text(row.get("title")) or f"Policy {pid}"
        except Exception:
            policy_labels = {}

    coords = researcher_umap_df.copy()
    coords["n_id"] = pd.to_numeric(coords.get("n_id"), errors="coerce").astype("Int64")
    coords["topic"] = pd.to_numeric(coords.get("cluster"), errors="coerce").astype("Int64")
    coords["x"] = pd.to_numeric(coords.get("umap_x"), errors="coerce")
    coords["y"] = pd.to_numeric(coords.get("umap_y"), errors="coerce")
    coords = coords.dropna(subset=["n_id", "topic", "x", "y"])

    if profiles_df is not None:
        coords = coords.merge(
            profiles_df[["n_id", "display_name", "school", "department", "email", "biography", "research_interests"]],
            on="n_id",
            how="left",
            suffixes=("_umap", "_profile")
        )
    if researcher_summary_df is not None:
        summary_cols = ["n_id", "one_line_summary", "overall_research_area", "topics", "subfields"]
        existing = [c for c in summary_cols if c in researcher_summary_df.columns]
        if len(existing) > 1:
            coords = coords.merge(
                researcher_summary_df[existing],
                on="n_id",
                how="left"
            )

    points: List[Dict[str, Any]] = []
    for _, row in coords.iterrows():
        n_id = _safe_int(row.get("n_id"))
        topic_id = _safe_int(row.get("topic"))
        if n_id is None or topic_id is None:
            continue
        topic_meta = cluster_descriptions.get(str(topic_id), {})
        name = _clean_text(row.get("display_name")) or f"Researcher {n_id}"
        school_val = _clean_text(row.get("school_profile")) or _clean_text(row.get("school_umap")) or _clean_text(row.get("school"))
        department_val = _clean_text(row.get("department_profile")) or _clean_text(row.get("department_umap")) or _clean_text(row.get("department"))
        email_val = _clean_text(row.get("email")) or _clean_text(row.get("email_profile"))
        policy_id = cluster_to_policy.get(topic_id)
        points.append({
            "key": f"{n_id}:researcher",
            "n_id": n_id,
            "publication_index": None,
            "name": name,
            "school": school_val,
            "department": department_val,
            "topic": topic_id,
            "topic_name": topic_meta.get("topic_name") or f"Topic {topic_id}",
            "policy_domain": policy_id,
            "policy_name": policy_labels.get(policy_id, f"Policy {policy_id}") if policy_id is not None else None,
            "x": float(row["x"]),
            "y": float(row["y"]),
            "title": name,
            "abstract": "",
            "email": email_val,
            "one_line_summary": _clean_text(row.get("one_line_summary")),
            "research_area": _clean_text(row.get("overall_research_area")),
            "topics": _clean_text(row.get("topics")),
            "subfields": _clean_text(row.get("subfields")),
            "centroid_similarity": None,
        })
    return points


def _school_key(value: str) -> str:
    raw = (value or "").strip().lower()
    # Preserve comma boundaries as double underscores to match school folder naming.
    raw = raw.replace(", ", "__").replace(",", "__")
    s = re.sub(r"[^a-z0-9_]+", "_", raw)
    s = re.sub(r"_{3,}", "__", s)
    s = re.sub(r"_+", "_", s.replace("__", "§§")).replace("§§", "__")
    return s.strip("_")


def _strip_cluster_prefix(text: str) -> str:
    s = _clean_text(text)
    if not s:
        return s
    patterns = [
        r"^This\s+(thematic\s+)?(academic\s+)?cluster\s+(explores|examines|focuses on|addresses|discusses|highlights|investigates)\s+",
        r"^This\s+macro[-\s]?theme\s+(explores|examines|focuses on|addresses|discusses|highlights|investigates)\s+",
        r"^This\s+macro[-\s]?theme\s+",
        r"^This\s+cluster\s+",
        r"^The\s+cluster\s+",
    ]
    out = s
    for pat in patterns:
        out = re.sub(pat, "", out, flags=re.IGNORECASE).strip()
    if out:
        out = out[0].upper() + out[1:]
    return out


def _to_app_policy_relevance(text: str, funding_domains: Optional[List[str]] = None) -> str:
    base = _clean_text(text)
    base = re.sub(
        r"^(The|This)\s+theme\s+(addresses|focuses on|explores|examines|highlights|considers)\s+",
        "",
        base,
        flags=re.IGNORECASE,
    ).strip()
    if base:
        if base[0].isupper():
            base = base[0].lower() + base[1:]
        if base and base[-1] not in ".!?":
            base += "."
        return f"TRISS researchers in this area produce evidence directly relevant to {base}"

    domains = [d for d in (funding_domains or []) if _clean_text(d)]
    if domains:
        if len(domains) == 1:
            domain_text = domains[0]
        elif len(domains) == 2:
            domain_text = f"{domains[0]} and {domains[1]}"
        else:
            domain_text = ", ".join(domains[:-1]) + f", and {domains[-1]}"
        return f"TRISS researchers in this area produce evidence directly relevant to {domain_text}."

    return ""


def _build_report_payloads() -> None:
    global report_schools_cache, report_school_detail_cache, report_statistics_cache
    global report_key_areas_cache, report_topic_experts_cache
    global report_school_topics_cache, report_school_topic_experts_cache
    global report_macro_themes_cache, report_macro_theme_experts_cache

    report_schools_cache = []
    report_school_detail_cache = {}
    report_statistics_cache = {}
    report_key_areas_cache = []
    report_topic_experts_cache = {}
    report_school_topics_cache = {}
    report_school_topic_experts_cache = {}
    report_macro_themes_cache = []
    report_macro_theme_experts_cache = {}

    if profiles_df is None:
        return

    profiles = profiles_df.copy()
    profiles["school"] = profiles["school"].fillna("").astype(str)
    profiles["department"] = profiles["department"].fillna("").astype(str)
    profiles["n_id"] = pd.to_numeric(profiles["n_id"], errors="coerce").astype("Int64")
    if "n_publications" in profiles.columns:
        profiles["n_publications"] = pd.to_numeric(profiles["n_publications"], errors="coerce").fillna(0)
    else:
        profiles["n_publications"] = 0
    profiles = profiles.dropna(subset=["n_id"])
    profiles["n_id"] = profiles["n_id"].astype(int)

    pubs = publications_df.copy() if publications_df is not None else pd.DataFrame(columns=["n_id", "school", "department"])
    if "n_id" not in pubs.columns:
        pubs["n_id"] = pd.Series(dtype="Int64")
    if "school" not in pubs.columns:
        pubs["school"] = ""
    if "department" not in pubs.columns:
        pubs["department"] = ""
    pubs["n_id"] = pd.to_numeric(pubs["n_id"], errors="coerce").astype("Int64")
    pubs = pubs.dropna(subset=["n_id"])
    pubs["n_id"] = pubs["n_id"].astype(int)
    pubs["school"] = pubs["school"].fillna("").astype(str)
    pubs["department"] = pubs["department"].fillna("").astype(str)
    if "Title" in pubs.columns:
        pubs = pubs[pubs["Title"].fillna("").astype(str).str.strip() != ""]

    profile_school_map = profiles.set_index("n_id")["school"].to_dict()
    profile_dept_map = profiles.set_index("n_id")["department"].to_dict()
    pubs.loc[pubs["school"].str.strip() == "", "school"] = pubs["n_id"].map(profile_school_map).fillna("")
    pubs.loc[pubs["department"].str.strip() == "", "department"] = pubs["n_id"].map(profile_dept_map).fillna("")
    pubs = pubs[pubs["school"].fillna("").astype(str).str.strip() != ""]

    active_nids = set(pubs["n_id"].unique().tolist())

    school_rows: List[Dict[str, Any]] = []
    school_stats_rows: List[Dict[str, Any]] = []
    schools_sorted = sorted([s for s in profiles["school"].dropna().unique().tolist() if str(s).strip()])
    synthesis_school = distilled_synthesis.get("school_themes", {}) if isinstance(distilled_synthesis, dict) else {}

    for school in schools_sorted:
        prof_school = profiles[profiles["school"] == school]
        pub_school = pubs[pubs["school"] == school]
        total_researchers = int(len(prof_school))
        active_researchers = int(prof_school["n_id"].isin(active_nids).sum())
        total_publications_all = int(prof_school["n_publications"].sum())
        total_publications_recent = int(len(pub_school))
        departments_count = int(prof_school["department"].nunique())
        school_key = _school_key(school)
        school_themes = synthesis_school.get(school_key, [])
        if not school_themes and school_key.replace("__", "_") in synthesis_school:
            school_themes = synthesis_school.get(school_key.replace("__", "_"), [])
        cleaned_themes = []
        for theme in school_themes:
            cleaned_themes.append({
                "theme_name": theme.get("theme_name", ""),
                "description": _strip_cluster_prefix(theme.get("description", "")),
            })

        school_rows.append({
            "key": school,
            "name": school,
            "departments": departments_count,
            "researchers": total_researchers,
            "active_researchers": active_researchers,
            "publications": total_publications_all,
            "publications_recent": total_publications_recent,
        })
        school_stats_rows.append({
            "school": school,
            "total": total_researchers,
            "active": active_researchers,
            "publications_all": total_publications_all,
            "publications_recent": total_publications_recent,
        })

        departments_payload: List[Dict[str, Any]] = []
        for dept in sorted([d for d in prof_school["department"].dropna().unique().tolist() if str(d).strip()]):
            dept_profiles = prof_school[prof_school["department"] == dept]
            dept_pubs = pub_school[pub_school["department"] == dept]
            dept_total = int(len(dept_profiles))
            dept_active = int(dept_profiles["n_id"].isin(active_nids).sum())
            dept_pub_recent = int(len(dept_pubs))
            dept_pub_all = int(dept_profiles["n_publications"].sum())
            departments_payload.append({
                "department": dept,
                "researchers": dept_total,
                "publications": dept_pub_all,
                "publications_recent": dept_pub_recent,
                "avg_pubs": round(dept_pub_all / dept_total, 1) if dept_total else 0,
                "share_active": round((dept_active / dept_total) * 100) if dept_total else 0,
            })

        report_school_detail_cache[school] = {
            "key": school,
            "name": school,
            "researchers": total_researchers,
            "publications": total_publications_all,
            "publications_recent": total_publications_recent,
            "active_researchers": active_researchers,
            "themes": cleaned_themes,
            "policy_domains": [],
            "departments": departments_payload,
            "school_topics": [],
        }

    report_schools_cache = sorted(school_rows, key=lambda r: r["researchers"], reverse=True)

    total_researchers = int(len(profiles))
    total_publications = int(profiles["n_publications"].sum())
    total_active = int(profiles["n_id"].isin(active_nids).sum())
    school_productivity_rows = []
    for row in school_stats_rows:
        total = int(row["total"])
        active = int(row["active"])
        publications_all = int(row["publications_all"])
        publications_recent = int(row["publications_recent"])
        avg_pubs_all = round(publications_all / total, 1) if total else 0
        avg_pubs_active = round(publications_recent / active, 1) if active else 0
        share_active = round((active / total) * 100, 1) if total else 0
        school_productivity_rows.append({
            "school": row["school"],
            "total": total,
            "active": active,
            "publications_all": publications_all,
            "publications_recent": publications_recent,
            "avg_pubs_all": avg_pubs_all,
            "avg_pubs_active": avg_pubs_active,
            "share_active": share_active,
        })

    largest_school = max(school_productivity_rows, key=lambda r: r["total"], default=None)
    most_productive_school = max(school_productivity_rows, key=lambda r: r["avg_pubs_all"], default=None)
    total_recent_publications = int(len(pubs))

    report_statistics_cache = {
        "summary": {
            "total_researchers": total_researchers,
            "total_active_researchers": total_active,
            "total_publications": total_publications,
            "total_recent_publications": total_recent_publications,
            "num_schools": int(profiles["school"].nunique()),
            "num_departments": int(profiles["department"].nunique()),
            "avg_pubs_per_researcher": round(total_publications / total_researchers, 1) if total_researchers else 0,
            "percent_recent": round((total_recent_publications / total_publications) * 100, 1) if total_publications else 0,
            "largest_school": largest_school,
            "most_productive_school": most_productive_school,
        },
        "researchers_by_school": school_stats_rows,
        "publications_by_school": sorted(school_productivity_rows, key=lambda r: r["publications_all"], reverse=True),
        "school_productivity": sorted(school_productivity_rows, key=lambda r: r["avg_pubs_all"], reverse=True),
    }

    # Key Areas + Top Semantic Experts (topic-centroid ranking in OpenAI embedding space).
    cluster_desc: Dict[str, Dict[str, Any]] = {}
    if (ANALYSIS_GLOBAL_DIR / "global_cluster_descriptions.json").exists():
        with (ANALYSIS_GLOBAL_DIR / "global_cluster_descriptions.json").open("r", encoding="utf-8") as fh:
            loaded = json.load(fh)
            if isinstance(loaded, dict):
                cluster_desc = loaded

    topic_counts: Dict[int, int] = {}
    assignments_df = None
    if GLOBAL_CLUSTER_ASSIGNMENTS_PATH.exists():
        assignments_df = pd.read_csv(GLOBAL_CLUSTER_ASSIGNMENTS_PATH)
        if "cluster" in assignments_df.columns:
            assignments_df["cluster"] = pd.to_numeric(assignments_df["cluster"], errors="coerce").astype("Int64")
            topic_counts = assignments_df["cluster"].value_counts(dropna=True).to_dict()

    openai_vectors: Dict[int, np.ndarray] = {}
    if OPENAI_RESEARCHER_EMBED_DIR.exists():
        for f in OPENAI_RESEARCHER_EMBED_DIR.glob("n_*.json"):
            try:
                with f.open("r", encoding="utf-8") as fh:
                    obj = json.load(fh)
                n_id = _safe_int(obj.get("n_id"))
                vec = obj.get("embeddings", {}).get("abstracts_mean")
                if n_id is None or vec is None:
                    continue
                openai_vectors[n_id] = _normalize_vector(np.asarray(vec, dtype=np.float32))
            except Exception:
                continue

    centroids = None
    if GLOBAL_CLUSTER_CENTROIDS_PATH.exists():
        centroids = np.load(GLOBAL_CLUSTER_CENTROIDS_PATH).astype(np.float32)
        centroids = np.vstack([_normalize_vector(row) for row in centroids])

    assigned_topic_map: Dict[int, int] = {}
    if assignments_df is not None and {"n_id", "cluster"}.issubset(assignments_df.columns):
        a = assignments_df.dropna(subset=["n_id", "cluster"]).copy()
        a["n_id"] = pd.to_numeric(a["n_id"], errors="coerce").astype("Int64")
        a["cluster"] = pd.to_numeric(a["cluster"], errors="coerce").astype("Int64")
        a = a.dropna(subset=["n_id", "cluster"])
        assigned_topic_map = {int(r["n_id"]): int(r["cluster"]) for _, r in a.iterrows()}

    if centroids is not None and len(openai_vectors):
        n_ids = np.array(sorted(openai_vectors.keys()), dtype=int)
        mat = np.vstack([openai_vectors[int(n)] for n in n_ids])
        sims = mat @ centroids.T

        for topic_id in range(centroids.shape[0]):
            key = str(topic_id)
            payload = cluster_desc.get(key, {})
            report_key_areas_cache.append({
                "topic_id": topic_id,
                "theme_name": payload.get("topic_name") or f"Topic {topic_id}",
                "description": _strip_cluster_prefix(payload.get("description") or ""),
                "keywords": payload.get("keywords", []),
                "count": int(topic_counts.get(topic_id, 0)),
            })

            topic_scores = sims[:, topic_id]
            top_idx = _topk_indices(topic_scores, 20)
            experts: List[Dict[str, Any]] = []
            for idx in top_idx:
                nid = int(n_ids[int(idx)])
                meta = researcher_meta_cache.get(nid, {})
                experts.append({
                    "n_id": nid,
                    "name": meta.get("name", f"Researcher {nid}"),
                    "school": meta.get("school", ""),
                    "department": meta.get("department", ""),
                    "similarity": float(topic_scores[int(idx)]),
                    "assigned_topic": assigned_topic_map.get(nid),
                })
            report_topic_experts_cache[topic_id] = experts

    report_key_areas_cache = sorted(report_key_areas_cache, key=lambda r: (r["topic_id"]))

    # Stage-3 policy domains replace Stage-2 macro themes in report Themes tab.
    stage3_meta_path = ANALYSIS_STAGE3_DIR / "policy_domains_metadata.json"
    stage3_cluster_weights_path = ANALYSIS_STAGE3_DIR / "global_cluster_policy_weights.csv"
    stage3_researcher_weights_path = ANALYSIS_STAGE3_DIR / "researcher_policy_weights.csv"
    key_areas_by_id = {int(item["topic_id"]): item for item in report_key_areas_cache}
    policy_domain_rows: List[Dict[str, Any]] = []
    cluster_policy_df = None
    if stage3_meta_path.exists():
        try:
            with stage3_meta_path.open("r", encoding="utf-8") as fh:
                loaded = json.load(fh)
                if isinstance(loaded, list):
                    policy_domain_rows = loaded
        except Exception:
            policy_domain_rows = []
    if stage3_cluster_weights_path.exists():
        try:
            cluster_policy_df = pd.read_csv(stage3_cluster_weights_path)
            cluster_policy_df["cluster_id"] = pd.to_numeric(cluster_policy_df.get("cluster_id"), errors="coerce").astype("Int64")
            cluster_policy_df["hard_policy_domain"] = pd.to_numeric(cluster_policy_df.get("hard_policy_domain"), errors="coerce").astype("Int64")
            cluster_policy_df = cluster_policy_df.dropna(subset=["cluster_id", "hard_policy_domain"])
        except Exception:
            cluster_policy_df = None

    domain_counts: Dict[int, int] = {}
    researcher_policy_df = None
    if stage3_researcher_weights_path.exists():
        try:
            researcher_policy_df = pd.read_csv(stage3_researcher_weights_path)
            researcher_policy_df["n_id"] = pd.to_numeric(researcher_policy_df.get("n_id"), errors="coerce").astype("Int64")
            researcher_policy_df["hard_policy_domain"] = pd.to_numeric(researcher_policy_df.get("hard_policy_domain"), errors="coerce").astype("Int64")
            researcher_policy_df = researcher_policy_df.dropna(subset=["n_id", "hard_policy_domain"])
            domain_counts = researcher_policy_df["hard_policy_domain"].value_counts(dropna=True).to_dict()
        except Exception:
            researcher_policy_df = None

    if policy_domain_rows:
        for row in sorted(policy_domain_rows, key=lambda x: _safe_int(x.get("order_index")) or 999):
            domain_id = _safe_int(row.get("policy_domain_id"))
            if domain_id is None:
                continue
            aligned_area_ids: List[int] = []
            if cluster_policy_df is not None:
                sub = cluster_policy_df[cluster_policy_df["hard_policy_domain"] == domain_id].copy()
                score_col = f"policy_domain_{domain_id}"
                if score_col in sub.columns:
                    sub[score_col] = pd.to_numeric(sub[score_col], errors="coerce").fillna(0.0)
                    sub = sub.sort_values(by=score_col, ascending=False)
                aligned_area_ids = [
                    int(v) for v in sub["cluster_id"].tolist()
                    if _safe_int(v) is not None
                ]
            aligned_areas: List[Dict[str, Any]] = []
            for tid in aligned_area_ids:
                area = key_areas_by_id.get(tid)
                if area:
                    aligned_areas.append(area)
                else:
                    aligned_areas.append({
                        "topic_id": tid,
                        "theme_name": f"Topic {tid}",
                        "description": "",
                        "keywords": [],
                        "count": int(topic_counts.get(tid, 0)),
                    })

            report_macro_themes_cache.append({
                "theme_id": domain_id,
                "title": _clean_text(row.get("title")),
                "description": _strip_cluster_prefix(row.get("description") or ""),
                "policy_relevance": "",
                "keywords": [],
                "funding_domains": [],
                "methodological_approaches": [],
                "schools_contributing": [],
                "n_researchers": int(domain_counts.get(domain_id, 0)),
                "stage1_cluster_ids": aligned_area_ids,
                "aligned_research_areas": aligned_areas,
            })
        report_macro_themes_cache = sorted(report_macro_themes_cache, key=lambda x: int(x["theme_id"]))

    if researcher_policy_df is not None:
        for theme in report_macro_themes_cache:
            theme_id = int(theme["theme_id"])
            score_col = f"policy_domain_{theme_id}"
            subset = researcher_policy_df[researcher_policy_df["hard_policy_domain"] == theme_id].copy()
            if score_col in subset.columns:
                subset[score_col] = pd.to_numeric(subset[score_col], errors="coerce").fillna(0.0)
                subset = subset.sort_values(by=score_col, ascending=False)
            experts: List[Dict[str, Any]] = []
            for _, r in subset.head(50).iterrows():
                n_id = int(r["n_id"])
                meta = researcher_meta_cache.get(n_id, {})
                experts.append({
                    "n_id": n_id,
                    "name": meta.get("name", f"Researcher {n_id}"),
                    "school": meta.get("school", ""),
                    "department": meta.get("department", ""),
                    "similarity": float(r.get(score_col, 0.0)) if score_col in r else 0.0,
                    "assigned_topic": assigned_topic_map.get(n_id),
                })
            report_macro_theme_experts_cache[theme_id] = experts

    # School-specific topic experts (using school-level centroids; experts constrained to school members).
    if len(openai_vectors):
        for school in report_school_detail_cache.keys():
            school_slug = _school_key(school)
            school_dir = ANALYSIS_SCHOOLS_DIR / school_slug
            if not school_dir.exists():
                continue
            desc_file = school_dir / "school_cluster_descriptions.json"
            centroid_file = school_dir / "school_cluster_centroids.npy"
            assign_file = school_dir / "school_cluster_assignments.csv"
            counts_file = school_dir / "school_topic_counts.csv"
            if not (desc_file.exists() and centroid_file.exists() and assign_file.exists()):
                continue
            try:
                with desc_file.open("r", encoding="utf-8") as fh:
                    school_desc = json.load(fh)
                school_centroids = np.load(centroid_file).astype(np.float32)
                school_centroids = np.vstack([_normalize_vector(row) for row in school_centroids])
                school_assign_df = pd.read_csv(assign_file)
                school_assign_df["n_id"] = pd.to_numeric(school_assign_df.get("n_id"), errors="coerce").astype("Int64")
                school_assign_df["cluster"] = pd.to_numeric(school_assign_df.get("cluster"), errors="coerce").astype("Int64")
                school_assign_df = school_assign_df.dropna(subset=["n_id", "cluster"])
                school_nids = sorted([int(n) for n in school_assign_df["n_id"].unique().tolist() if int(n) in openai_vectors])
                if not school_nids:
                    continue
                mat = np.vstack([openai_vectors[n] for n in school_nids])
                sims = mat @ school_centroids.T
                assign_map = {int(r["n_id"]): int(r["cluster"]) for _, r in school_assign_df.iterrows()}
                topic_counts_local: Dict[int, int] = {}
                if counts_file.exists():
                    cdf = pd.read_csv(counts_file)
                    if {"cluster", "size"}.issubset(cdf.columns):
                        for _, r in cdf.iterrows():
                            topic_counts_local[int(r["cluster"])] = int(r["size"])

                topics_payload: List[Dict[str, Any]] = []
                topic_experts_payload: Dict[int, List[Dict[str, Any]]] = {}
                for topic_id in range(school_centroids.shape[0]):
                    desc = school_desc.get(str(topic_id), {}) if isinstance(school_desc, dict) else {}
                    topics_payload.append({
                        "topic_id": topic_id,
                        "theme_name": desc.get("topic_name") or f"Topic {topic_id}",
                        "description": _strip_cluster_prefix(desc.get("description") or ""),
                        "keywords": desc.get("keywords", []),
                        "count": int(topic_counts_local.get(topic_id, 0)),
                    })
                    scores = sims[:, topic_id]
                    top_idx = _topk_indices(scores, 20)
                    topic_experts: List[Dict[str, Any]] = []
                    for idx in top_idx:
                        nid = int(school_nids[int(idx)])
                        meta = researcher_meta_cache.get(nid, {})
                        topic_experts.append({
                            "n_id": nid,
                            "name": meta.get("name", f"Researcher {nid}"),
                            "school": meta.get("school", ""),
                            "department": meta.get("department", ""),
                            "similarity": float(scores[int(idx)]),
                            "assigned_topic": assign_map.get(nid),
                        })
                    topic_experts_payload[topic_id] = topic_experts

                report_school_topics_cache[school] = topics_payload
                report_school_topic_experts_cache[school] = topic_experts_payload
                if school in report_school_detail_cache:
                    report_school_detail_cache[school]["school_topics"] = topics_payload
            except Exception:
                continue


def _make_publication_result(
    pub_row: pd.Series,
    similarity: float,
    researcher_name: str,
) -> Dict[str, Any]:
    abstract_text = _clean_text(pub_row.get("abstract", ""))
    snippet = abstract_text[:360] + ("..." if len(abstract_text) > 360 else "")
    n_id = _safe_int(pub_row.get("n_id")) or 0
    publication_index = _extract_publication_index(str(pub_row.get("article_id", "")))
    return {
        "n_id": n_id,
        "researcher_name": researcher_name,
        "school": _clean_text(researcher_meta_cache.get(n_id, {}).get("school", "")),
        "department": _clean_text(researcher_meta_cache.get(n_id, {}).get("department", "")),
        "publication_index": publication_index,
        "article_id": str(pub_row.get("article_id", "")),
        "title": _clean_text(pub_row.get("Title", "")) or "Untitled",
        "abstract_snippet": snippet if snippet else None,
        "main_url": _clean_text(pub_row.get("main_url", "")) or None,
        "similarity": float(similarity),
    }


def _rank_theme_publications(
    pubs: pd.DataFrame,
    topic_lookup: Dict[str, int],
    target_topic: int,
    researcher_scores: Dict[int, float],
    top_k: int,
) -> List[Dict[str, Any]]:
    if pubs.empty:
        return []
    ranked: Dict[str, Dict[str, Any]] = {}
    for _, row in pubs.iterrows():
        article_id = str(row.get("article_id", "")).strip()
        if not article_id:
            continue
        topic = topic_lookup.get(article_id)
        if topic != target_topic:
            continue
        n_id = _safe_int(row.get("n_id")) or 0
        score = float(researcher_scores.get(n_id, 0.0))
        candidate = _make_publication_result(row, score, _clean_text(researcher_meta_cache.get(n_id, {}).get("name", "")))
        existing = ranked.get(article_id)
        if existing is None or candidate["similarity"] > existing["similarity"]:
            ranked[article_id] = candidate
    ordered = sorted(ranked.values(), key=lambda r: r.get("similarity", 0.0), reverse=True)
    return ordered[:top_k]


def _rank_publications_by_researcher_scores(
    pubs: pd.DataFrame,
    researcher_scores: Dict[int, float],
    top_k: int,
) -> List[Dict[str, Any]]:
    if pubs.empty or not researcher_scores:
        return []
    ranked: Dict[str, Dict[str, Any]] = {}
    for _, row in pubs.iterrows():
        article_id = str(row.get("article_id", "")).strip()
        if not article_id:
            continue
        n_id = _safe_int(row.get("n_id")) or 0
        if n_id not in researcher_scores:
            continue
        score = float(researcher_scores.get(n_id, 0.0))
        candidate = _make_publication_result(
            row,
            score,
            _clean_text(researcher_meta_cache.get(n_id, {}).get("name", "")),
        )
        existing = ranked.get(article_id)
        if existing is None or candidate["similarity"] > existing["similarity"]:
            ranked[article_id] = candidate
    ordered = sorted(ranked.values(), key=lambda r: r.get("similarity", 0.0), reverse=True)
    return ordered[:top_k]


def _build_expert_theme_options() -> None:
    global expert_theme_options_cache
    options: List[Dict[str, Any]] = []

    for topic in sorted(report_key_areas_cache, key=lambda x: (x.get("theme_name", "") or "").lower()):
        topic_id = _safe_int(topic.get("topic_id"))
        if topic_id is None:
            continue
        options.append({
            "option_id": f"global:{topic_id}",
            "label": f"TRISS - {topic.get('theme_name', f'Topic {topic_id}')}",
            "scope": "global",
            "school": None,
            "school_label": None,
            "topic_id": topic_id,
        })

    for school in sorted(report_school_topics_cache.keys(), key=lambda s: s.lower()):
        school_topics = report_school_topics_cache.get(school, [])
        for topic in sorted(school_topics, key=lambda x: (x.get("theme_name", "") or "").lower()):
            topic_id = _safe_int(topic.get("topic_id"))
            if topic_id is None:
                continue
            options.append({
                "option_id": f"school:{school}:{topic_id}",
                "label": f"{school} - {topic.get('theme_name', f'Topic {topic_id}')}",
                "scope": "school",
                "school": school,
                "school_label": school,
                "topic_id": topic_id,
            })
    for policy in sorted(report_macro_themes_cache, key=lambda x: (x.get("title", "") or "").lower()):
        theme_id = _safe_int(policy.get("theme_id"))
        if theme_id is None:
            continue
        options.append({
            "option_id": f"policy:{theme_id}",
            "label": _clean_text(policy.get("title")) or f"Policy Domain {theme_id}",
            "scope": "policy",
            "school": None,
            "school_label": None,
            "topic_id": theme_id,
        })
    expert_theme_options_cache = options


def _build_expert_search_filters() -> None:
    global expert_search_filters_cache
    if profiles_df is None:
        expert_search_filters_cache = {"schools": [], "departments_by_school": {}}
        return
    schools = sorted([s for s in profiles_df["school"].dropna().astype(str).unique().tolist() if s.strip()])
    departments_by_school: Dict[str, List[str]] = {}
    for school in schools:
        departments = sorted([
            d for d in profiles_df.loc[profiles_df["school"] == school, "department"].dropna().astype(str).unique().tolist()
            if d.strip()
        ])
        departments_by_school[school] = departments
    expert_search_filters_cache = {"schools": schools, "departments_by_school": departments_by_school}


def _normalize_filter_value(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    s = str(value).strip()
    if not s or s.lower() == "all":
        return None
    return s


def _researcher_matches_filters(n_id: int, school: Optional[str], department: Optional[str]) -> bool:
    meta = researcher_meta_cache.get(int(n_id), {})
    if school and _clean_text(meta.get("school")) != _clean_text(school):
        return False
    if department and _clean_text(meta.get("department")) != _clean_text(department):
        return False
    return True


# ---------------------------------------------------------------------
# STARTUP
# ---------------------------------------------------------------------
@app.on_event("startup")
async def load_data():
    global profiles_df, researcher_summary_df, similarity_matrix_df, pub_similarity_df, publications_df, distilled_synthesis
    global umap_coords_df, umap_coords_mpnet_df, pub_umap_coords_df, pub_umap_coords_mpnet_df
    global map_researcher_points_cache, map_v2_researcher_points_cache
    TRISS_DATA_DIR.mkdir(parents=True, exist_ok=True)
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Loading unified TRISS data from {FINAL_DIR} ...")

    missing_required = _run_runtime_initializer_if_needed()
    if missing_required:
        profiles_df = None
        print("[startup] Skipping data load; required runtime files are missing.")
        return

    try:
        # 1. Profiles
        profiles_df = pd.read_csv(PROFILES_DIR / "1. profiles_summary.csv")
        profiles_df["n_id"] = pd.to_numeric(profiles_df["n_id"], errors="coerce")
        profiles_df = profiles_df.dropna(subset=["n_id"])
        profiles_df["n_id"] = profiles_df["n_id"].astype(int)
        profiles_df["email_norm"] = profiles_df["email"].apply(_norm_text)
        profiles_df["lastname_clean"] = profiles_df["lastname"].astype(str).str.replace("_", " ", regex=False).str.strip()
        profiles_df["firstname_clean"] = profiles_df["firstname"].astype(str).str.strip()
        profiles_df["display_name"] = profiles_df.apply(lambda r: _titlecase_name(r["firstname_clean"], r["lastname_clean"]), axis=1)
        profiles_df["name_norm"] = profiles_df["display_name"].apply(_norm_text)

        # 2. Researcher Summaries (LLM one-liners)
        summ_path = PROFILES_DIR / "4. triss_researcher_summary.csv"
        if summ_path.exists():
            researcher_summary_df = pd.read_csv(summ_path)
            researcher_summary_df["n_id"] = researcher_summary_df["n_id"].astype(int)

        # 3. Themes
        synth_path = ANALYSIS_GLOBAL_DIR / "triss_distilled_synthesis_v3.json"
        if synth_path.exists():
            with open(synth_path, "r") as f:
                distilled_synthesis = json.load(f)

        # 4. Publications (for own-pub retrieval - in pipeline folder)
        pub_path = PUBLICATIONS_DIR / "4.All measured publications.csv"
        if pub_path.exists():
            publications_df = pd.read_csv(pub_path)
            publications_df["n_id"] = pd.to_numeric(publications_df["n_id"], errors="coerce").fillna(0).astype(int)
            publications_df["article_id"] = publications_df["article_id"].astype(str)
            publications_df["title_norm"] = publications_df["Title"].apply(_norm_text) if "Title" in publications_df.columns else ""
            print(f"Loaded {len(publications_df)} publications with abstracts.")
        else:
            fallback_pub_path = PUBLICATIONS_DIR / "3. All listed publications 2019 +.csv"
            if fallback_pub_path.exists():
                publications_df = pd.read_csv(fallback_pub_path)
                publications_df["n_id"] = pd.to_numeric(publications_df["n_id"], errors="coerce").fillna(0).astype(int)
                publications_df["article_id"] = publications_df["article_id"].astype(str)
                publications_df["title_norm"] = publications_df["Title"].apply(_norm_text) if "Title" in publications_df.columns else ""
                print(f"Loaded {len(publications_df)} publications (no abstracts).")

        # 5. Similarity Matrix (researcher-level cosine similarities)
        sim_path = NETWORK_DIR / "7.researcher_similarity_matrix.csv"
        if sim_path.exists():
            similarity_matrix_df = pd.read_csv(sim_path, index_col=0)
            similarity_matrix_df.index = similarity_matrix_df.index.astype(int)
            similarity_matrix_df.columns = similarity_matrix_df.columns.astype(int)
            print(f"Loaded similarity matrix: {similarity_matrix_df.shape}")
        else:
            print(f"WARNING: similarity matrix not found at {sim_path}")

        # 6. Publication Similarity (query_n_id/target_n_id/title/similarity schema)
        pub_sim_path = NETWORK_DIR / "8. user_to_other_publication_similarity_openai.csv"
        if pub_sim_path.exists():
            print("Loading publication similarity (this may take a moment)...")
            pub_similarity_df = pd.read_csv(pub_sim_path)
            pub_similarity_df["query_n_id"] = pub_similarity_df["query_n_id"].astype(int)
            pub_similarity_df["target_n_id"] = pub_similarity_df["target_n_id"].astype(int)
            print(f"Loaded pub similarity: {len(pub_similarity_df)} rows")
        else:
            print(f"WARNING: pub similarity not found at {pub_sim_path}")

        # 7. UMAP Coordinates
        umap_path = ANALYSIS_GLOBAL_DIR / "global_umap_coordinates.csv"
        if umap_path.exists():
            umap_coords_df = pd.read_csv(umap_path)
            if "n_id" in umap_coords_df.columns:
                umap_coords_df["n_id"] = umap_coords_df["n_id"].astype(int)
        umap_mpnet_path = ANALYSIS_GLOBAL_MPNET_DIR / "global_umap_coordinates.csv"
        if umap_mpnet_path.exists():
            umap_coords_mpnet_df = pd.read_csv(umap_mpnet_path)
            if "n_id" in umap_coords_mpnet_df.columns:
                umap_coords_mpnet_df["n_id"] = pd.to_numeric(umap_coords_mpnet_df["n_id"], errors="coerce").fillna(0).astype(int)

        pub_umap_path = ANALYSIS_GLOBAL_DIR / "global_publication_umap_coordinates.csv"
        if pub_umap_path.exists():
            pub_umap_coords_df = pd.read_csv(pub_umap_path)
        pub_umap_mpnet_path = ANALYSIS_GLOBAL_MPNET_DIR / "global_publication_umap_coordinates.csv"
        if pub_umap_mpnet_path.exists():
            pub_umap_coords_mpnet_df = pd.read_csv(pub_umap_mpnet_path)

        _build_map_payloads()
        _build_map_payloads_v2()
        map_researcher_points_cache = _build_map_researcher_points_for(ANALYSIS_GLOBAL_DIR, umap_coords_df)
        map_v2_researcher_points_cache = _build_map_researcher_points_for(ANALYSIS_GLOBAL_MPNET_DIR, umap_coords_mpnet_df)
        _load_expert_search_embeddings()
        _build_report_payloads()
        _build_expert_theme_options()
        _build_expert_search_filters()
        _set_startup_status(
            state="loaded",
            message="Runtime data loaded successfully.",
            loaded=True,
            attempted_bootstrap=startup_init_status.get("attempted_bootstrap", False),
            missing_required_files=[],
        )
        print("Data loading complete.")
    except Exception as exc:
        profiles_df = None
        _set_startup_status(
            state="load_failed",
            message=f"Runtime data load failed: {exc}",
            loaded=False,
            attempted_bootstrap=startup_init_status.get("attempted_bootstrap", False),
            missing_required_files=_missing_runtime_files(),
        )
        print(traceback.format_exc())
        print("[startup] Continuing without loaded data; endpoints will return 503 until data is ready.")

# ---------------------------------------------------------------------
# ROUTES: HEALTH
# ---------------------------------------------------------------------
@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "loaded": profiles_df is not None,
        "triss_data_dir": str(TRISS_DATA_DIR),
        "final_dir": str(FINAL_DIR),
        "missing_required_files": _missing_runtime_files(),
        "startup_init_status": startup_init_status,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }

# ---------------------------------------------------------------------
# ROUTES: REPORTING
# ---------------------------------------------------------------------
@app.get("/api/report/overview")
async def get_overview():
    _require_loaded()
    return {
        "core_identity": distilled_synthesis.get("core_identity_summary", ""),
        "cross_cutting_themes": distilled_synthesis.get("cross_cutting_themes", []),
        "stats": {
            "total_researchers": len(profiles_df),
            "num_schools": int(profiles_df["school"].nunique()),
            "num_departments": int(profiles_df["department"].nunique())
        }
    }

@app.get("/api/report/schools")
async def get_schools_list():
    _require_loaded()
    schools = profiles_df.groupby("school").size().reset_index(name="count")
    return schools.to_dict(orient="records")


@app.get("/api/report/v3/overview")
async def get_report_v3_overview():
    _require_loaded()
    cleaned_themes = []
    for theme in distilled_synthesis.get("cross_cutting_themes", []):
        cleaned_themes.append({
            "theme_name": theme.get("theme_name", ""),
            "description": _strip_cluster_prefix(theme.get("description", "")),
        })
    return {
        "core_identity": distilled_synthesis.get("core_identity_summary", ""),
        "cross_cutting_themes": cleaned_themes,
        "stats": report_statistics_cache.get("summary", {}),
    }


@app.get("/api/report/v3/statistics")
async def get_report_v3_statistics():
    _require_loaded()
    return report_statistics_cache


@app.get("/api/report/v3/schools")
async def get_report_v3_schools():
    _require_loaded()
    return report_schools_cache


@app.get("/api/report/v3/school/{school_key}")
async def get_report_v3_school_detail(school_key: str):
    _require_loaded()
    payload = report_school_detail_cache.get(school_key)
    if payload is None:
        raise HTTPException(status_code=404, detail="School not found")
    return payload


@app.get("/api/report/v3/school/{school_key}/topics")
async def get_report_v3_school_topics(school_key: str):
    _require_loaded()
    return report_school_topics_cache.get(school_key, [])


@app.get("/api/report/v3/school/{school_key}/topics/{topic_id}/experts")
async def get_report_v3_school_topic_experts(school_key: str, topic_id: int, limit: int = Query(12, ge=1, le=50)):
    _require_loaded()
    experts = report_school_topic_experts_cache.get(school_key, {}).get(topic_id, [])
    return experts[:limit]


@app.get("/api/report/v3/key-areas")
async def get_report_v3_key_areas():
    _require_loaded()
    return report_key_areas_cache


@app.get("/api/report/v3/key-areas/{topic_id}/experts")
async def get_report_v3_key_area_experts(topic_id: int, limit: int = Query(12, ge=1, le=50)):
    _require_loaded()
    experts = report_topic_experts_cache.get(topic_id, [])
    return experts[:limit]


@app.get("/api/report/v3/themes")
async def get_report_v3_themes():
    _require_loaded()
    return report_macro_themes_cache


@app.get("/api/report/v3/themes/{theme_id}/experts")
async def get_report_v3_theme_experts(theme_id: int, limit: int = Query(12, ge=1, le=50)):
    _require_loaded()
    experts = report_macro_theme_experts_cache.get(theme_id, [])
    return experts[:limit]


@app.get("/api/report/v3/pdf")
async def get_report_v3_pdf():
    if not REPORT_V3_PDF_PATH.exists():
        raise HTTPException(status_code=404, detail="Report PDF not found")
    return FileResponse(REPORT_V3_PDF_PATH, media_type="application/pdf", filename="triss-report-v3.pdf")

# ---------------------------------------------------------------------
# ROUTES: NETWORK / SEARCH
# ---------------------------------------------------------------------
@app.get("/api/network/search")
async def search(q: str = Query(..., min_length=2)):
    _require_loaded()
    query = _norm_text(q)
    
    # Matching logic similar to legacy app for better UX
    # 0) exact email match, 1) email startswith, 2) name startswith, 3) name contains
    
    m_email_exact = profiles_df["email_norm"] == query
    m_email_starts = profiles_df["email_norm"].str.startswith(query, na=False)
    m_name_starts = profiles_df["name_norm"].str.startswith(query, na=False)
    m_name_contains = profiles_df["name_norm"].str.contains(query, na=False)
    
    mask = m_email_exact | m_email_starts | m_name_starts | m_name_contains
    matches = profiles_df[mask].copy()
    
    if matches.empty: return []
    
    # Assign scores for ranking
    matches["rank_score"] = 4
    matches.loc[m_name_contains, "rank_score"] = 3
    matches.loc[m_name_starts, "rank_score"] = 2
    matches.loc[m_email_starts, "rank_score"] = 1
    matches.loc[m_email_exact, "rank_score"] = 0
    
    matches = matches.sort_values("rank_score").head(10)
    
    results = []
    for _, r in matches.iterrows():
        results.append({
            "n_id": int(r["n_id"]),
            "name": r["display_name"],
            "email": r.get("email", ""),
            "department": r.get("department", ""),
            "school": r.get("school", "")
        })
    return results

@app.post("/api/network/lookup", response_model=LookupResponse)
async def lookup_by_email(request: EmailLookup):
    _require_loaded()
    email = _norm_text(request.email)
    match = profiles_df[profiles_df["email_norm"] == email]
    if match.empty:
        raise HTTPException(status_code=404, detail="Email not found.")
    row = match.iloc[0]
    return LookupResponse(
        n_id=int(row["n_id"]),
        name=row["display_name"],
        department=row.get("department", ""),
        school=row.get("school", "")
    )

@app.get("/api/network/researcher/{n_id}", response_model=ResearcherProfile)
async def get_profile(n_id: int):
    _require_loaded()
    match = profiles_df[profiles_df["n_id"] == n_id]
    if match.empty:
        raise HTTPException(status_code=404, detail="Researcher not found.")
    
    r = match.iloc[0]
    summ = researcher_summary_df[researcher_summary_df["n_id"] == n_id] if researcher_summary_df is not None else None
    
    found_summ = summ.iloc[0] if (summ is not None and not summ.empty) else {}
    
    return ResearcherProfile(
        n_id=n_id,
        name=r["display_name"],
        email=r.get("email", ""),
        department=r.get("department", ""),
        school=r.get("school", ""),
        one_line_summary=found_summ.get("one_line_summary"),
        topics=found_summ.get("topics"),
        research_area=found_summ.get("overall_research_area"),
        subfields=found_summ.get("subfields")
    )

@app.get("/api/network/neighbours/{n_id}", response_model=List[NeighbourResponse])
async def get_neighbours(
    n_id: int, 
    k: int = 10,
    exclude_department: bool = False,
    exclude_school: bool = False
):
    _require_loaded()
    if similarity_matrix_df is None or n_id not in similarity_matrix_df.index:
        return []
    
    user_profile = profiles_df[profiles_df["n_id"] == n_id]
    if user_profile.empty:
        raise HTTPException(status_code=404, detail="Researcher not found.")
    
    u_dept = user_profile.iloc[0].get("department", "")
    u_school = user_profile.iloc[0].get("school", "")

    sims = similarity_matrix_df.loc[n_id].sort_values(ascending=False)
    results = []
    for other_id, score in sims.items():
        if len(results) >= k: break
        other_id = int(other_id)
        if other_id == n_id: continue
        
        p = profiles_df[profiles_df["n_id"] == other_id]
        if p.empty: continue
        row = p.iloc[0]

        if exclude_department and row.get("department", "") == u_dept:
            continue
        if exclude_school and row.get("school", "") == u_school:
            continue

        results.append({
            "n_id": other_id,
            "name": row["display_name"],
            "department": row.get("department", ""),
            "school": row.get("school", ""),
            "similarity": float(score),
            "rank": len(results) + 1
        })
    return results

@app.get("/api/network/publications/{query_n_id}/{target_n_id}", response_model=List[PublicationMatch])
async def get_matching_publications(query_n_id: int, target_n_id: int, limit: int = 5):
    _require_loaded()
    print(f"API: get_matching_publications {query_n_id} -> {target_n_id}")
    try:
        if pub_similarity_df is None: return []

        matches = pub_similarity_df[
            (pub_similarity_df["query_n_id"] == query_n_id) & 
            (pub_similarity_df["target_n_id"] == target_n_id)
        ].copy()

        if matches.empty: return []

        matches = matches.sort_values("similarity", ascending=False).drop_duplicates(subset=["title"]).head(limit)
        
        results = []
        for _, row in matches.iterrows():
            pub_title = _clean_text(row.get("title", "Untitled"))
            similarity_score = round(float(row["similarity"]), 4)
            
            # Try to enrich with main_url and abstract from publications_df by title match
            main_url = None
            abstract = None
            doi = None
            if publications_df is not None:
                title_norm = _norm_text(pub_title)
                # Match on normalised title within target's publications
                cands = publications_df[
                    (publications_df["n_id"] == target_n_id) &
                    (publications_df["title_norm"] == title_norm)
                ]
                if cands.empty:
                    # Fallback: substring match on title (less strict)
                    cands = publications_df[
                        (publications_df["n_id"] == target_n_id) &
                        (publications_df["Title"].str.contains(pub_title[:40], case=False, na=False))
                    ]
                if not cands.empty:
                    p = cands.iloc[0]
                    main_url = str(p.get("main_url")) if pd.notna(p.get("main_url")) else None
                    abstract = _clean_text(p["abstract"]) if pd.notna(p.get("abstract")) else None
                    doi = str(p.get("Dig Ob Id")) if pd.notna(p.get("Dig Ob Id")) else None
                    authors = _clean_text(p.get("Authors")) if pd.notna(p.get("Authors")) else None
                    # Use the enriched title from publications_df if available
                    pub_title = _clean_text(p.get("Title", pub_title))

            results.append(PublicationMatch(
                title=pub_title,
                abstract=abstract,
                similarity=similarity_score,
                doi=doi,
                main_url=main_url,
                authors=authors
            ))
                
        return results
    except Exception as e:
        print(f"Error in matching_publications: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/network/my-publications/{n_id}", response_model=List[OwnPublication])
async def get_own_publications(n_id: int, limit: int = 5):
    _require_loaded()
    print(f"API: get_own_publications for {n_id}")
    try:
        if publications_df is None: return []
        
        pubs = publications_df[publications_df["n_id"] == n_id].copy()
        if pubs.empty: return []

        # Use 'Year Descending' from the V3 schema
        year_col = "Year Descending"
        if year_col in pubs.columns:
            pubs[year_col] = pd.to_numeric(pubs[year_col], errors="coerce")
            pubs = pubs.sort_values(year_col, ascending=False, na_position="last")
        
        # Deduplicate and take top results
        pubs = pubs.drop_duplicates(subset=["Title"]).head(limit)
        
        results = []
        for _, row in pubs.iterrows():
            y = row.get(year_col)
            results.append(OwnPublication(
                title=_clean_text(row.get("Title", "Untitled")),
                abstract=_clean_text(row["abstract"]) if pd.notna(row.get("abstract")) else None,
                year=int(y) if pd.notna(y) else None,
                doi=str(row.get("Dig Ob Id")) if pd.notna(row.get("Dig Ob Id")) else None,
                main_url=str(row.get("main_url")) if pd.notna(row.get("main_url")) else None,
                authors=_clean_text(row.get("Authors")) if pd.notna(row.get("Authors")) else None
            ))
        return results
    except Exception as e:
        print(f"Error in own_publications: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/expert-search", response_model=ExpertSearchResponse)
async def expert_search(request: ExpertSearchRequest):
    _require_loaded()
    query = " ".join(request.query.split())
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    school = _normalize_filter_value(request.school)
    department = _normalize_filter_value(request.department)
    has_filters = school is not None or department is not None
    qvec = _get_query_embedding(query)
    top_r = max(1, min(request.top_k_researchers, 50))
    top_p = max(1, min(request.top_k_publications, 100))
    researcher_pool = 500 if has_filters else top_r
    publication_pool = 1200 if has_filters else top_p

    ranked_researchers = _rank_researchers_for_query(qvec, researcher_pool)
    ranked_publications = _rank_publications_for_query(qvec, publication_pool)

    if has_filters:
        ranked_researchers = [
            r for r in ranked_researchers
            if _researcher_matches_filters(int(r.get("n_id", -1)), school, department)
        ]
        ranked_publications = [
            p for p in ranked_publications
            if _researcher_matches_filters(int(p.get("n_id", -1)), school, department)
        ]

    top_researchers = ranked_researchers[:top_r]
    top_publications = ranked_publications[:top_p]

    return ExpertSearchResponse(
        query=query,
        top_researchers=[ExpertResearcherResult(**r) for r in top_researchers],
        top_publications=[ExpertPublicationResult(**p) for p in top_publications]
    )


@app.get("/api/expert-search/themes/options", response_model=List[ExpertThemeOption])
async def expert_search_theme_options():
    _require_loaded()
    return [ExpertThemeOption(**o) for o in expert_theme_options_cache]


@app.get("/api/expert-search/filters", response_model=ExpertSearchFiltersResponse)
async def expert_search_filters():
    _require_loaded()
    return ExpertSearchFiltersResponse(**expert_search_filters_cache)


@app.post("/api/expert-search/themes", response_model=ExpertSearchResponse)
async def expert_search_by_theme(request: ExpertThemeSearchRequest):
    _require_loaded()
    option_id = " ".join((request.option_id or "").split())
    if not option_id:
        raise HTTPException(status_code=400, detail="Theme option cannot be empty.")

    parts = option_id.split(":", 2)
    if len(parts) < 2:
        raise HTTPException(status_code=400, detail="Invalid theme option.")

    top_r = max(1, min(request.top_k_researchers, 50))
    top_p = max(1, min(request.top_k_publications, 100))
    school = _normalize_filter_value(request.school)
    department = _normalize_filter_value(request.department)

    if parts[0] == "global":
        topic_id = _safe_int(parts[1])
        if topic_id is None:
            raise HTTPException(status_code=400, detail="Invalid TRISS topic id.")
        experts_all = report_topic_experts_cache.get(topic_id, [])
        experts_filtered = [
            e for e in experts_all
            if _safe_int(e.get("n_id")) is not None and _researcher_matches_filters(int(e["n_id"]), school, department)
        ]
        experts = experts_filtered[:top_r]
        researcher_scores = {int(e["n_id"]): float(e.get("similarity", 0.0)) for e in experts_filtered if _safe_int(e.get("n_id")) is not None}
        if pub_umap_coords_df is None or publications_df is None:
            publications = []
        else:
            topic_lookup = {
                str(row.get("article_id", "")).strip(): _safe_int(row.get("predicted_cluster")) or -1
                for _, row in pub_umap_coords_df.iterrows()
            }
            publications = _rank_theme_publications(publications_df, topic_lookup, int(topic_id), researcher_scores, top_p)
        topic_name = next((o.get("label", f"TRISS Topic {topic_id}") for o in expert_theme_options_cache if o.get("option_id") == option_id), f"TRISS Topic {topic_id}")
        return ExpertSearchResponse(
            query=topic_name,
            top_researchers=[ExpertResearcherResult(**r) for r in experts],
            top_publications=[ExpertPublicationResult(**p) for p in publications],
        )

    if parts[0] == "school":
        if len(parts) != 3:
            raise HTTPException(status_code=400, detail="Invalid school theme option.")
        topic_school = parts[1]
        topic_id = _safe_int(parts[2])
        if topic_id is None:
            raise HTTPException(status_code=400, detail="Invalid school topic id.")
        topic_name = next((o.get("label", f"{topic_school} Topic {topic_id}") for o in expert_theme_options_cache if o.get("option_id") == option_id), f"{topic_school} Topic {topic_id}")
        if school and school != topic_school:
            return ExpertSearchResponse(query=topic_name, top_researchers=[], top_publications=[])
        experts_all = report_school_topic_experts_cache.get(topic_school, {}).get(topic_id, [])
        experts_filtered = [
            e for e in experts_all
            if _safe_int(e.get("n_id")) is not None and _researcher_matches_filters(int(e["n_id"]), school, department)
        ]
        experts = experts_filtered[:top_r]
        researcher_scores = {int(e["n_id"]): float(e.get("similarity", 0.0)) for e in experts_filtered if _safe_int(e.get("n_id")) is not None}

        school_slug = _school_key(topic_school)
        school_pub_path = ANALYSIS_SCHOOLS_DIR / school_slug / "school_publication_umap_coordinates.csv"
        publications: List[Dict[str, Any]] = []
        if publications_df is not None and school_pub_path.exists():
            school_pub_df = pd.read_csv(school_pub_path)
            topic_lookup = {
                str(row.get("article_id", "")).strip(): _safe_int(row.get("predicted_cluster")) or -1
                for _, row in school_pub_df.iterrows()
            }
            pubs_school = publications_df[publications_df["school"].fillna("").astype(str) == topic_school]
            if department:
                pubs_school = pubs_school[pubs_school["department"].fillna("").astype(str) == department]
            publications = _rank_theme_publications(pubs_school, topic_lookup, int(topic_id), researcher_scores, top_p)

        return ExpertSearchResponse(
            query=topic_name,
            top_researchers=[ExpertResearcherResult(**r) for r in experts],
            top_publications=[ExpertPublicationResult(**p) for p in publications],
        )

    if parts[0] == "policy":
        topic_id = _safe_int(parts[1]) if len(parts) > 1 else None
        if topic_id is None:
            raise HTTPException(status_code=400, detail="Invalid policy domain id.")
        experts_all = report_macro_theme_experts_cache.get(topic_id, [])
        experts_filtered = [
            e for e in experts_all
            if _safe_int(e.get("n_id")) is not None and _researcher_matches_filters(int(e["n_id"]), school, department)
        ]
        experts = experts_filtered[:top_r]
        researcher_scores = {
            int(e["n_id"]): float(e.get("similarity", 0.0))
            for e in experts_filtered
            if _safe_int(e.get("n_id")) is not None
        }
        publications: List[Dict[str, Any]] = []
        if publications_df is not None:
            pubs = publications_df
            if school:
                pubs = pubs[pubs["school"].fillna("").astype(str) == school]
            if department:
                pubs = pubs[pubs["department"].fillna("").astype(str) == department]
            publications = _rank_publications_by_researcher_scores(pubs, researcher_scores, top_p)
        topic_name = next(
            (o.get("label", f"Policy Domain {topic_id}") for o in expert_theme_options_cache if o.get("option_id") == option_id),
            f"Policy Domain {topic_id}",
        )
        return ExpertSearchResponse(
            query=topic_name,
            top_researchers=[ExpertResearcherResult(**r) for r in experts],
            top_publications=[ExpertPublicationResult(**p) for p in publications],
        )

    raise HTTPException(status_code=400, detail="Unsupported theme scope.")


@app.get("/api/expert-search/researcher/{n_id}/publications", response_model=List[ExpertPublicationResult])
async def expert_search_researcher_publications(
    n_id: int,
    query: str = Query(..., min_length=2),
    limit: int = Query(15, ge=1, le=100)
):
    _require_loaded()
    profile_match = profiles_df[profiles_df["n_id"] == n_id]
    if profile_match.empty:
        raise HTTPException(status_code=404, detail="Researcher not found.")
    qvec = _get_query_embedding(query)
    ranked = _rank_publications_for_researcher_query(n_id, qvec, limit)
    return [ExpertPublicationResult(**p) for p in ranked]

# ---------------------------------------------------------------------
# ROUTES: MAP
# ---------------------------------------------------------------------
@app.get("/api/points")
@app.get("/api/map/points")
async def get_map_points():
    _require_loaded()
    return map_points_cache


@app.get("/api/topics")
@app.get("/api/map/topics")
async def get_map_topics():
    _require_loaded()
    return map_topics_cache


@app.get("/api/filters")
@app.get("/api/map/filters")
async def get_map_filters():
    _require_loaded()
    return map_filters_cache


@app.get("/api/map-v2/points")
async def get_map_v2_points():
    _require_loaded()
    return map_v2_points_cache


@app.get("/api/map-v2/topics")
async def get_map_v2_topics():
    _require_loaded()
    return map_v2_topics_cache


@app.get("/api/map-v2/filters")
async def get_map_v2_filters():
    _require_loaded()
    return map_v2_filters_cache


@app.get("/api/map/researchers")
async def get_map_researchers():
    _require_loaded()
    return map_researcher_points_cache


@app.get("/api/map-v2/researchers")
async def get_map_v2_researchers():
    _require_loaded()
    return map_v2_researcher_points_cache


@app.get("/api/map/publications")
async def get_map_pubs():
    _require_loaded()
    if pub_umap_coords_df is None:
        return []
    # Ensure keys are x, y
    df = pub_umap_coords_df.rename(columns={"umap_x": "x", "umap_y": "y"})
    return df.dropna(subset=["x", "y"]).to_dict(orient="records")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
