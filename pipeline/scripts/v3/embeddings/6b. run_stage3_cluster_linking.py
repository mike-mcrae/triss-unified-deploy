#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
TRISS V3 Pipeline — Stage 3 Extension: Policy Domain Cluster Linking
=====================================================================
Extends stage-3 policy domain linkage to ALL cluster levels.
Additive only — does NOT modify any Stage-1 or Stage-2 files.

Inputs (from stage3/):
  policy_domain_embeddings.npy          (5, 3072) — already generated
  researcher_policy_weights.csv         (316 rows) — already generated

Inputs (from global/):
  global_cluster_centroids.npy          (25, 3072)

Inputs (from schools/*/):
  school_cluster_centroids.npy          (10, 3072) per school

Outputs (all in stage3/):
  global_cluster_policy_weights.csv
  school_cluster_policy_weights.csv
  researcher_policy_confidence.csv
  policy_domains_metadata.json
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
import os

# Path configuration (env-overridable, no machine-specific absolutes)
_PIPELINE_ROOT = Path(
    os.environ.get(
        "TRISS_SOURCE_PIPELINE_DIR",
        str(next((p for p in Path(__file__).resolve().parents if p.name == "pipeline"), Path(__file__).resolve().parents[2]))
    )
).expanduser().resolve()
_PROJECT_ROOT = Path(
    os.environ.get("TRISS_SOURCE_PROJECT_DIR", str(_PIPELINE_ROOT.parent))
).expanduser().resolve()
_SHARED_DATA_ROOT = Path(
    os.environ.get("TRISS_SOURCE_SHARED_DATA_DIR", str(_PROJECT_ROOT / "1. data"))
).expanduser().resolve()
_REPORT_ROOT = Path(
    os.environ.get("TRISS_SOURCE_REPORT_DIR", str(_PROJECT_ROOT / "triss-report-app-v3"))
).expanduser().resolve()

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
BASE       = _PIPELINE_ROOT
GLOBAL_DIR = BASE / "1. data/5. analysis/global"
SCHOOLS_DIR= BASE / "1. data/5. analysis/schools"
STAGE3_DIR = GLOBAL_DIR / "stage3"
N_DOMAINS  = 5

SCHOOLS = [
    "business",
    "education",
    "law",
    "linguistic__speech_and_communication_sciences",
    "psychology",
    "religion__theology_and_peace_studies",
    "social_work_and_social_policy",
    "ssp",
]

# Short labels: 3 words max, for presentation
SHORT_LABELS = {
    0: "Democratic Governance",
    1: "Economic Development",
    2: "Health & Wellbeing",
    3: "Social Protection",
    4: "Inclusive Education",
}


# --------------------------------------------------
# HELPERS
# --------------------------------------------------
def l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


def policy_weights_for_centroids(centroids: np.ndarray, domain_emb: np.ndarray) -> pd.DataFrame:
    """
    For each centroid row, compute:
      - cosine similarity to each domain embedding
      - clip negatives to 0
      - normalise to sum to 1
      - hard_policy_domain = argmax of raw similarities
    Returns a DataFrame with columns: hard_policy_domain, policy_domain_0..4
    """
    centroids = l2_normalize(centroids.astype(np.float32))
    sims = cosine_similarity(centroids, domain_emb)       # (N, 5)
    hard = np.argmax(sims, axis=1).astype(int)
    sims_pos = np.clip(sims, 0, None)
    totals = sims_pos.sum(axis=1, keepdims=True)
    totals[totals == 0] = 1.0
    weights = sims_pos / totals                          # (N, 5) normalised

    rows = []
    for i in range(len(centroids)):
        row = {"hard_policy_domain": int(hard[i])}
        for d in range(N_DOMAINS):
            row[f"policy_domain_{d}"] = round(float(weights[i, d]), 6)
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------
# STEP 1 — Confirm policy domain embeddings exist
# --------------------------------------------------
def step1_confirm():
    print("\n══ STEP 1: Confirm policy domain embeddings ══")
    emb_path = STAGE3_DIR / "policy_domain_embeddings.npy"
    if not emb_path.exists():
        raise FileNotFoundError(
            f"Missing: {emb_path}\nRun '6. run_stage3_policy_domains.py' first.")
    domain_emb = np.load(emb_path).astype(np.float32)
    domain_emb = l2_normalize(domain_emb)
    print(f"  ✓ policy_domain_embeddings.npy  shape={domain_emb.shape}")
    return domain_emb


# --------------------------------------------------
# STEP 2 — Global TRISS clusters → policy domains
# --------------------------------------------------
def step2_global(domain_emb):
    print("\n══ STEP 2: Global TRISS clusters → policy domain mapping ══")
    centroids_path = GLOBAL_DIR / "global_cluster_centroids.npy"
    centroids = np.load(centroids_path).astype(np.float32)
    print(f"  Global centroids: {centroids.shape}")

    df = policy_weights_for_centroids(centroids, domain_emb)
    df.insert(0, "cluster_id", range(len(df)))

    out_path = STAGE3_DIR / "global_cluster_policy_weights.csv"
    df.to_csv(out_path, index=False)
    print(f"  Saved: global_cluster_policy_weights.csv  ({len(df)} clusters)")

    distribution = df["hard_policy_domain"].value_counts().sort_index()
    for did, cnt in distribution.items():
        print(f"    Domain {did}: {cnt} global clusters")
    return df


# --------------------------------------------------
# STEP 3 — School clusters → policy domains
# --------------------------------------------------
def step3_schools(domain_emb):
    print("\n══ STEP 3: School clusters → policy domain mapping ══")
    all_rows = []

    for school in SCHOOLS:
        school_dir = SCHOOLS_DIR / school
        centroids_path = school_dir / "school_cluster_centroids.npy"
        if not centroids_path.exists():
            print(f"  ⚠  Skipping {school} — no centroids file")
            continue
        centroids = np.load(centroids_path).astype(np.float32)
        df_school = policy_weights_for_centroids(centroids, domain_emb)
        df_school.insert(0, "cluster_id", range(len(df_school)))
        df_school.insert(0, "school_name", school)
        all_rows.append(df_school)
        distribution = df_school["hard_policy_domain"].value_counts().sort_index().to_dict()
        print(f"  {school}: {len(df_school)} clusters → domains {distribution}")

    df_all = pd.concat(all_rows, ignore_index=True)
    out_path = STAGE3_DIR / "school_cluster_policy_weights.csv"
    df_all.to_csv(out_path, index=False)
    print(f"  Saved: school_cluster_policy_weights.csv  ({len(df_all)} total school clusters)")
    return df_all


# --------------------------------------------------
# STEP 4 — Researcher policy confidence metrics
# --------------------------------------------------
def step4_confidence():
    print("\n══ STEP 4: Researcher policy confidence metrics ══")
    weights_path = STAGE3_DIR / "researcher_policy_weights.csv"
    df = pd.read_csv(weights_path)

    domain_cols = [f"policy_domain_{d}" for d in range(N_DOMAINS)]
    W = df[domain_cols].values

    sorted_w = np.sort(W, axis=1)[:, ::-1]   # descending per row
    max_w    = sorted_w[:, 0]
    sec_w    = sorted_w[:, 1]
    gap      = max_w - sec_w

    conf_df = pd.DataFrame({
        "n_id":                 df["n_id"].astype(str),
        "hard_policy_domain":   df["hard_policy_domain"].astype(int),
        "max_policy_weight":    np.round(max_w, 6),
        "second_policy_weight": np.round(sec_w, 6),
        "weight_gap":           np.round(gap,   6),
    })

    out_path = STAGE3_DIR / "researcher_policy_confidence.csv"
    conf_df.to_csv(out_path, index=False)
    print(f"  Saved: researcher_policy_confidence.csv  ({len(conf_df)} researchers)")
    print(f"  Mean max weight : {max_w.mean():.3f}")
    print(f"  Mean weight gap : {gap.mean():.3f}")
    print(f"  Highly confident (gap > 0.1): {(gap > 0.1).sum()} researchers")
    return conf_df


# --------------------------------------------------
# STEP 5 — Policy domain metadata file (presentation)
# --------------------------------------------------
def step5_metadata():
    print("\n══ STEP 5: Policy domain metadata (presentation) ══")
    src = json.loads((STAGE3_DIR / "policy_domain_overview.json").read_text())

    metadata = []
    for item in src:
        did = int(item["policy_domain_id"])
        metadata.append({
            "policy_domain_id": did,
            "title":            item["title"],
            "short_label":      SHORT_LABELS.get(did, item["title"]),
            "description":      item["description"],
            "order_index":      did,
        })

    out_path = STAGE3_DIR / "policy_domains_metadata.json"
    out_path.write_text(json.dumps(metadata, indent=2))
    print(f"  Saved: policy_domains_metadata.json  ({len(metadata)} domains)")
    for m in metadata:
        print(f"    {m['policy_domain_id']}: {m['short_label']}")
    return metadata


# --------------------------------------------------
# FINAL CHECK
# --------------------------------------------------
def final_check():
    print("\n══ FINAL CHECK ══")
    required = [
        "policy_domain_embeddings.npy",
        "policy_domain_overview.json",
        "policy_domains_metadata.json",
        "macro_theme_policy_weights.csv",
        "global_cluster_policy_weights.csv",
        "school_cluster_policy_weights.csv",
        "researcher_policy_weights.csv",
        "researcher_policy_confidence.csv",
        "policy_domain_summary_stats.json",
    ]
    all_ok = True
    for fname in required:
        path = STAGE3_DIR / fname
        exists = path.exists()
        size   = f"{path.stat().st_size:,} bytes" if exists else "MISSING"
        status = "✓" if exists else "✗"
        print(f"  {status} {fname:50s} {size}")
        if not exists:
            all_ok = False

    if all_ok:
        print("\n  ✅ All Stage-3 policy domain files verified.")
    else:
        print("\n  ⚠  Some files are missing — check errors above.")


# --------------------------------------------------
# MAIN
# --------------------------------------------------
def main():
    print(f"Stage-3 directory: {STAGE3_DIR}")
    domain_emb = step1_confirm()
    step2_global(domain_emb)
    step3_schools(domain_emb)
    step4_confidence()
    step5_metadata()
    final_check()
    print("\n✅ Stage-3 policy domain cluster linking complete.")


if __name__ == "__main__":
    main()
