#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
TRISS V3 Pipeline — Stage 3: Policy Domain Layer
=================================================
Adds a TRISS-defined 5-domain policy classification layer on top of the
Stage-2 macro themes.  Completely additive — does NOT modify any Stage-1
or Stage-2 file.

Inputs
------
  stage3/Policy_domain_definitions.json   — 5 policy domains (pre-defined)
  stage2/cluster_centroids_stage1.npy     — L2-normalised (25, 3072) Stage-1
  stage2/macro_cluster_assignments.csv    — stage1 → macro mapping
  stage2/macro_themes_overview.json       — macro theme metadata
  stage2/researcher_macro_theme_weights.csv

Outputs (all in stage3/)
------------------------
  policy_domain_embeddings.npy            (5, 3072)
  policy_domain_overview.json             (copy with embeddings omitted)
  macro_theme_policy_weights.csv
  researcher_policy_weights.csv
  policy_domain_summary_stats.json
"""

import json
import shutil
import numpy as np
import pandas as pd
from pathlib import Path
from openai import OpenAI
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
STAGE2_DIR = GLOBAL_DIR / "stage2"
STAGE3_DIR = GLOBAL_DIR / "stage3"
STAGE3_DIR.mkdir(parents=True, exist_ok=True)

EMBED_MODEL = "text-embedding-3-large"
EMBED_DIM   = 3072
N_DOMAINS   = 5

client = OpenAI()


# --------------------------------------------------
# HELPER
# --------------------------------------------------
def get_embedding(text: str) -> np.ndarray:
    """Embed text and return L2-normalised vector."""
    response = client.embeddings.create(
        model=EMBED_MODEL,
        input=text,
        dimensions=EMBED_DIM,
    )
    vec = np.array(response.data[0].embedding, dtype=np.float32)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


def make_domain_text(domain: dict) -> str:
    """Concatenate domain fields into a single embedding string."""
    parts = [
        domain.get("title", ""),
        domain.get("description", ""),
        ", ".join(domain.get("institutional_systems", [])),
        ", ".join(domain.get("policy_levers", [])),
    ]
    return " ".join(p for p in parts if p)


# --------------------------------------------------
# STEP 1 — Load inputs
# --------------------------------------------------
def step1_load():
    print("\n══ STEP 1: Load Stage-2 outputs ══")
    policy_defs_path = STAGE3_DIR / "Policy_domain_definitions.json"
    policy_domains = json.loads(policy_defs_path.read_text())
    print(f"  Policy domains: {len(policy_domains)}")

    centroids = np.load(STAGE2_DIR / "cluster_centroids_stage1.npy").astype(np.float32)
    macro_map = pd.read_csv(STAGE2_DIR / "macro_cluster_assignments.csv")
    macro_overview = json.loads((STAGE2_DIR / "macro_themes_overview.json").read_text())
    researcher_weights = pd.read_csv(STAGE2_DIR / "researcher_macro_theme_weights.csv")
    researcher_weights["n_id"] = researcher_weights["n_id"].astype(str)

    # Load researcher school info from global assignments
    assignments = pd.read_csv(GLOBAL_DIR / "global_cluster_assignments.csv")
    assignments["n_id"] = assignments["n_id"].astype(str)

    # stage1 cluster → macro ID
    stage1_to_macro = {int(r["cluster_id_stage1"]): int(r["macro_cluster_id"])
                       for _, r in macro_map.iterrows()}
    best_k = max(stage1_to_macro.values()) + 1
    print(f"  Stage-1 clusters: {centroids.shape[0]} | Macro themes: {best_k}")
    print(f"  Researchers: {len(researcher_weights)}")
    return policy_domains, centroids, stage1_to_macro, best_k, macro_overview, researcher_weights, assignments


# --------------------------------------------------
# STEP 2 — Embed policy domains
# --------------------------------------------------
def step2_embed_domains(policy_domains):
    print("\n══ STEP 2: Generate policy domain embeddings ══")
    embed_path = STAGE3_DIR / "policy_domain_embeddings.npy"

    embeddings = []
    for domain in policy_domains:
        did  = domain["policy_domain_id"]
        text = make_domain_text(domain)
        print(f"  Embedding domain {did}: {domain['title'][:50]}...")
        vec = get_embedding(text)
        embeddings.append(vec)
    embed_matrix = np.vstack(embeddings)   # (5, 3072)
    np.save(embed_path, embed_matrix)
    print(f"  Saved: policy_domain_embeddings.npy  shape={embed_matrix.shape}")

    # Save clean overview (no extra data)
    overview_out = [
        {
            "policy_domain_id": d["policy_domain_id"],
            "title": d["title"],
            "description": d["description"],
            "institutional_systems": d.get("institutional_systems", []),
            "policy_levers": d.get("policy_levers", []),
        }
        for d in policy_domains
    ]
    (STAGE3_DIR / "policy_domain_overview.json").write_text(
        json.dumps(overview_out, indent=2))
    print("  Saved: policy_domain_overview.json")
    return embed_matrix


# --------------------------------------------------
# STEP 3 — Link macro themes to policy domains
# --------------------------------------------------
def step3_macro_to_domains(centroids, stage1_to_macro, best_k, domain_embeddings):
    print("\n══ STEP 3: Link macro themes → policy domains ══")

    # Compute macro-theme centroid = mean of stage1 centroids in group
    macro_vecs = {}
    for mid in range(best_k):
        ids = [cid for cid, m in stage1_to_macro.items() if m == mid]
        if ids:
            macro_vecs[mid] = centroids[ids].mean(axis=0)
            norm = np.linalg.norm(macro_vecs[mid])
            if norm > 0:
                macro_vecs[mid] /= norm

    rows = []
    print("  Macro theme → policy domain mapping:")
    for mid in range(best_k):
        if mid not in macro_vecs:
            continue
        mv = macro_vecs[mid].reshape(1, -1)
        sims = cosine_similarity(mv, domain_embeddings)[0]   # (5,)
        sims_pos = np.clip(sims, 0, None)
        total = sims_pos.sum()
        weights = sims_pos / total if total > 0 else np.ones(N_DOMAINS) / N_DOMAINS
        hard = int(np.argmax(sims))   # use raw (can be negative) for argmax
        print(f"    Macro {mid} → Domain {hard}  (sims: {[round(float(s),3) for s in sims]})")
        rows.append({
            "macro_theme_id": mid,
            "hard_policy_domain": hard,
            **{f"policy_domain_{d}": round(float(weights[d]), 6) for d in range(N_DOMAINS)}
        })

    df = pd.DataFrame(rows)
    df.to_csv(STAGE3_DIR / "macro_theme_policy_weights.csv", index=False)
    print("  Saved: macro_theme_policy_weights.csv")
    return df


# --------------------------------------------------
# STEP 4 — Propagate to researchers
# --------------------------------------------------
def step4_researcher_domains(researcher_weights, macro_policy_df, best_k):
    print("\n══ STEP 4: Propagate policy weights to researchers ══")

    macro_weight_cols  = [f"macro_theme_{m}" for m in range(best_k)]
    domain_weight_cols = [f"policy_domain_{d}" for d in range(N_DOMAINS)]

    # Build macro → hard domain lookup and soft weight matrix (best_k, N_DOMAINS)
    macro_to_hard = {int(r["macro_theme_id"]): int(r["hard_policy_domain"])
                     for _, r in macro_policy_df.iterrows()}
    M2D = np.zeros((best_k, N_DOMAINS), dtype=np.float32)
    for _, row in macro_policy_df.iterrows():
        mid = int(row["macro_theme_id"])
        for d in range(N_DOMAINS):
            M2D[mid, d] = float(row[f"policy_domain_{d}"])

    rows = []
    for _, res in researcher_weights.iterrows():
        nid        = str(res["n_id"])
        hard_macro = int(res.get("hard_macro_theme", res.get("primary_macro_theme", 0)))
        # Hard domain: inherit from researcher's hard macro → that macro's hard domain
        hard       = macro_to_hard.get(hard_macro, 0)
        # Soft weights: propagate researcher macro soft weights through M2D
        rm         = np.array([float(res.get(c, 0.0)) for c in macro_weight_cols], dtype=np.float32)
        policy_vec = rm @ M2D
        total      = policy_vec.sum()
        if total > 0:
            policy_vec /= total
        primary    = int(np.argmax(policy_vec))
        rows.append({
            "n_id": nid,
            "hard_policy_domain": hard,
            "primary_policy_domain": primary,
            **{f"policy_domain_{d}": round(float(policy_vec[d]), 6) for d in range(N_DOMAINS)}
        })

    df = pd.DataFrame(rows)
    df.to_csv(STAGE3_DIR / "researcher_policy_weights.csv", index=False)
    print(f"  Saved: researcher_policy_weights.csv  ({len(df)} researchers)")
    return df


# --------------------------------------------------
# STEP 5 — Summary statistics
# --------------------------------------------------
def step5_stats(researcher_policy, researcher_weights, assignments, policy_domains):
    print("\n══ STEP 5: Summary statistics ══")

    domain_weight_cols = [f"policy_domain_{d}" for d in range(N_DOMAINS)]

    # Merge school info
    df = researcher_policy.merge(
        assignments[["n_id", "school"]].drop_duplicates("n_id"),
        on="n_id", how="left"
    )

    # Counts per hard domain
    counts = df["hard_policy_domain"].value_counts().to_dict()

    # School distribution per domain
    school_dist = {}
    for d in range(N_DOMAINS):
        sub = df[df["hard_policy_domain"] == d]
        school_dist[d] = sub["school"].value_counts().to_dict()

    # Overlap matrix (correlation of weight vectors)
    W = researcher_policy[domain_weight_cols].values
    corr = np.corrcoef(W.T) if len(researcher_policy) > 1 else np.eye(N_DOMAINS)
    overlap = {
        f"{i}_{j}": round(float(corr[i, j]), 4)
        for i in range(N_DOMAINS) for j in range(i + 1, N_DOMAINS)
    }

    # % multi-domain (primary weight < 0.6 = spread across domains)
    max_weights = researcher_policy[domain_weight_cols].max(axis=1)
    pct_multi = float((max_weights < 0.6).mean()) * 100

    stats = {
        "n_policy_domains": N_DOMAINS,
        "researchers_per_domain": {str(k): int(v) for k, v in counts.items()},
        "school_distribution_per_domain": {str(k): v for k, v in school_dist.items()},
        "cross_domain_overlap_matrix": overlap,
        "pct_multi_domain_researchers": round(pct_multi, 1),
        "domain_names": {str(d["policy_domain_id"]): d["title"] for d in policy_domains},
    }
    (STAGE3_DIR / "policy_domain_summary_stats.json").write_text(json.dumps(stats, indent=2))
    print("  Saved: policy_domain_summary_stats.json")

    print("\n─── POLICY DOMAIN DISTRIBUTION ───")
    for d in range(N_DOMAINS):
        n = int(counts.get(d, 0))
        name = policy_domains[d]["title"]
        print(f"  Domain {d} ({n:3d} researchers): {name}")
    print(f"\n  Multi-domain researchers (max weight <0.6): {pct_multi:.1f}%")
    return stats


# --------------------------------------------------
# STEP 6 — Print backend sketch reminder
# --------------------------------------------------
def step6_backend_sketch():
    print("\n══ STEP 6: Backend integration sketch ══")
    sketch = """
# Add to main.py startup:
STAGE3_DIR = ANALYSIS_GLOBAL_DIR / "stage3"
policy_domains_cache: list = []
researcher_policy_cache: pd.DataFrame | None = None

# Load:
if (STAGE3_DIR / "policy_domain_overview.json").exists():
    policy_domains_cache = json.load(open(STAGE3_DIR / "policy_domain_overview.json"))
if (STAGE3_DIR / "researcher_policy_weights.csv").exists():
    researcher_policy_cache = pd.read_csv(STAGE3_DIR / "researcher_policy_weights.csv")

# Endpoints to add:
# GET /api/policy-domains              → policy_domains_cache
# GET /api/policy-domains/{id}         → policy_domains_cache[id]
# GET /api/policy-domains/{id}/researchers
#   → researcher_policy_cache[researcher_policy_cache.hard_policy_domain == id]
"""
    print(sketch)
    notes_path = STAGE3_DIR / "STAGE3_BACKEND_NOTES.md"
    notes_path.write_text(f"""# Stage-3 Policy Domains — Backend Integration Notes

## Output files (all in `global/stage3/`)

| File | Contents |
|------|---------|
| `policy_domain_overview.json` | Array of 5 policy domain objects (title, description, institutional_systems, policy_levers) |
| `policy_domain_embeddings.npy` | (5, 3072) L2-normalised embedding matrix |
| `macro_theme_policy_weights.csv` | Each macro theme → hard domain + soft weights policy_domain_0…4 |
| `researcher_policy_weights.csv` | Each researcher n_id → hard_policy_domain + soft weights |
| `policy_domain_summary_stats.json` | Counts, school distribution, overlap matrix |

## Suggested API endpoints

```python
STAGE3_DIR = ANALYSIS_GLOBAL_DIR / "stage3"

# GET /api/policy-domains
@app.get("/api/policy-domains")
async def get_policy_domains():
    return policy_domains_cache   # list of 5 dicts

# GET /api/policy-domains/{{id}}
@app.get("/api/policy-domains/{{domain_id}}")
async def get_policy_domain(domain_id: int):
    return next((d for d in policy_domains_cache if d["policy_domain_id"] == domain_id), None)

# GET /api/policy-domains/{{id}}/researchers
@app.get("/api/policy-domains/{{domain_id}}/researchers")
async def get_policy_domain_researchers(domain_id: int):
    if researcher_policy_cache is None:
        return []
    score_col = f"policy_domain_{{domain_id}}"
    subset = researcher_policy_cache[researcher_policy_cache["hard_policy_domain"] == domain_id].copy()
    if score_col in subset.columns:
        subset = subset.sort_values(by=score_col, ascending=False)
    return subset.head(50).to_dict(orient="records")
```

## Key fields

- `hard_policy_domain` — primary domain (argmax of propagated weights)
- `policy_domain_0` … `policy_domain_4` — soft weights summing to 1.0
- Use soft weights for visualisation (e.g. proportional coloring on the map)
""")
    print(f"  Saved: STAGE3_BACKEND_NOTES.md")


# --------------------------------------------------
# MAIN
# --------------------------------------------------
def main():
    print(f"Stage-3 output directory: {STAGE3_DIR}")

    policy_domains, centroids, stage1_to_macro, best_k, macro_overview, researcher_weights, assignments = step1_load()
    domain_embeddings = step2_embed_domains(policy_domains)
    macro_policy_df   = step3_macro_to_domains(centroids, stage1_to_macro, best_k, domain_embeddings)
    researcher_policy = step4_researcher_domains(researcher_weights, macro_policy_df, best_k)
    stats             = step5_stats(researcher_policy, researcher_weights, assignments, policy_domains)
    step6_backend_sketch()

    print("\n✅ Stage-3 policy domain pipeline complete.")
    print(f"All outputs in: {STAGE3_DIR}")


if __name__ == "__main__":
    main()
