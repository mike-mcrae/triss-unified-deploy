#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
TRISS V3 Pipeline: Semantic Cluster Visualisation (Publications level)
=======================================================================
Generates dark-theme UMAP scatter plots — one per publication,
coloured by the researcher-level cluster centroid assignment.

Methodology:
  1. Load researcher cluster assignments (from step 3, KMeans on researcher embeddings)
  2. Load individual publication embeddings (OpenAI by_publication/)
  3. Assign each publication to the nearest researcher centroid (cosine similarity)
  4. Run UMAP on the publication embeddings (n_neighbors=30, cosine metric)
  5. Trim 1–99 percentile outliers (visual only — coordinates CSV is unclipped)
  6. Render dark-theme scatter: small alpha dots, coloured-pill topic labels

Outputs per directory:
  {prefix}_publication_umap_coordinates.csv  — full coordinates (all points)
  {prefix}_umap_scatter.png                  — researcher-level UMAP (dark theme)
  {prefix}_publication_umap_scatter.png      — publication-level UMAP (dark theme)

Usage:
  python3 "4. visualise_clusters.py"            # global + all schools
  python3 "4. visualise_clusters.py" --global-only
"""

import json
import warnings
import argparse
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import umap.umap_ as umap
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

warnings.filterwarnings("ignore", message="n_jobs value")
matplotlib.use("Agg")  # headless rendering

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
BASE        = _PIPELINE_ROOT
OUT_BASE    = BASE / "1. data/5. analysis"
EMBED_BASE  = BASE / "1. data/4. embeddings/v3/openai/by_publication"
RESEARCHER_EMBED_BASE = BASE / "1. data/4. embeddings/v3/openai/by_researcher"

FIGSIZE            = (18, 14)
PUB_POINT_SIZE     = 4
RES_POINT_SIZE     = 60
POINT_ALPHA        = 0.55
BG_COLOR           = "#0f1117"
LABEL_MAX_CHARS    = 26
UMAP_N_NEIGHBORS   = 30
UMAP_MIN_DIST      = 0.1
RANDOM_STATE       = 42

LABEL_COLLISION_RADIUS = 0.05
MAX_LABEL_NUDGE_ITERS  = 200

# --------------------------------------------------
# HELPERS
# --------------------------------------------------

def get_cmap(n):
    return matplotlib.colormaps["tab20"]


def nudge_labels(label_positions, radius, max_iter=MAX_LABEL_NUDGE_ITERS):
    topics = sorted(label_positions.keys())
    pos = {t: np.array(label_positions[t], dtype=np.float64) for t in topics}
    for _ in range(max_iter):
        moved = False
        for i, t in enumerate(topics):
            for j in range(i + 1, len(topics)):
                u = topics[j]
                d = pos[t] - pos[u]
                dist = float(np.linalg.norm(d))
                if dist == 0.0:
                    pos[t] += np.array([radius, 0.0])
                    moved = True
                elif dist < radius:
                    push = (radius - dist) * (d / dist) * 0.5
                    pos[t] += push
                    pos[u] -= push
                    moved = True
        if not moved:
            break
    return {t: (float(pos[t][0]), float(pos[t][1])) for t in topics}


def dark_scatter(ax, fig, x, y, colors, sizes, alpha, title_str):
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.scatter(x, y, c=colors, s=sizes, alpha=alpha, linewidths=0)
    ax.set_title(title_str, color="white", fontsize=13, pad=14)
    ax.tick_params(colors="#555")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")
    ax.set_xlabel("UMAP-1", color="#888", fontsize=9)
    ax.set_ylabel("UMAP-2", color="#888", fontsize=9)


def add_dark_labels(ax, label_positions, desc, cmap_fn, n_clusters):
    for t, (xc, yc) in label_positions.items():
        topic_info = desc.get(str(t), {})
        name = topic_info.get("topic_name", f"Topic {t}")
        short = name[:LABEL_MAX_CHARS] + "…" if len(name) > LABEL_MAX_CHARS else name
        color = cmap_fn(int(t) / max(n_clusters, 1))
        ax.annotate(
            f"{t}. {short}",
            (xc, yc),
            fontsize=6.5,
            color="white",
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.25", fc=color, alpha=0.60, lw=0)
        )


def clip_outliers(df, x_col="umap_x", y_col="umap_y", pct=(1, 99)):
    if len(df) <= 100:
        return df
    x_lo, x_hi = np.percentile(df[x_col], pct)
    y_lo, y_hi = np.percentile(df[y_col], pct)
    return df[(df[x_col] >= x_lo) & (df[x_col] <= x_hi) &
              (df[y_col] >= y_lo) & (df[y_col] <= y_hi)]


def compute_label_positions(df_vis, cluster_col):
    positions = {}
    for t in df_vis[cluster_col].dropna().unique():
        sub = df_vis[df_vis[cluster_col] == t]
        positions[int(t)] = (float(sub["umap_x"].mean()), float(sub["umap_y"].mean()))
    return positions


# --------------------------------------------------
# RESEARCHER-LEVEL SCATTER (from step 3 UMAP coords)
# --------------------------------------------------

def plot_researcher_umap(data_dir, prefix, desc):
    coord_file = data_dir / f"{prefix}_umap_coordinates.csv"
    if not coord_file.exists():
        print(f"  Skipping researcher scatter — no {coord_file.name}")
        return

    df = pd.read_csv(coord_file)
    if not {"umap_x", "umap_y", "cluster"}.issubset(df.columns):
        print(f"  Skipping researcher scatter — missing columns in {coord_file.name}")
        return

    n_clusters = int(df["cluster"].max()) + 1
    cmap_fn    = get_cmap(n_clusters)

    colors = [cmap_fn(int(c) / n_clusters) for c in df["cluster"]]

    fig, ax = plt.subplots(figsize=FIGSIZE)
    dark_scatter(ax, fig, df["umap_x"], df["umap_y"], colors, RES_POINT_SIZE, 0.85,
                 f"TRISS Researcher UMAP — {prefix} (K={n_clusters})")

    df_vis = clip_outliers(df)
    positions = compute_label_positions(df_vis, "cluster")
    x_range = df_vis["umap_x"].max() - df_vis["umap_x"].min()
    y_range = df_vis["umap_y"].max() - df_vis["umap_y"].min()
    radius  = max(x_range, y_range) * 0.05
    positions = nudge_labels(positions, radius=radius)
    add_dark_labels(ax, positions, desc, cmap_fn, n_clusters)

    plt.tight_layout()
    out_path = data_dir / f"{prefix}_umap_scatter.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Researcher scatter → {out_path.name}")


# --------------------------------------------------
# PUBLICATION-LEVEL SCATTER (UMAP on pub embeddings)
# --------------------------------------------------

def plot_publication_umap(data_dir, prefix, desc, df_all_pubs):
    assign_file   = data_dir / f"{prefix}_cluster_assignments.csv"
    centroid_file = data_dir / f"{prefix}_cluster_centroids.npy"

    if not assign_file.exists() or not centroid_file.exists():
        print(f"  Skipping publication scatter — missing inputs for {prefix}")
        return

    df_users  = pd.read_csv(assign_file)
    centroids = np.load(centroid_file)

    valid_nids = set(df_users["n_id"].astype(str))
    df_slice   = df_all_pubs[df_all_pubs["n_id"].astype(str).isin(valid_nids)].copy()

    print(f"  Loading {len(df_slice)} publication embeddings...")
    vecs, rows = [], []
    for _, row in df_slice.iterrows():
        fp = EMBED_BASE / f"article_{row['article_id']}.json"
        if not fp.exists():
            continue
        try:
            obj = json.loads(fp.read_text())
            vec = obj.get("embeddings", {}).get("abstract")
            if vec is None:
                continue
            v = np.array(vec, dtype=np.float32)
            norm = np.linalg.norm(v)
            if norm > 0:
                v /= norm
            vecs.append(v)
            rows.append(row)
        except Exception:
            continue

    if not vecs:
        print(f"  No publication embeddings found — skipping.")
        return

    X      = np.vstack(vecs)
    df_meta = pd.DataFrame(rows).reset_index(drop=True)

    # Assign each publication to nearest researcher centroid
    sims = cosine_similarity(X, centroids)
    df_meta["predicted_cluster"] = np.argmax(sims, axis=1)

    print(f"  Running UMAP on {len(X)} publications...")
    reducer = umap.UMAP(
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        n_components=2,
        metric="cosine",
        random_state=RANDOM_STATE
    )
    umap_2d = reducer.fit_transform(X)
    df_meta["umap_x"] = umap_2d[:, 0]
    df_meta["umap_y"] = umap_2d[:, 1]

    # Save full coordinates
    df_meta[["article_id", "predicted_cluster", "umap_x", "umap_y"]].to_csv(
        data_dir / f"{prefix}_publication_umap_coordinates.csv", index=False
    )

    n_clusters = int(df_meta["predicted_cluster"].max()) + 1
    cmap_fn    = get_cmap(n_clusters)
    colors     = [cmap_fn(int(c) / n_clusters) for c in df_meta["predicted_cluster"]]

    df_vis = clip_outliers(df_meta)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    dark_scatter(ax, fig, df_vis["umap_x"], df_vis["umap_y"],
                 [cmap_fn(int(c) / n_clusters) for c in df_vis["predicted_cluster"]],
                 PUB_POINT_SIZE, POINT_ALPHA,
                 f"TRISS Publication UMAP — {prefix} (K={n_clusters}, n={len(df_vis):,})")

    positions = compute_label_positions(df_vis, "predicted_cluster")
    x_range = df_vis["umap_x"].max() - df_vis["umap_x"].min()
    y_range = df_vis["umap_y"].max() - df_vis["umap_y"].min()
    radius  = max(x_range, y_range) * 0.05
    positions = nudge_labels(positions, radius=radius)
    add_dark_labels(ax, positions, desc, cmap_fn, n_clusters)

    plt.tight_layout()
    out_path = data_dir / f"{prefix}_publication_umap_scatter.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Publication scatter → {out_path.name}")


# --------------------------------------------------
# ORCHESTRATOR
# --------------------------------------------------

def process_dir(data_dir, prefix, df_all_pubs):
    desc_file = data_dir / f"{prefix}_cluster_descriptions.json"
    if not desc_file.exists():
        print(f"Skipping {data_dir.name}: no descriptions JSON.")
        return
    desc = json.loads(desc_file.read_text())

    print(f"\n── {prefix.upper()} @ {data_dir.name} ──")
    plot_researcher_umap(data_dir, prefix, desc)
    plot_publication_umap(data_dir, prefix, desc, df_all_pubs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--global-only", action="store_true", help="Skip school-level plots.")
    args = parser.parse_args()

    print("Loading measured publications...")
    measured_pubs_path = BASE / "1. data/3. final/4.All measured publications.csv"
    if not measured_pubs_path.exists():
        # Try capitalised variant
        measured_pubs_path = BASE / "1. data/3. Final/4.All measured publications.csv"
    df_all_pubs = pd.read_csv(measured_pubs_path, low_memory=False)
    df_all_pubs["n_id"] = pd.to_numeric(df_all_pubs["n_id"], errors="coerce").dropna().astype(int).astype(str)
    df_all_pubs["article_id"] = df_all_pubs["article_id"].astype(str)
    print(f"  {len(df_all_pubs):,} publications loaded.")

    # 1. Global
    process_dir(OUT_BASE / "global", "global", df_all_pubs)

    # 2. Schools
    if not args.global_only:
        schools_dir = OUT_BASE / "schools"
        if schools_dir.exists():
            school_dirs = sorted([d for d in schools_dir.iterdir() if d.is_dir()])
            print(f"\n{len(school_dirs)} school directories found.")
            for school_dir in school_dirs:
                process_dir(school_dir, "school", df_all_pubs)

    print("\n✅ All visualisations complete.")


if __name__ == "__main__":
    main()
