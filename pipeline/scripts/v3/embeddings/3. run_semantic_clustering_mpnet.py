#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
TRISS V3 Pipeline: Full Semantic Clustering with MPNet Embeddings
=================================================================
MPNet (all-mpnet-base-v2) equivalent of '3. run_semantic_clustering.py'.

Differences from the OpenAI version:
  - Embedding source: mpnet/by_researcher/ (D=768 vs 3072 for OpenAI Large-3)
  - Output base: 1. data/5. analysis/global_mpnet/  and  .../schools_mpnet/
  - Everything else (K-search, UMAP, LLM descriptions, word clouds) is identical

Produces:
  Global:
    global_cluster_assignments.csv
    global_cluster_centroids.npy
    global_umap_coordinates.csv
    global_cluster_descriptions.json
    global_topic_counts.csv
    global_wordcloud.png

  Per school (if ≥ MIN_USERS_FOR_SCHOOL researchers):
    schools_mpnet/<slug>/school_cluster_assignments.csv
    schools_mpnet/<slug>/school_cluster_centroids.npy
    schools_mpnet/<slug>/school_umap_coordinates.csv
    schools_mpnet/<slug>/school_cluster_descriptions.json
    schools_mpnet/<slug>/school_topic_counts.csv
    schools_mpnet/<slug>/school_wordcloud.png
"""

import json
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
from tqdm import tqdm

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
import umap.umap_ as umap
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from openai import OpenAI
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
BASE = _PIPELINE_ROOT
DATA_DIR = BASE / "1. data/3. Final"
EMBED_DIR = BASE / "1. data/4. embeddings/v3"

OUT_BASE = BASE / "1. data/5. analysis"

MODEL = "mpnet"        # ← MPNet instead of OpenAI

# Clustering hyperparams (same ranges as OpenAI version)
GLOBAL_K_MIN = 8
GLOBAL_K_MAX = 25
SCHOOL_K_MIN = 4
SCHOOL_K_MAX = 12
MIN_USERS_FOR_SCHOOL = 5

# UMAP hyperparams
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST    = 0.1
RANDOM_STATE     = 42

STOPWORDS = set(ENGLISH_STOP_WORDS).union({
    "study", "analysis", "paper", "research", "results", "based",
    "data", "approach", "using", "method", "model", "new", "used",
    "different", "effect", "effects", "author", "authors", "article",
    "chapter", "book", "university", "college", "department"
})

# Output dirs — separate from the OpenAI outputs so nothing is overwritten
GLOBAL_OUT_DIR = OUT_BASE / "global_mpnet"
SCHOOLS_OUT_DIR = OUT_BASE / "schools_mpnet"


# --------------------------------------------------
# HELPERS (identical to the OpenAI script)
# --------------------------------------------------

def calculate_optimal_k(embeddings, k_min, k_max):
    best_k = k_min
    best_score = -1
    best_labels = None
    best_centers = None

    for k in range(k_min, k_max + 1):
        kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=50)
        labels = kmeans.fit_predict(embeddings)

        if k > 1 and len(np.unique(labels)) > 1:
            score = silhouette_score(embeddings, labels)
            print(f"  K={k:2d}  silhouette={score:.4f}")
            if score > best_score:
                best_score = score
                best_k = k
                best_labels = labels
                best_centers = kmeans.cluster_centers_

    return best_k, best_score, best_labels, best_centers


def generate_llm_descriptions(client, cluster_abstract_titles, top_keywords):
    prompt = f"""
You are a semantic cluster explicitly summarizing an academic topic.
Based on the following titles and TF-IDF keywords, provide a JSON response summarizing the cluster.

Titles:
{json.dumps(cluster_abstract_titles, indent=2)}

Keywords:
{json.dumps(top_keywords)}

Return ONLY JSON matching this structure exactly:
{{
    "topic_name": "<Short concise name, max 6 words>",
    "description": "<2-3 sentence thematic description>",
    "keywords": ["<keyword1>", "<keyword2>", "...5 to 10 keywords..."]
}}
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"  ⚠  OpenAI API error: {e}")
        return {
            "topic_name": "Unknown Topic",
            "description": "Failed to generate LLM description.",
            "keywords": top_keywords[:5]
        }


def generate_wordcloud(text_corpus, out_file):
    if not text_corpus:
        return
    vec = CountVectorizer(stop_words=list(STOPWORDS), ngram_range=(1, 2), min_df=2)
    try:
        X = vec.fit_transform(text_corpus)
    except ValueError:
        return
    vocab = np.array(vec.get_feature_names_out())
    tf = np.asarray(X.sum(axis=0)).flatten()
    freq_dict = dict(zip(vocab, tf))
    wc = WordCloud(width=800, height=400, background_color="white", max_words=100)
    wc.generate_from_frequencies(freq_dict)
    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.savefig(out_file, bbox_inches="tight")
    plt.close()


def extract_tfidf_keywords(texts, labels, k_clusters, top_n=10):
    df = pd.DataFrame({"topic": labels, "text": texts})
    vectorizer = CountVectorizer(stop_words=list(STOPWORDS), ngram_range=(1, 2), min_df=2)
    try:
        X = vectorizer.fit_transform(df["text"])
        vocab = np.array(vectorizer.get_feature_names_out())
    except ValueError:
        return {i: [] for i in range(k_clusters)}

    topic_words = {}
    for topic_id in range(k_clusters):
        idx = df["topic"] == topic_id
        if not idx.any():
            topic_words[topic_id] = []
            continue
        tf = np.asarray(X[idx.values].sum(axis=0)).flatten()
        tfidf = tf * np.log((1 + len(df)) / (1 + (X > 0).sum(axis=0))).A1
        top_idx = np.argsort(tfidf)[::-1][:top_n]
        topic_words[topic_id] = vocab[top_idx].tolist()

    return topic_words


def run_clustering_pipeline(df_users, df_pubs, out_dir, prefix, k_min, k_max, client):
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n--- Running MPNet clustering for {prefix} ({len(df_users)} users) ---")

    # 1. K-Means with silhouette search
    X_users = np.vstack(df_users["embedding"].values)
    best_k, best_score, labels, centers = calculate_optimal_k(X_users, k_min, min(k_max, len(X_users) - 1))

    print(f"{prefix} Optimal K: {best_k} (Silhouette: {best_score:.4f})")

    assignments = df_users[["n_id", "school", "department"]].copy()
    assignments["cluster"] = labels
    assignments.to_csv(out_dir / f"{prefix}_cluster_assignments.csv", index=False)
    np.save(out_dir / f"{prefix}_cluster_centroids.npy", centers)

    # 2. UMAP
    print(f"  Running UMAP for {prefix}...")
    reducer = umap.UMAP(
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        metric="cosine",
        random_state=RANDOM_STATE
    )
    umap_2d = reducer.fit_transform(X_users)

    umap_df = assignments.copy()
    umap_df["umap_x"] = umap_2d[:, 0]
    umap_df["umap_y"]  = umap_2d[:, 1]
    umap_df.to_csv(out_dir / f"{prefix}_umap_coordinates.csv", index=False)

    # 3. TF-IDF keywords + LLM descriptions + word clouds
    print(f"  Generating LLM descriptions for {prefix}...")

    user_to_pubs = df_pubs.groupby("n_id")

    user_docs = []
    for n_id in df_users["n_id"]:
        if n_id in user_to_pubs.groups:
            titles = user_to_pubs.get_group(n_id)["Title"].fillna("").tolist()
            user_docs.append(" ".join(titles))
        else:
            user_docs.append("")

    topic_words = extract_tfidf_keywords(user_docs, labels, best_k)

    sims = cosine_similarity(X_users, centers)
    descriptions = {}

    for cluster_id in tqdm(range(best_k), desc=f"  LLM ({prefix})"):
        user_idx_in_cluster = np.where(labels == cluster_id)[0]
        if len(user_idx_in_cluster) == 0:
            descriptions[str(cluster_id)] = {"topic_name": "Empty"}
            continue

        cluster_sims     = sims[user_idx_in_cluster, cluster_id]
        top_idx_relative = np.argsort(cluster_sims)[::-1][:25]
        top_idx_absolute = user_idx_in_cluster[top_idx_relative]
        top_nids         = df_users.iloc[top_idx_absolute]["n_id"].values

        cluster_titles = []
        for nid in top_nids:
            if nid in user_to_pubs.groups:
                cluster_titles.extend(
                    user_to_pubs.get_group(nid)["Title"].fillna("").head(3).tolist()
                )

        desc_json = generate_llm_descriptions(client, cluster_titles[:40], topic_words[cluster_id])
        descriptions[str(cluster_id)] = desc_json

    with open(out_dir / f"{prefix}_cluster_descriptions.json", "w") as f:
        json.dump(descriptions, f, indent=2)

    # Word cloud
    all_slice_pubs = []
    for nid in df_users["n_id"]:
        if nid in user_to_pubs.groups:
            grp = user_to_pubs.get_group(nid)
            texts = (grp["Title"].fillna("") + " " + grp["abstract"].fillna("")).tolist()
            all_slice_pubs.extend(texts)

    generate_wordcloud(all_slice_pubs, out_dir / f"{prefix}_wordcloud.png")

    # 4. Topic counts
    total_users = len(df_users)
    counts = assignments["cluster"].value_counts().reset_index()
    counts.columns = ["cluster", "size"]
    counts["proportion"] = counts["size"] / total_users

    school_breakdown = assignments.groupby(["cluster", "school"]).size().unstack(fill_value=0)
    counts = counts.merge(school_breakdown, left_on="cluster", right_index=True)
    counts.to_csv(out_dir / f"{prefix}_topic_counts.csv", index=False)

    assert len(assignments) == total_users
    assert counts["size"].sum() == total_users

    print(f"  ✅ {prefix} done — K={best_k}, outputs in {out_dir}")


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run TRISS semantic clustering using MPNet embeddings.")
    parser.add_argument("--global-only", action="store_true", help="Skip school-level clustering.")
    args = parser.parse_args()

    client = OpenAI()

    print("Loading publications...")
    pubs_csv = DATA_DIR / "4.All measured publications.csv"
    with open(pubs_csv, "r", encoding="utf-8", errors="replace") as f:
        df_pubs = pd.read_csv(f, low_memory=False)
    df_pubs["n_id"] = pd.to_numeric(df_pubs["n_id"], errors="coerce").dropna().astype(int).astype(str)
    print(f"  {len(df_pubs):,} publication rows.")

    print(f"Loading MPNet researcher embeddings...")
    user_embed_dir = EMBED_DIR / MODEL / "by_researcher"
    if not user_embed_dir.exists():
        print(f"ERROR: MPNet embedding directory not found: {user_embed_dir}")
        return

    users = []
    for f in sorted(user_embed_dir.glob("n_*.json")):
        with open(f, "r") as jf:
            obj = json.load(jf)
        nid = str(obj["n_id"])
        emb = np.array(obj["embeddings"]["abstracts_mean"])
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        users.append({"n_id": nid, "embedding": emb})

    df_users = pd.DataFrame(users)
    if df_users.empty:
        print("No MPNet user embeddings found. Exiting.")
        return

    meta = df_pubs.groupby("n_id").first()[["school", "department"]].reset_index()
    df_users = df_users.merge(meta, on="n_id", how="left")
    df_users["school"]     = df_users["school"].fillna("Unknown")
    df_users["department"] = df_users["department"].fillna("Unknown")

    print(f"  {len(df_users)} researchers. Embedding dim: {df_users['embedding'].iloc[0].shape[0]}")

    # ----------------------------
    # GLOBAL TRISS
    # ----------------------------
    run_clustering_pipeline(
        df_users,
        df_pubs,
        GLOBAL_OUT_DIR,
        "global",
        GLOBAL_K_MIN,
        GLOBAL_K_MAX,
        client
    )

    if args.global_only:
        print("\n✅ Global-only run complete.")
        return

    # ----------------------------
    # SCHOOL LEVEL
    # ----------------------------
    for school, group in df_users.groupby("school"):
        if school == "Unknown" or len(group) < MIN_USERS_FOR_SCHOOL:
            print(f"  Skipping school: {school} ({len(group)} users)")
            continue

        slug = "".join(c if c.isalnum() else "_" for c in school).lower().strip("_")

        run_clustering_pipeline(
            group.copy(),
            df_pubs,
            SCHOOLS_OUT_DIR / slug,
            "school",
            SCHOOL_K_MIN,
            min(SCHOOL_K_MAX, len(group) - 1),
            client
        )

    print("\n✅ MPNet clustering pipeline finished.")


if __name__ == "__main__":
    main()
