#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
TRISS V3 Pipeline: Forced-10-Topic Semantic Clustering (TEST/COMPARISON ONLY)
=============================================================================
This is a stripped-down version of '3. run_semantic_clustering.py' that:
  - Runs TRISS-level clustering ONLY (no school-level breakdown)
  - Forces exactly K=10 clusters (no silhouette-based search)
  - Writes all output to: 1. data/5. analysis/global/topic_10/

Purpose: assess what the TRISS topic landscape looks like with fewer, 
broader clusters compared to the optimised K found by the main script.

Outputs in topic_10/:
  global_cluster_assignments.csv   — researcher → cluster ID
  global_cluster_centroids.npy     — (10, D) centroid matrix
  global_umap_coordinates.csv      — 2-D UMAP coords per researcher
  global_cluster_descriptions.json — LLM topic names/descriptions/keywords
  global_topic_counts.csv          — sizes and school breakdown per cluster
  global_wordcloud.png             — word cloud of all publications
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from sklearn.cluster import KMeans
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
DATA_DIR   = BASE / "1. data/3. Final"
EMBED_DIR  = BASE / "1. data/4. embeddings/v3"
OUT_DIR    = BASE / "1. data/5. analysis/global/topic_10"

FORCED_K   = 20          # ← the only change that matters
MODEL      = "openai"    # embedding source (same as main script)
RANDOM_STATE = 42

UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST    = 0.1

STOPWORDS = set(ENGLISH_STOP_WORDS).union({
    "study", "analysis", "paper", "research", "results", "based",
    "data", "approach", "using", "method", "model", "new", "used",
    "different", "effect", "effects", "author", "authors", "article",
    "chapter", "book", "university", "college", "department"
})

# --------------------------------------------------
# HELPERS (copied verbatim from main script)
# --------------------------------------------------

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


def generate_llm_description(client, cluster_titles, top_keywords):
    prompt = f"""
You are a semantic cluster explicitly summarizing an academic topic.
Based on the following titles and TF-IDF keywords, provide a JSON response summarizing the cluster.

Titles:
{json.dumps(cluster_titles, indent=2)}

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
    print(f"  Word cloud → {out_file}")


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    client = OpenAI()

    # 1. Load publications (for titles / metadata)
    print("Loading publications...")
    pubs_csv = DATA_DIR / "4.All measured publications.csv"
    with open(pubs_csv, "r", encoding="utf-8", errors="replace") as f:
        df_pubs = pd.read_csv(f, low_memory=False)
    df_pubs["n_id"] = pd.to_numeric(df_pubs["n_id"], errors="coerce").dropna().astype(int).astype(str)
    print(f"  {len(df_pubs):,} publication rows loaded.")

    # 2. Load researcher-level OpenAI embeddings
    print(f"Loading researcher embeddings ({MODEL})...")
    user_embed_dir = EMBED_DIR / MODEL / "by_researcher"
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
        print("No user embeddings found. Exiting.")
        return

    # Attach school/department metadata from publications
    meta = df_pubs.groupby("n_id").first()[["school", "department"]].reset_index()
    df_users = df_users.merge(meta, on="n_id", how="left")
    df_users["school"]     = df_users["school"].fillna("Unknown")
    df_users["department"] = df_users["department"].fillna("Unknown")
    print(f"  {len(df_users)} researchers with embeddings.")

    X = np.vstack(df_users["embedding"].values)

    # 3. Forced K=10 KMeans
    print(f"\nRunning KMeans with K={FORCED_K} (forced, no silhouette search)...")
    kmeans = KMeans(n_clusters=FORCED_K, random_state=RANDOM_STATE, n_init=50)
    labels  = kmeans.fit_predict(X)
    centers = kmeans.cluster_centers_

    assignments = df_users[["n_id", "school", "department"]].copy()
    assignments["cluster"] = labels
    assignments.to_csv(OUT_DIR / "global_cluster_assignments.csv", index=False)
    np.save(OUT_DIR / "global_cluster_centroids.npy", centers)
    print(f"  Cluster distribution:\n{assignments['cluster'].value_counts().sort_index().to_string()}")

    # 4. UMAP
    print(f"\nRunning UMAP...")
    reducer = umap.UMAP(
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        metric="cosine",
        random_state=RANDOM_STATE
    )
    umap_2d = reducer.fit_transform(X)
    umap_df = assignments.copy()
    umap_df["umap_x"] = umap_2d[:, 0]
    umap_df["umap_y"]  = umap_2d[:, 1]
    umap_df.to_csv(OUT_DIR / "global_umap_coordinates.csv", index=False)
    print(f"  UMAP → {OUT_DIR / 'global_umap_coordinates.csv'}")

    # 5. TF-IDF keywords
    print("\nExtracting TF-IDF keywords per cluster...")
    user_to_pubs = df_pubs.groupby("n_id")
    user_docs = []
    for nid in df_users["n_id"]:
        if nid in user_to_pubs.groups:
            titles = user_to_pubs.get_group(nid)["Title"].fillna("").tolist()
            user_docs.append(" ".join(titles))
        else:
            user_docs.append("")

    topic_words = extract_tfidf_keywords(user_docs, labels, FORCED_K)

    # 6. LLM descriptions
    print(f"\nGenerating LLM descriptions for {FORCED_K} topics...")
    sims = cosine_similarity(X, centers)
    descriptions = {}

    for cluster_id in tqdm(range(FORCED_K), desc="Topics"):
        user_idx_in_cluster = np.where(labels == cluster_id)[0]
        if len(user_idx_in_cluster) == 0:
            descriptions[str(cluster_id)] = {"topic_name": "Empty", "description": "", "keywords": []}
            continue

        cluster_sims = sims[user_idx_in_cluster, cluster_id]
        top_idx_rel  = np.argsort(cluster_sims)[::-1][:25]
        top_idx_abs  = user_idx_in_cluster[top_idx_rel]
        top_nids     = df_users.iloc[top_idx_abs]["n_id"].values

        cluster_titles = []
        for nid in top_nids:
            if nid in user_to_pubs.groups:
                cluster_titles.extend(
                    user_to_pubs.get_group(nid)["Title"].fillna("").head(3).tolist()
                )

        desc = generate_llm_description(client, cluster_titles[:40], topic_words[cluster_id])
        descriptions[str(cluster_id)] = desc
        print(f"  Topic {cluster_id}: {desc.get('topic_name', '?')}")

    with open(OUT_DIR / "global_cluster_descriptions.json", "w") as f:
        json.dump(descriptions, f, indent=2)
    print(f"  Descriptions → {OUT_DIR / 'global_cluster_descriptions.json'}")

    # 7. Topic counts + school breakdown
    counts = assignments["cluster"].value_counts().reset_index()
    counts.columns = ["cluster", "size"]
    counts["proportion"] = counts["size"] / len(assignments)
    school_breakdown = assignments.groupby(["cluster", "school"]).size().unstack(fill_value=0)
    counts = counts.merge(school_breakdown, left_on="cluster", right_index=True, how="left")
    counts.to_csv(OUT_DIR / "global_topic_counts.csv", index=False)

    # 8. Word cloud
    print("\nGenerating word cloud...")
    all_pubs_text = []
    for nid in df_users["n_id"]:
        if nid in user_to_pubs.groups:
            grp = user_to_pubs.get_group(nid)
            texts = (grp["Title"].fillna("") + " " + grp["abstract"].fillna("")).tolist()
            all_pubs_text.extend(texts)
    generate_wordcloud(all_pubs_text, OUT_DIR / "global_wordcloud.png")

    print(f"\n✅ Done! All outputs in:\n   {OUT_DIR}")
    print("\nSummary:")
    for cid in sorted(descriptions, key=int):
        name  = descriptions[cid].get("topic_name", "?")
        n_res = int((assignments["cluster"] == int(cid)).sum())
        print(f"  Cluster {int(cid):2d} ({n_res:3d} researchers): {name}")


if __name__ == "__main__":
    main()
