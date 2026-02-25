#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
TRISS V3 Pipeline: Semantic Clustering and Topic Modeling
Objective:
1. Topic clusters for ALL TRISS (global)
2. Topic clusters by SCHOOL
3. UMAP visualisations (global + by school)
4. LLM-based topic descriptions (using gpt-4o-mini)
5. Topic counts
6. Word clouds (global + by school)
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

# Which embeddings to use
DEFAULT_MODEL = "openai"

# Clustering hyperparams
GLOBAL_K_MIN = 8
GLOBAL_K_MAX = 25
SCHOOL_K_FORCED = 10   # force exactly 10 clusters per school
MIN_USERS_FOR_SCHOOL = 5

# UMAP hyperparams
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1
RANDOM_STATE = 42

# General exclusions for word clouds
STOPWORDS = set(ENGLISH_STOP_WORDS).union({
    "study", "analysis", "paper", "research", "results", "based", 
    "data", "approach", "using", "method", "model", "new", "used", 
    "different", "effect", "effects", "author", "authors", "article",
    "chapter", "book", "university", "college", "department"
})

def calculate_optimal_k(embeddings, k_min, k_max):
    best_k = k_min
    best_score = -1
    best_labels = None
    best_centers = None
    
    # Range upper exclusive
    for k in range(k_min, k_max + 1):
        kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=50)
        labels = kmeans.fit_predict(embeddings)
        
        # Silhouette requires at least 2 clusters
        if k > 1 and len(np.unique(labels)) > 1:
            score = silhouette_score(embeddings, labels)
            if score > best_score:
                best_score = score
                best_k = k
                best_labels = labels
                best_centers = kmeans.cluster_centers_
                
    return best_k, best_score, best_labels, best_centers

def generate_llm_descriptions(client, cluster_abstract_titles, top_keywords):
    """
    Given a list of titles and list of keywords, return a dictionary describing the topic.
    """
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
        print(f"Error calling OpenAI API: {e}")
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
        return # Empty vocab
        
    vocab = np.array(vec.get_feature_names_out())
    tf = np.asarray(X.sum(axis=0)).flatten()
    
    freq_dict = dict(zip(vocab, tf))
    
    wc = WordCloud(
        width=800, 
        height=400, 
        background_color="white", 
        max_words=100
    )
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
        # simplified TF-IDF per scikit-learn approach matching 18b
        tfidf = tf * np.log((1 + len(df)) / (1 + (X > 0).sum(axis=0))).A1
        top_idx = np.argsort(tfidf)[::-1][:top_n]
        topic_words[topic_id] = vocab[top_idx].tolist()
        
    return topic_words

def run_clustering_pipeline(df_users, df_pubs, out_dir, prefix, k_min, k_max, client):
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n--- Running clustering for {prefix} ({len(df_users)} users) ---")
    
    # 1. K-Means
    X_users = np.vstack(df_users["embedding"].values)
    best_k, best_score, labels, centers = calculate_optimal_k(X_users, k_min, min(k_max, len(X_users)-1))
    
    print(f"{prefix} Optimal K: {best_k} (Silhouette: {best_score:.4f})")
    
    assignments = df_users[["n_id", "school", "department"]].copy()
    assignments["cluster"] = labels
    assignments.to_csv(out_dir / f"{prefix}_cluster_assignments.csv", index=False)
    np.save(out_dir / f"{prefix}_cluster_centroids.npy", centers)
    
    # 2. UMAP
    print(f"Running UMAP for {prefix}...")
    reducer = umap.UMAP(n_neighbors=UMAP_N_NEIGHBORS, min_dist=UMAP_MIN_DIST, metric="cosine", random_state=RANDOM_STATE)
    umap_2d = reducer.fit_transform(X_users)
    
    umap_df = assignments.copy()
    umap_df["umap_x"] = umap_2d[:, 0]
    umap_df["umap_y"] = umap_2d[:, 1]
    umap_df.to_csv(out_dir / f"{prefix}_umap_coordinates.csv", index=False)
    
    # 3. LLM Topic Descriptions and 6. Word Clouds
    print(f"Generating LLM descriptions and Word Cloud for {prefix}...")
    
    # Map n_ids to publications for descriptions
    user_to_pubs = df_pubs.groupby("n_id")
    
    user_docs = []
    for n_id in df_users["n_id"]:
        if n_id in user_to_pubs.groups:
            titles = user_to_pubs.get_group(n_id)["Title"].fillna("").tolist()
            user_docs.append(" ".join(titles))
        else:
            user_docs.append("")
            
    # Top keywords
    topic_words = extract_tfidf_keywords(user_docs, labels, best_k)
    
    # Distance to centroids for central users
    sims = cosine_similarity(X_users, centers)
    
    descriptions = {}
    for cluster_id in range(best_k):
        # find top 25 users closest to this centroid
        user_idx_in_cluster = np.where(labels == cluster_id)[0]
        if len(user_idx_in_cluster) == 0:
            descriptions[str(cluster_id)] = {"topic_name": "Empty"}
            continue
            
        cluster_sims = sims[user_idx_in_cluster, cluster_id]
        top_idx_relative = np.argsort(cluster_sims)[::-1][:25]
        top_idx_absolute = user_idx_in_cluster[top_idx_relative]
        
        top_nids = df_users.iloc[top_idx_absolute]["n_id"].values
        
        # collect their publications
        cluster_titles = []
        for nid in top_nids:
            if nid in user_to_pubs.groups:
                cluster_titles.extend(user_to_pubs.get_group(nid)["Title"].fillna("").head(3).tolist()) # grab up to 3 titles per user for prompt length
                
        # Send to LLM
        desc_json = generate_llm_descriptions(client, cluster_titles[:40], topic_words[cluster_id])
        descriptions[str(cluster_id)] = desc_json
        
    with open(out_dir / f"{prefix}_cluster_descriptions.json", "w") as f:
        json.dump(descriptions, f, indent=2)
        
    # Word cloud for ALL publications in this slice
    all_slice_pubs = []
    for nid in df_users["n_id"]:
        if nid in user_to_pubs.groups:
            pubs = user_to_pubs.get_group(nid)
            texts = (pubs["Title"].fillna("") + " " + pubs["abstract"].fillna("")).tolist()
            all_slice_pubs.extend(texts)
            
    generate_wordcloud(all_slice_pubs, out_dir / f"{prefix}_wordcloud.png")
    
    # 4. Topic counts
    print(f"Calculating topic counts for {prefix}...")
    total_users = len(df_users)
    counts = assignments["cluster"].value_counts().reset_index()
    counts.columns = ["cluster", "size"]
    counts["proportion"] = counts["size"] / total_users
    
    # Breakdown by school
    school_breakdown = assignments.groupby(["cluster", "school"]).size().unstack(fill_value=0)
    counts = counts.merge(school_breakdown, left_on="cluster", right_index=True)
    
    counts.to_csv(out_dir / f"{prefix}_topic_counts.csv", index=False)
    
    # Validations
    assert len(assignments) == total_users
    assert counts["size"].sum() == total_users
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compare-mpnet", action="store_true")
    args = parser.parse_args()
    
    client = OpenAI()
    
    print("Loading Measured Publications...")
    # Load all measured publications to map abstracts/titles
    pubs_csv = DATA_DIR / "4.All measured publications.csv"
    with open(pubs_csv, 'r', encoding='utf-8', errors='replace') as f:
        df_pubs = pd.read_csv(f, low_memory=False)
    # Ensure n_id is string
    df_pubs["n_id"] = pd.to_numeric(df_pubs["n_id"], errors="coerce").dropna().astype(int).astype(str)
        
    print(f"Loading user average embeddings ({DEFAULT_MODEL})...")
    user_embed_dir = EMBED_DIR / DEFAULT_MODEL / "by_researcher"
    
    users = []
    for f in user_embed_dir.glob("n_*.json"):
        with open(f, "r") as jf:
            obj = json.load(jf)
            nid = str(obj["n_id"])
            emb = np.array(obj["embeddings"]["abstracts_mean"])
            if np.linalg.norm(emb) > 0:
                emb = emb / np.linalg.norm(emb)
            users.append({"n_id": nid, "embedding": emb})
            
    df_users = pd.DataFrame(users)
    
    if df_users.empty:
        print("No user embeddings found. Exiting.")
        return
        
    # Map school and department from the publications list (take first occurrence)
    meta = df_pubs.groupby("n_id").first()[["school", "department"]].reset_index()
    df_users = df_users.merge(meta, on="n_id", how="left")
    df_users["school"] = df_users["school"].fillna("Unknown")
    df_users["department"] = df_users["department"].fillna("Unknown")
    
    print(f"Loaded {len(df_users)} users with metadata.")
    
    # ----------------------------------------------------
    # GLOBAL TRISS
    # ----------------------------------------------------
    run_clustering_pipeline(
        df_users, 
        df_pubs, 
        OUT_BASE / "global", 
        "global", 
        GLOBAL_K_MIN, 
        GLOBAL_K_MAX,
        client
    )
    
    # ----------------------------------------------------
    # SCHOOL LEVEL
    # ----------------------------------------------------
    for school, group in df_users.groupby("school"):
        if school == "Unknown" or len(group) < MIN_USERS_FOR_SCHOOL:
            print(f"Skipping school: {school} (users: {len(group)})")
            continue
            
        slug = "".join(c if c.isalnum() else "_" for c in school).lower().strip("_")
        
        # Force K=10 for all schools (override silhouette search)
        school_k = min(SCHOOL_K_FORCED, len(group) - 1)
        run_clustering_pipeline(
            group.copy(),
            df_pubs,
            OUT_BASE / "schools" / slug,
            "school",
            school_k,
            school_k,
            client
        )
        
    print("\n✅ Pipelines successfully finished.")

if __name__ == "__main__":
    main()
