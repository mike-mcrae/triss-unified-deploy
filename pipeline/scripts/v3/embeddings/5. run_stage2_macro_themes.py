#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
TRISS V3 Pipeline — Stage 2: Hierarchical Macro-Theme Derivation
================================================================
Derives ~6 empirically grounded macro-themes from the existing Stage-1
K=25 OpenAI research clusters via agglomerative hierarchical clustering.

DOES NOT:
  • Re-run KMeans
  • Change base embeddings
  • Overwrite any Stage-1 cluster files

Inputs (from 1. data/5. analysis/global/):
  global_cluster_assignments.csv    — researcher → stage-1 cluster
  global_cluster_centroids.npy      — (25, 3072) centroid matrix
  global_cluster_descriptions.json  — stage-1 LLM labels

Reads supplementary data:
  1. data/3. Final/4.All measured publications.csv  — for abstracts/titles

Outputs (all in 1. data/5. analysis/global/stage2/):
  cluster_centroids_stage1.npy
  cluster_distance_matrix_stage2.npy
  macro_cluster_assignments.csv
  macro_cluster_samples.json
  macro_theme_descriptions.json
  researcher_macro_theme_weights.csv
  macro_theme_summary_stats.json
  macro_themes_overview.json
  stage2_dendrogram.png
"""

import json
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from openai import OpenAI
from tqdm import tqdm
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

matplotlib.use("Agg")

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
BASE          = _PIPELINE_ROOT
GLOBAL_DIR    = BASE / "1. data/5. analysis/global"
OUT_DIR       = GLOBAL_DIR / "stage2"
PUBS_CSV      = BASE / "1. data/3. Final/4.All measured publications.csv"
EMBED_DIR     = BASE / "1. data/4. embeddings/v3/openai/by_researcher"

TARGET_MACRO  = 6       # default target macro-themes
MACRO_MIN     = 4       # search range
MACRO_MAX     = 8

STOPWORDS = set(ENGLISH_STOP_WORDS).union({
    "study", "analysis", "paper", "research", "results", "based",
    "data", "approach", "using", "method", "model", "new", "used",
    "different", "effect", "effects", "author", "authors", "article",
    "chapter", "book", "university", "college", "department", "findings"
})

# --------------------------------------------------
# STEP 1 — Load existing Stage-1 outputs
# --------------------------------------------------
def step1_load():
    print("\n══ STEP 1: Load Stage-1 cluster outputs ══")
    assignments = pd.read_csv(GLOBAL_DIR / "global_cluster_assignments.csv")
    assignments["n_id"] = assignments["n_id"].astype(str)
    centroids = np.load(GLOBAL_DIR / "global_cluster_centroids.npy").astype(np.float32)
    with open(GLOBAL_DIR / "global_cluster_descriptions.json") as f:
        descriptions = json.load(f)

    K = int(centroids.shape[0])
    print(f"  Stage-1 clusters: K={K}")
    print(f"  Centroid shape: {centroids.shape}")
    print(f"  Researchers: {len(assignments)}")

    # Normalise centroids
    norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    norms[norms == 0] = 1
    centroids = centroids / norms

    # Save normalised centroids as stage1 artefact
    np.save(OUT_DIR / "cluster_centroids_stage1.npy", centroids)
    print(f"  Saved: cluster_centroids_stage1.npy")

    # Load researcher embeddings for later weighting
    researcher_embeddings = {}
    for f in sorted(EMBED_DIR.glob("n_*.json")):
        try:
            obj = json.loads(f.read_text())
            nid = str(obj["n_id"])
            vec = np.array(obj["embeddings"]["abstracts_mean"], dtype=np.float32)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec /= norm
            researcher_embeddings[nid] = vec
        except Exception:
            continue
    print(f"  Researcher embeddings loaded: {len(researcher_embeddings)}")

    return assignments, centroids, descriptions, K, researcher_embeddings


# --------------------------------------------------
# STEP 2 — Cluster-to-cluster similarity + distance
# --------------------------------------------------
def step2_distances(centroids, K):
    print("\n══ STEP 2: Compute cluster-to-cluster distance matrix ══")
    sim_matrix = cosine_similarity(centroids)
    dist_matrix = 1.0 - sim_matrix
    np.fill_diagonal(dist_matrix, 0.0)
    np.save(OUT_DIR / "cluster_distance_matrix_stage2.npy", dist_matrix)
    print(f"  Similarity range: [{sim_matrix.min():.3f}, {sim_matrix.max():.3f}]")
    print(f"  Saved: cluster_distance_matrix_stage2.npy")
    return sim_matrix, dist_matrix


# --------------------------------------------------
# STEP 3 — Hierarchical clustering on centroids
# --------------------------------------------------
def step3_hierarchical(centroids, dist_matrix, assignments, K, descriptions):
    print("\n══ STEP 3: Hierarchical clustering (Ward linkage) ══")
    condensed = squareform(dist_matrix, checks=False)
    Z = linkage(condensed, method="ward")

    # --- Dendrogram ---
    fig, ax = plt.subplots(figsize=(16, 7))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#0f1117")
    dendrogram(
        Z,
        labels=[
            descriptions.get(str(i), {}).get("topic_name", f"T{i}")[:20]
            for i in range(K)
        ],
        ax=ax,
        color_threshold=0,
        above_threshold_color="white",
        leaf_font_size=8,
        leaf_rotation=55
    )
    ax.set_title("Stage-2 Dendrogram — TRISS Cluster Hierarchy", color="white", fontsize=13)
    ax.tick_params(colors="#aaa")
    ax.yaxis.label.set_color("#aaa")
    for sp in ax.spines.values():
        sp.set_edgecolor("#333")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "stage2_dendrogram.png", dpi=150,
                bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print("  Saved: stage2_dendrogram.png")

    # --- Silhouette search for optimal cut ---
    best_k, best_score = TARGET_MACRO, -1
    print(f"  Silhouette search for optimal K ({MACRO_MIN}–{MACRO_MAX}):")
    for k in range(MACRO_MIN, MACRO_MAX + 1):
        labels = fcluster(Z, t=k, criterion="maxclust") - 1  # 0-indexed
        if len(np.unique(labels)) < 2:
            continue
        score = silhouette_score(centroids, labels, metric="cosine")
        print(f"    K={k}  silhouette={score:.4f}")
        if score > best_score:
            best_score, best_k = score, k

    macro_labels = fcluster(Z, t=best_k, criterion="maxclust") - 1  # 0-indexed
    print(f"\n  Optimal macro K={best_k}  (silhouette={best_score:.4f})")

    # Stage1 cluster → macro cluster map
    stage1_to_macro = {i: int(macro_labels[i]) for i in range(K)}

    macro_df = pd.DataFrame([
        {"cluster_id_stage1": cid, "macro_cluster_id": macro_labels[cid]}
        for cid in range(K)
    ])
    macro_df.to_csv(OUT_DIR / "macro_cluster_assignments.csv", index=False)
    print("  Saved: macro_cluster_assignments.csv")

    return macro_labels, stage1_to_macro, best_k


# --------------------------------------------------
# STEP 4 — Validate structural coherence
# --------------------------------------------------
def step4_samples(assignments, stage1_to_macro, descriptions, best_k):
    print("\n══ STEP 4: Validate structural coherence ══")
    pubs = pd.read_csv(PUBS_CSV, low_memory=False)
    pubs["n_id"] = pd.to_numeric(pubs["n_id"], errors="coerce").dropna().astype(int).astype(str)
    user_pubs = pubs.groupby("n_id")

    # Collect texts per stage-1 cluster
    cluster_texts: dict[int, list[str]] = {cid: [] for cid in stage1_to_macro}
    for _, row in assignments.iterrows():
        nid = str(row["n_id"])
        cid = int(row["cluster"])
        if nid in user_pubs.groups:
            grp = user_pubs.get_group(nid)
            titles = grp["Title"].fillna("").tolist()
            abstracts = grp["abstract"].fillna("").tolist()
            texts = [f"{t} {a}" for t, a in zip(titles, abstracts)]
            cluster_texts[cid].extend(texts)

    # TF-IDF top terms per stage-1 cluster
    all_corpus = {cid: " ".join(texts) for cid, texts in cluster_texts.items() if texts}
    try:
        vec = CountVectorizer(stop_words=list(STOPWORDS), ngram_range=(1, 2),
                              min_df=1, max_features=5000)
        all_docs = list(all_corpus.values())
        all_ids = list(all_corpus.keys())
        Xc = vec.fit_transform(all_docs)
        vocab = np.array(vec.get_feature_names_out())
    except Exception:
        vocab = np.array([])
        Xc = None

    def top_terms(idx_in_all, top_n=20):
        if Xc is None or len(vocab) == 0:
            return []
        tf = np.asarray(Xc[idx_in_all].sum(axis=0)).flatten()
        idf = np.log((1 + len(all_docs)) / (1 + (Xc > 0).sum(axis=0))).A1
        tfidf = tf * idf
        top = np.argsort(tfidf)[::-1][:top_n]
        return vocab[top].tolist()

    # Group into macro clusters
    macro_samples: dict[int, dict] = {mid: {
        "macro_cluster_id": mid,
        "stage1_clusters": [],
        "top_terms": [],
        "sample_abstracts": []
    } for mid in range(max(stage1_to_macro.values()) + 1)}

    for cid, mid in stage1_to_macro.items():
        topic_name = descriptions.get(str(cid), {}).get("topic_name", f"Cluster {cid}")
        keywords   = descriptions.get(str(cid), {}).get("keywords", [])
        # Representative abstracts: 10 per stage-1 cluster
        texts = cluster_texts.get(cid, [])[:10]
        idx = all_ids.index(cid) if cid in all_ids else None
        terms = top_terms(idx) if idx is not None else []
        macro_samples[mid]["stage1_clusters"].append({
            "cluster_id": cid,
            "topic_name": topic_name,
            "keywords": keywords,
            "top_terms": terms,
            "sample_texts": texts[:5]
        })
        macro_samples[mid]["top_terms"] = list(set(macro_samples[mid]["top_terms"] + terms))
        macro_samples[mid]["sample_abstracts"].extend(texts[:3])

    out = {str(k): v for k, v in macro_samples.items()}
    with open(OUT_DIR / "macro_cluster_samples.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"  Saved: macro_cluster_samples.json  ({len(macro_samples)} macro-clusters)")
    return macro_samples, cluster_texts, user_pubs, pubs


# --------------------------------------------------
# STEP 5 — LLM macro-theme labelling
# --------------------------------------------------
def step5_llm(macro_samples, descriptions, assignments, stage1_to_macro, best_k):
    print("\n══ STEP 5: Generate empirical macro-theme labels (LLM) ══")
    client = OpenAI()
    macro_descriptions = {}

    # School distribution per macro cluster
    assignments["macro"] = assignments["cluster"].apply(lambda c: stage1_to_macro.get(int(c), -1))
    for mid in tqdm(range(best_k), desc="Macro themes"):
        mc = macro_samples.get(mid, {})
        stage1 = mc.get("stage1_clusters", [])
        sub_names   = [s["topic_name"] for s in stage1]
        all_keywords = list({kw for s in stage1 for kw in (s.get("keywords") or [])})
        all_terms = list({t for s in stage1 for t in (s.get("top_terms") or [])})[:30]
        sample_texts = mc.get("sample_abstracts", [])[:15]
        schools = assignments[assignments["macro"] == mid]["school"].value_counts().to_dict()

        prompt = f"""You are a senior research analyst mapping the thematic landscape of a university's research output.

The following academic sub-themes have been empirically grouped into a single macro-theme via hierarchical clustering.

Sub-themes (Stage-1 Clusters):
{json.dumps(sub_names, indent=2)}

Top TF-IDF Terms:
{json.dumps(all_terms[:20], indent=2)}

Related Keywords:
{json.dumps(all_keywords[:20], indent=2)}

Sample Research Abstracts (10–15):
{json.dumps(sample_texts, indent=2)}

Schools contributing to this macro-theme:
{json.dumps(schools, indent=2)}

Based ONLY on the above evidence (do NOT hallucinate), return JSON with this exact structure:
{{
  "theme_name": "<3–6 word name>",
  "description": "<2–3 sentence empirical description of the shared conceptual core>",
  "policy_relevance": "<2–3 sentences on relevant policy domains and societal value>",
  "methodological_approaches": ["<approach 1>", "<approach 2>", "<...>"],
  "funding_domains": ["<domain 1>", "<domain 2>", "<...>"],
  "keywords": ["<keyword 1>", "<keyword 2>", "<...8–12 keywords...>"]
}}"""

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            result = json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"  ⚠  LLM error for macro {mid}: {e}")
            result = {
                "theme_name": f"Macro Theme {mid}",
                "description": "",
                "policy_relevance": "",
                "methodological_approaches": [],
                "funding_domains": [],
                "keywords": all_terms[:8]
            }
        macro_descriptions[str(mid)] = result
        print(f"  Macro {mid}: {result.get('theme_name','?')}")

    with open(OUT_DIR / "macro_theme_descriptions.json", "w") as f:
        json.dump(macro_descriptions, f, indent=2)
    print("  Saved: macro_theme_descriptions.json")
    return macro_descriptions


# --------------------------------------------------
# STEP 6 — Map researchers to macro-themes
# --------------------------------------------------
def step6_researcher_weights(assignments, centroids, researcher_embeddings,
                              stage1_to_macro, best_k, macro_descriptions):
    print("\n══ STEP 6: Map researchers to macro-themes ══")
    rows = []
    for _, row in assignments.iterrows():
        nid = str(row["n_id"])
        vec = researcher_embeddings.get(nid)
        if vec is None:
            # Hard assignment only
            hard_macro = stage1_to_macro.get(int(row["cluster"]), 0)
            weights = {f"macro_theme_{m}": 0.0 for m in range(best_k)}
            weights[f"macro_theme_{hard_macro}"] = 1.0
            primary = hard_macro
        else:
            # Soft assignment: cosine similarity to all 25 centroids,
            # then sum similarities within each macro group and normalise
            sims = centroids @ vec  # (K,)
            sims = np.clip(sims, 0, None)   # only positive relevance
            macro_scores = np.zeros(best_k)
            for cid in range(len(sims)):
                macro_id = stage1_to_macro.get(cid, 0)
                macro_scores[macro_id] += float(sims[cid])
            total = macro_scores.sum()
            if total > 0:
                macro_scores /= total
            primary = int(np.argmax(macro_scores))
            weights = {f"macro_theme_{m}": round(float(macro_scores[m]), 5)
                       for m in range(best_k)}

        row_out = {"n_id": nid,
                   "primary_macro_theme": primary,
                   "primary_macro_theme_name": macro_descriptions.get(str(primary), {}).get("theme_name", ""),
                   **weights}
        rows.append(row_out)

    df_weights = pd.DataFrame(rows)
    df_weights.to_csv(OUT_DIR / "researcher_macro_theme_weights.csv", index=False)
    print(f"  Saved: researcher_macro_theme_weights.csv  ({len(df_weights)} researchers)")
    return df_weights


# --------------------------------------------------
# STEP 7 — Summary statistics
# --------------------------------------------------
def step7_stats(df_weights, assignments, macro_descriptions, stage1_to_macro,
                best_k, macro_samples):
    print("\n══ STEP 7: TRISS-level summary statistics ══")
    weight_cols = [f"macro_theme_{m}" for m in range(best_k)]

    # Shannon entropy per researcher (cross-theme spread)
    def entropy(row):
        w = np.array([row[c] for c in weight_cols], dtype=float)
        w = w[w > 0]
        if len(w) == 0:
            return 0.0
        return float(-np.sum(w * np.log(w + 1e-12)))

    df_weights["entropy"] = df_weights.apply(entropy, axis=1)

    # Researchers per macro-theme (primary)
    n_per_theme = df_weights["primary_macro_theme"].value_counts().to_dict()

    # School distribution per macro-theme
    assignments["macro"] = assignments["cluster"].apply(lambda c: stage1_to_macro.get(int(c), -1))
    school_dist = {}
    for mid in range(best_k):
        sub = assignments[assignments["macro"] == mid]
        school_dist[mid] = sub["school"].value_counts().to_dict()

    # Cross-theme overlap (how many researchers have >10% weight in >1 theme)
    def multi_theme(row):
        return sum(1 for c in weight_cols if row[c] > 0.10) > 1

    pct_multi = float(df_weights.apply(multi_theme, axis=1).mean()) * 100

    # Bridging researchers (top 20 by highest entropy)
    bridging = (df_weights.nlargest(20, "entropy")
                [["n_id", "primary_macro_theme", "primary_macro_theme_name", "entropy"]]
                .to_dict(orient="records"))

    # Cross-theme similarity matrix (macro centroid means)
    macro_centroid_sims = {}
    for i in range(best_k):
        for j in range(best_k):
            macro_centroid_sims[f"{i}_{j}"] = 0.0  # placeholder — filled from weight correlations

    # Correlation between researcher weight vectors
    if len(df_weights) > 1:
        W = df_weights[weight_cols].values
        corr = np.corrcoef(W.T)
        overlap_matrix = {f"{i}_{j}": round(float(corr[i, j]), 4)
                          for i in range(best_k) for j in range(i + 1, best_k)}
    else:
        overlap_matrix = {}

    stats = {
        "n_macro_themes": best_k,
        "researchers_per_macro_theme": {str(k): int(v) for k, v in n_per_theme.items()},
        "school_distribution_per_theme": {str(k): v for k, v in school_dist.items()},
        "pct_researchers_multi_theme": round(pct_multi, 1),
        "cross_theme_overlap_matrix": overlap_matrix,
        "bridging_researchers": bridging,
        "theme_names": {
            str(mid): macro_descriptions.get(str(mid), {}).get("theme_name", f"Theme {mid}")
            for mid in range(best_k)
        }
    }
    with open(OUT_DIR / "macro_theme_summary_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  Saved: macro_theme_summary_stats.json")
    print(f"  Multi-theme researchers: {pct_multi:.1f}%")
    return stats


# --------------------------------------------------
# STEP 8 — Final deliverables for unified app
# --------------------------------------------------
def step8_deliverables(macro_descriptions, macro_samples, df_weights,
                       assignments, stage1_to_macro, best_k,
                       stats, descriptions):
    print("\n══ STEP 8: Final deliverables for unified app ══")

    # macro_themes_overview.json
    overview = []
    for mid in range(best_k):
        desc = macro_descriptions.get(str(mid), {})
        mc   = macro_samples.get(mid, {})
        stage1_ids = [s["cluster_id"] for s in mc.get("stage1_clusters", [])]
        n_researchers = int(stats["researchers_per_macro_theme"].get(str(mid), 0))
        schools_contributing = list(stats["school_distribution_per_theme"].get(str(mid), {}).keys())
        overview.append({
            "theme_id": mid,
            "title": desc.get("theme_name", f"Theme {mid}"),
            "description": desc.get("description", ""),
            "policy_relevance": desc.get("policy_relevance", ""),
            "methodological_approaches": desc.get("methodological_approaches", []),
            "funding_domains": desc.get("funding_domains", []),
            "keywords": desc.get("keywords", []),
            "schools_contributing": schools_contributing,
            "n_researchers": n_researchers,
            "stage1_cluster_ids": stage1_ids
        })

    with open(OUT_DIR / "macro_themes_overview.json", "w") as f:
        json.dump(overview, f, indent=2)
    print("  Saved: macro_themes_overview.json")

    # macro_theme_cluster_map.csv
    cluster_map_rows = []
    for cid, mid in stage1_to_macro.items():
        cluster_map_rows.append({
            "macro_theme_id": mid,
            "macro_theme_name": macro_descriptions.get(str(mid), {}).get("theme_name", ""),
            "stage1_cluster_id": cid,
            "stage1_cluster_name": descriptions.get(str(cid), {}).get("topic_name", "")
        })
    pd.DataFrame(cluster_map_rows).to_csv(OUT_DIR / "macro_theme_cluster_map.csv", index=False)
    print("  Saved: macro_theme_cluster_map.csv")

    # Print final summary
    print("\n┌─────────────────────────────────────────────────────────┐")
    print("│  TRISS MACRO-THEME SUMMARY                              │")
    print("├─────────────────────────────────────────────────────────┤")
    for item in overview:
        theme_id = item["theme_id"]
        name = item["title"]
        n = item["n_researchers"]
        clusters = item["stage1_cluster_ids"]
        stage1_names = [descriptions.get(str(c), {}).get("topic_name", f"C{c}") for c in clusters]
        print(f"│  Theme {theme_id} ({n:3d} researchers): {name[:40]}")
        for sn in stage1_names:
            print(f"│    ↳ {sn[:50]}")
        print("│")
    print("└─────────────────────────────────────────────────────────┘")


# --------------------------------------------------
# MAIN
# --------------------------------------------------
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUT_DIR}")

    assignments, centroids, descriptions, K, researcher_embeddings = step1_load()
    sim_matrix, dist_matrix = step2_distances(centroids, K)
    macro_labels, stage1_to_macro, best_k = step3_hierarchical(
        centroids, dist_matrix, assignments, K, descriptions)
    macro_samples, cluster_texts, user_pubs, pubs = step4_samples(
        assignments, stage1_to_macro, descriptions, best_k)
    macro_descriptions = step5_llm(
        macro_samples, descriptions, assignments, stage1_to_macro, best_k)
    df_weights = step6_researcher_weights(
        assignments, centroids, researcher_embeddings,
        stage1_to_macro, best_k, macro_descriptions)
    stats = step7_stats(
        df_weights, assignments, macro_descriptions,
        stage1_to_macro, best_k, macro_samples)
    step8_deliverables(
        macro_descriptions, macro_samples, df_weights,
        assignments, stage1_to_macro, best_k, stats, descriptions)

    print("\n✅ Stage-2 pipeline complete.")
    print(f"All outputs in: {OUT_DIR}")


if __name__ == "__main__":
    main()
