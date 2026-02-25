#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute user-level average embeddings and similarity matrices.
1. Groups publication embeddings by n_id and mean-pools them.
2. Saves user average embeddings to by_researcher/n_{n_id}.json.
3. Computes N x N researcher similarity matrix -> 7.researcher_similarity_matrix_{model}.csv
4. Computes user-to-other-publications similarity -> 8.user_to_other_publication_similarity_{model}.csv
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
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
BASE = _PIPELINE_ROOT
MODELS = ["openai", "mpnet"]

def main():
    for model in MODELS:
        print(f"\n{'='*50}")
        print(f"Processing model: {model}")
        print(f"{'='*50}")
        
        pub_dir = BASE / f"1. data/4. embeddings/v3/{model}/by_publication"
        user_dir = BASE / f"1. data/4. embeddings/v3/{model}/by_researcher"
        user_dir.mkdir(parents=True, exist_ok=True)
        
        out_matrix = BASE / f"1. data/4. embeddings/v3/{model}/7.researcher_similarity_matrix_{model}.csv"
        out_sims = BASE / f"1. data/4. embeddings/v3/{model}/8.user_to_other_publication_similarity_{model}.csv"
        
        # 1. Load pub embeddings
        print("Loading publication embeddings...")
        pubs = []
        for f in tqdm(list(pub_dir.glob("*.json")), desc="Reading pub JSONs"):
            with open(f, "r") as jf:
                obj = json.load(jf)
                # Ignore unknown n_ids as they can't be assigned to a specific user
                if obj.get("n_id") != "unknown" and obj.get("n_id") is not None:
                    # n_id could be a float initially converted to string, so parse safety
                    n_id_val = str(obj["n_id"]).replace(".0", "")
                    pubs.append({
                        "n_id": n_id_val,
                        "article_id": str(obj.get("article_id", "")),
                        "embedding": np.array(obj["embeddings"]["abstract"])
                    })
        
        df_pubs = pd.DataFrame(pubs)
        
        if df_pubs.empty:
            print(f"No valid publications found for {model}. Skipping.")
            continue
            
        print(f"Loaded {len(df_pubs)} valid publication embeddings.")
        
        # Normalize pub embeddings to be safe
        df_pubs["embedding"] = df_pubs["embedding"].apply(lambda v: v / np.linalg.norm(v))
        
        # 2. Compute Mean User Embeddings
        print("Computing user-level average embeddings...")
        mean_vectors = {}
        for n_id, group in tqdm(df_pubs.groupby("n_id"), desc="Mean pooling by user"):
            vecs = np.vstack(group["embedding"].values)
            mean_v = np.mean(vecs, axis=0)
            mean_v = mean_v / np.linalg.norm(mean_v)
            mean_vectors[n_id] = mean_v
            
            # Save to user dir
            with open(user_dir / f"n_{n_id}.json", "w") as f:
                json.dump({
                    "n_id": n_id,
                    "model": model,
                    "embeddings": {
                        "abstracts_mean": mean_v.tolist()
                    }
                }, f)
                
        # 3. N x N matrix
        print("Computing N x N Researcher Similarity Matrix...")
        n_ids = sorted(mean_vectors.keys(), key=lambda x: int(x) if x.isdigit() else x)
        X = np.vstack([mean_vectors[i] for i in n_ids])
        S = cosine_similarity(X)
        df_S = pd.DataFrame(S, index=n_ids, columns=n_ids)
        df_S.to_csv(out_matrix)
        print(f"Saved N x N matrix to {out_matrix.name}")
        
        # 4. User to other publications
        print("Computing User-to-Other-Publications Similarity...")
        rows = []
        for query_n_id, qvec in tqdm(mean_vectors.items(), desc=f"User-to-pub"):
            others = df_pubs[df_pubs["n_id"] != query_n_id]
            if others.empty:
                continue
                
            X_pubs = np.vstack(others["embedding"].values)
            sims = cosine_similarity(qvec.reshape(1, -1), X_pubs).flatten()
            
            for (_, row), sim in zip(others.iterrows(), sims):
                rows.append({
                    "query_n_id": query_n_id,
                    "target_n_id": row["n_id"],
                    "target_article_id": row["article_id"],
                    "similarity": float(sim)
                })
                
        df_sims = pd.DataFrame(rows)
        df_sims.to_csv(out_sims, index=False)
        print(f"Saved user-to-other-pub similarities to {out_sims.name} ({len(df_sims):,} rows)")

if __name__ == "__main__":
    main()
