#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate embeddings for TRISS measured publication abstracts.
"""

import argparse
import json
import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm

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

# -----------------------------
# CONFIG (paths)
# -----------------------------
PUBS_CSV = Path(
    str(_PIPELINE_ROOT / "1. data/3. Final/4.All measured publications.csv")
)

EMBED_ROOT = Path(
    str(_PIPELINE_ROOT / "1. data/4. embeddings/v3")
)

# -----------------------------
# TEXT HELPERS
# -----------------------------
def clean_text(x) -> str:
    if pd.isna(x):
        return ""
    return " ".join(str(x).split())

def safe_read_csv(filepath):
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        return pd.read_csv(f, low_memory=False)

# -----------------------------
# MAIN
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["openai", "mpnet"])
    parser.add_argument("--start_idx", type=int, required=True)
    parser.add_argument("--end_idx", type=int, required=True)
    args = parser.parse_args()

    # Load data
    pubs = safe_read_csv(PUBS_CSV)
    subset = pubs.iloc[args.start_idx : args.end_idx]

    # Model init
    if args.model == "openai":
        from openai import OpenAI
        client = OpenAI()
        model_name = "text-embedding-3-large"
        dim = 3072
        out_dir = EMBED_ROOT / "openai" / "by_publication"
    else:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
        model_name = "all-mpnet-base-v2"
        dim = 768
        out_dir = EMBED_ROOT / "mpnet" / "by_publication"

    out_dir.mkdir(parents=True, exist_ok=True)

    for idx, row in tqdm(subset.iterrows(), total=len(subset)):
        n_id = row["n_id"] if pd.notna(row["n_id"]) else "unknown"
        article_id = row.get("article_id", "unknown")
        if pd.isna(article_id) or article_id == "unknown":
            article_id = str(idx)
            
        # Clean article id to make safe file name
        safe_article_id = str(article_id).replace("/", "_").replace("\\", "_")

        out_path = out_dir / f"article_{safe_article_id}.json"

        # Skip if already done
        if out_path.exists():
            continue

        abs_text = clean_text(row.get("abstract", ""))
        if not abs_text:
            continue

        if args.model == "openai":
            try:
                emb_abstract = client.embeddings.create(
                    model=model_name,
                    input=abs_text
                ).data[0].embedding
            except Exception as e:
                print(f"Failed openai embedding for {article_id}: {e}")
                continue
        else:
            emb_abstract = model.encode(
                abs_text,
                normalize_embeddings=True
            ).tolist()

        result = {
            "n_id": n_id,
            "article_id": article_id,
            "model": model_name,
            "embedding_dim": dim,
            "embeddings": {
                "abstract": emb_abstract
            }
        }

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f)

    print(f"Finished chunk {args.start_idx} to {args.end_idx}")

if __name__ == "__main__":
    main()
