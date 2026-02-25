import pandas as pd
import os
import json
import time
from openai import OpenAI
from pathlib import Path

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

# =========================
# Configuration
# =========================
API_KEY = os.environ.get("OPENAI_API_KEY")
MODEL = "gpt-4o-mini"
SLEEP_SECONDS = 0.5

BASE_DIR = str(_PIPELINE_ROOT / "1. data/1. raw/0. profiles/v3")
CSV_PATH = os.path.join(BASE_DIR, "2c.all_listed_publications_2019_filter.csv")
LOG_FILE = os.path.join(BASE_DIR, "2c.llm_author_title_fill_log.jsonl")

REF_COL = "Publication Reference"
AUTHOR_COL = "Authors"
TITLE_COL = "Title"

# =========================
# Setup
# =========================
def setup_client():
    if not API_KEY:
        raise ValueError("OPENAI_API_KEY not set")
    return OpenAI(api_key=API_KEY)

# =========================
# LLM Call (single entry)
# =========================
def extract_single(client, reference_text):
    prompt = f"""
You are a bibliographic parsing expert.

Extract the Authors and Title from the reference below.

Rules:
- Output MUST be valid JSON
- No markdown
- No line breaks inside strings
- Escape quotation marks properly
- If unsure, return null fields

Reference:
{reference_text}

Return exactly:
{{
  "authors": "...",
  "title": "...",
  "confidence": 0.0,
  "justification": "..."
}}
"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You output only valid JSON."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        response_format={"type": "json_object"},
        max_tokens=1000,
    )

    raw = response.choices[0].message.content.strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        print("❌ JSON parse error")
        print("Raw response:")
        print(raw)
        raise e

# =========================
# Logging
# =========================
def load_processed_indices():
    processed = set()
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE) as f:
            for line in f:
                try:
                    processed.add(json.loads(line)["index"])
                except:
                    pass
    return processed

def append_log(idx, result):
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps({
            "index": idx,
            "timestamp": time.time(),
            "result": result
        }) + "\n")

# =========================
# Main
# =========================
def main():
    client = setup_client()

    print(f"Loading {CSV_PATH}")
    df = pd.read_csv(CSV_PATH, low_memory=False)

    # Ensure columns exist
    for col in [AUTHOR_COL, TITLE_COL]:
        if col not in df.columns:
            df[col] = None

    processed = load_processed_indices()

    # Rows to process: blank title + has reference
    mask = (
        df[REF_COL].notna()
        & (df[REF_COL].astype(str).str.strip() != "")
        & (df[TITLE_COL].isna() | (df[TITLE_COL].astype(str).str.strip() == ""))
    )

    target_indices = [
        idx for idx in df[mask].index.tolist()
        if idx not in processed
    ]

    print(f"Rows with missing titles: {len(target_indices)}")

    if not target_indices:
        print("Nothing to do.")
        return

    for n, idx in enumerate(target_indices, 1):
        ref = df.at[idx, REF_COL]

        print(f"\n[{n}/{len(target_indices)}] Processing row {idx}")
        print(ref[:300] + ("..." if len(ref) > 300 else ""))

        try:
            result = extract_single(client, ref)
        except Exception:
            print("⚠️ Skipping row due to parse error.")
            continue

        # Update DataFrame
        df.at[idx, AUTHOR_COL] = result.get("authors")
        df.at[idx, TITLE_COL] = result.get("title")

        append_log(idx, result)

        # Save immediately (small N, safest)
        df.to_csv(CSV_PATH, index=False)

        time.sleep(SLEEP_SECONDS)

    print("\nAll done. Final save.")
    df.to_csv(CSV_PATH, index=False)

if __name__ == "__main__":
    main()
