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

# Configuration
API_KEY = os.environ.get("OPENAI_API_KEY")

# Files
BASE_DIR = str(_PIPELINE_ROOT / "1. data/1. raw/0. profiles/v3")
INPUT_CSV = os.path.join(BASE_DIR, "0.all_listed_pubs.csv")
OUTPUT_CSV = os.path.join(BASE_DIR, "0.all_listed_pubs_enriched.csv")

def setup_client():
    if not API_KEY:
        raise ValueError("OPENAI_API_KEY environment variable not set.")
    return OpenAI(api_key=API_KEY)

def extract_year(client, reference):
    prompt = f"""
    You are a bibliographic expert. Extract the publication year from the following reference.
    
    Reference: "{reference}"
    
    Return a strictly valid JSON object with:
    - "extracted_year": integer or null
    - "confidence": float (0.0 to 1.0)
    - "justification": short string explaining why (e.g. "Found '2012' at end of string")
    
    If no year is present, return null for extracted_year.
    Do NOT guess. Do NOT return markdown formatting (like ```json). Just the raw JSON string.
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo", # Cost effective
            messages=[
                {"role": "system", "content": "You are a helpful assistant that outputs JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=150
        )
        content = response.choices[0].message.content.strip()
        # Clean markdown if present
        if content.startswith("```json"):
            content = content[7:]
        if content.endswith("```"):
            content = content[:-3]
        return json.loads(content.strip())
    except Exception as e:
        # print(f"Error processing reference: {reference[:50]}... -> {e}")
        return {"extracted_year": None, "confidence": 0.0, "justification": f"Error: {str(e)}"}

def main():
    print(f"Loading {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV, low_memory=False)
    
    # Identify target rows
    year_col = 'Year Descending'
    ref_col = 'Publication Reference'
    
    if year_col not in df.columns:
        print(f"Column '{year_col}' not found. Available: {df.columns.tolist()}")
        return

    # Filter where year is '-' or NaN
    mask = (df[year_col].astype(str).str.strip() == '-') | (df[year_col].isna())
    target_indices = df[mask].index
    
    print(f"Found {len(target_indices)} rows with missing years.")
    
    if len(target_indices) == 0:
        print("No rows to process.")
        return

    # Setup Client
    try:
        client = setup_client()
    except ValueError as e:
        print(f"Setup Error: {e}")
        print("Please set OPENAI_API_KEY environment variable.")
        return

    # Process
    print("Starting LLM extraction (OpenAI)...")
    print(f"Processing {len(target_indices)} rows...")
    
    processed_count = 0
    updated_count = 0
    
    for idx in target_indices:
        row = df.loc[idx]
        ref = row.get(ref_col, "")
        
        # Skip empty references
        if not ref or pd.isna(ref) or str(ref).strip() == "":
            continue
            
        processed_count += 1
        if processed_count % 10 == 0:
            print(f"Processed {processed_count} rows...")
            
        extraction = extract_year(client, ref)
        
        year = extraction.get("extracted_year")
        conf = extraction.get("confidence")
        just = extraction.get("justification")
        
        # Update DataFrame
        if year:
            df.at[idx, year_col] = str(year)
            updated_count += 1
            
        # Log metadata
        df.at[idx, 'llm_extraction_meta'] = json.dumps(extraction)
        
        # Rate limit safety? OpenAI handles concurrency usually but sequential is fine for 135 rows
        # time.sleep(0.1) 

    print("-" * 30)
    print(f"Extraction Complete.")
    print(f"Rows Processed: {processed_count}")
    print(f"Years Updated: {updated_count}")
    print(f"Saving enriched data to {OUTPUT_CSV}...")
    df.to_csv(OUTPUT_CSV, index=False)
    print("Done.")

if __name__ == "__main__":
    main()
