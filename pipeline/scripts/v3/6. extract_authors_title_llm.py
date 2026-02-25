import pandas as pd
import os
import json
import time
import sys
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
BATCH_SIZE = 50 # Increased slightly for efficiency
MODEL = "gpt-4o-mini"
CHECKPOINT_INTERVAL = 10 # Save CSV every 10 batches

# Files
BASE_DIR = str(_PIPELINE_ROOT / "1. data/1. raw/0. profiles/v3")
INPUT_CSV = os.path.join(BASE_DIR, "0.all_listed_pubs_enriched.csv")
OUTPUT_CSV = os.path.join(BASE_DIR, "0.all_listed_pubs_fully_enriched.csv")
LOG_FILE = os.path.join(BASE_DIR, "0.all_listed_pubs_llm_log.jsonl")

def setup_client():
    if not API_KEY:
        raise ValueError("OPENAI_API_KEY environment variable not set.")
    return OpenAI(api_key=API_KEY)

def extract_batch(client, refs_dict):
    """
    refs_dict: {index: reference_string}
    Returns: {index: {authors, title, confidence, justification}}
    """
    if not refs_dict:
        return {}
        
    # Prepare prompt (PRESERVED LOGIC)
    refs_text = ""
    for idx, ref in refs_dict.items():
        refs_text += f"ID_{idx}: {ref}\n"
        
    prompt = f"""
    You are a bibliographic parsing expert. Extract the Authors and Title from the following {len(refs_dict)} references.
    
    References:
    {refs_text}
    
    Return a strictly valid JSON object where keys are the IDs (e.g. "ID_123") and values are objects with:
    - "authors": string (exact author list) or null
    - "title": string (title only) or null
    - "confidence": float (0.0 to 1.0)
    - "justification": short string
    
    Example Output Format:
    {{
      "ID_1": {{ "authors": "Smith, J.", "title": "Study of X", ... }},
      "ID_2": {{ ... }}
    }}
    
    Do NOT return markdown formatting. Just raw JSON.
    """
    
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that outputs JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
            max_tokens=4000 
        )
        content = response.choices[0].message.content.strip()
        data = json.loads(content)
        
        # Remap ID_X back to index
        results = {}
        for key, val in data.items():
            if key.startswith("ID_"):
                try:
                    idx = int(key.split("_")[1])
                    results[idx] = val
                except:
                    pass
        return results
    except Exception as e:
        print(f"Batch Error: {e}")
        return {}

def load_processed_indices():
    """Load indices that have already been processed from the JSONL log."""
    processed = set()
    if os.path.exists(LOG_FILE):
        print(f"Loading progress from {LOG_FILE}...")
        try:
            with open(LOG_FILE, 'r') as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        if 'index' in entry:
                            processed.add(entry['index'])
                        elif 'batch_results' in entry: # Handle batch log if used
                             for k in entry['batch_results'].keys():
                                 processed.add(int(k))
                    except:
                        pass
        except Exception as e:
            print(f"Error loading log: {e}")
    return processed

def save_to_log(results_dict):
    """Append batch results to JSONL log."""
    with open(LOG_FILE, 'a') as f:
        # We can enable writing individual rows or the whole batch
        # Writing individual rows makes 'load_processed_indices' simpler
        for idx, res in results_dict.items():
            entry = {
                "index": idx,
                "timestamp": time.time(),
                "result": res
            }
            f.write(json.dumps(entry) + "\n")

def main():
    # Setup
    try:
        client = setup_client()
    except ValueError as e:
        print(f"Error: {e}")
        return

    # Load Data
    # If OUTPUT_CSV exists, load it to preserve previous work not found in log? 
    # Actually, relying on INPUT + LOG is safer for consistency. 
    # But ideally we start from the PARTIAL output if available to keep columns.
    
    if os.path.exists(OUTPUT_CSV):
        print(f"Loading existing output file {OUTPUT_CSV}...")
        df = pd.read_csv(OUTPUT_CSV, low_memory=False)
    else:
        print(f"Loading input file {INPUT_CSV}...")
        df = pd.read_csv(INPUT_CSV, low_memory=False)

    ref_col = 'Publication Reference'
    if ref_col not in df.columns:
        print(f"Column '{ref_col}' not found.")
        return

    # Ensure columns exist
    for col in ['Authors', 'Title', 'llm_authors_title_meta']:
        if col not in df.columns:
            df[col] = None

    # Identify rows to process
    # 1. Must have reference
    mask_has_ref = df[ref_col].notna() & (df[ref_col].astype(str).str.strip() != "")
    
    # 2. Must NOT be processed (check Log and/or non-null Title)
    processed_indices = load_processed_indices()
    print(f"Found {len(processed_indices)} already processed rows in log.")
    
    # Also check if DataFrame already has data (in case log is missing but file is filled)
    if 'Title' in df.columns:
        mask_already_done = df['Title'].notna() & (df['Title'].astype(str).str.strip() != "")
        logical_processed = set(df[mask_already_done].index)
        processed_indices.update(logical_processed) 
    
    all_indices = df[mask_has_ref].index.tolist()
    target_indices = [idx for idx in all_indices if idx not in processed_indices]
    
    total_to_process = len(target_indices)
    print(f"Total references: {len(all_indices)}")
    print(f"Remaining to process: {total_to_process}")
    
    if total_to_process == 0:
        print("All rows processed. Done.")
        return

    print(f"Starting Extraction with {MODEL} (Batch Size: {BATCH_SIZE})...")
    
    start_time = time.time()
    processed_count = 0
    
    # Batch processing loop
    for i in range(0, total_to_process, BATCH_SIZE):
        batch_indices = target_indices[i : i + BATCH_SIZE]
        batch_refs = {idx: df.at[idx, ref_col] for idx in batch_indices}
        
        # Estimate time
        elapsed = time.time() - start_time
        rate = processed_count / elapsed if elapsed > 0 else 0
        eta = (total_to_process - processed_count) / rate / 60 if rate > 0 else 0
        
        print(f"Processing batch {(i // BATCH_SIZE) + 1} / {(total_to_process // BATCH_SIZE) + 1} | Rows: {processed_count}/{total_to_process} | ETA: {eta:.1f} min")
        
        # LLM Call
        try:
            results = extract_batch(client, batch_refs)
        except Exception as e:
             print(f"Unexpected error in batch: {e}")
             time.sleep(5)
             continue
        
        # Process Results
        if results:
            # 1. Write to Log (Safe Write)
            save_to_log(results)
            
            # 2. Update DataFrame
            for idx in batch_indices:
                res = results.get(idx)
                if res:
                    df.at[idx, 'Authors'] = res.get('authors')
                    df.at[idx, 'Title'] = res.get('title')
                    df.at[idx, 'llm_authors_title_meta'] = json.dumps({
                        'confidence': res.get('confidence'),
                        'justification': res.get('justification')
                    })
        
        processed_count += len(batch_indices)
        
        # 3. Checkpoint CSV
        if (i // BATCH_SIZE) % CHECKPOINT_INTERVAL == 0:
             df.to_csv(OUTPUT_CSV, index=False)
             # print(f"Checkpoint saved to {OUTPUT_CSV}")
             
        time.sleep(0.5) 

    print("-" * 30)
    print("Done processing.")
    
    # Final Save and Column Reordering
    print(f"Saving final output to {OUTPUT_CSV}...")
    
    # Ensure columns are ordered: Ref, Authors, Title, ...
    cols = df.columns.tolist()
    if 'Authors' in cols and 'Title' in cols:
        try:
            # Drop from current position
            cols = [c for c in cols if c not in ['Authors', 'Title']]
            # Insert after reference
            ref_idx = cols.index(ref_col)
            cols.insert(ref_idx + 1, 'Authors')
            cols.insert(ref_idx + 2, 'Title')
            df = df[cols]
        except ValueError:
            pass

    df.to_csv(OUTPUT_CSV, index=False)
    print("Script completed successfully.")

if __name__ == "__main__":
    main()
