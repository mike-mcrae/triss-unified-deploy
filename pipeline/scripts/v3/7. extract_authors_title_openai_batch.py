import os
import json
import time
import argparse
import sys
import pandas as pd
from openai import OpenAI
from datetime import datetime, timedelta
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
MODEL = "gpt-4o-mini"
BASE_DIR = str(_PIPELINE_ROOT / "1. data/1. raw/0. profiles/v3")
SCRIPT_DIR = str(_PIPELINE_ROOT / "0. scripts/v3")
BATCH_RUNS_DIR = os.path.join(SCRIPT_DIR, "_batch_runs/authors_title")
INPUT_CSV = os.path.join(BASE_DIR, "0.all_listed_pubs_enriched.csv")
OUTPUT_CSV = os.path.join(BASE_DIR, "0.all_listed_pubs_fully_enriched.csv")

# State Files
JSONL_INPUT = os.path.join(BATCH_RUNS_DIR, "input.jsonl")
JSONL_OUTPUT = os.path.join(BATCH_RUNS_DIR, "output.jsonl")
JSONL_ERROR = os.path.join(BATCH_RUNS_DIR, "error.jsonl")
STATE_FILE = os.path.join(BATCH_RUNS_DIR, "state.json")

def get_client():
    if not API_KEY:
        raise ValueError("OPENAI_API_KEY not set.")
    return OpenAI(api_key=API_KEY)

def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_state(state):
    # Ensure dir exists
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2)

def generate_jsonl(dry_run=False):
    print(f"Loading {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV, low_memory=False)
    
    ref_col = 'Publication Reference'
    if ref_col not in df.columns:
        raise ValueError(f"Column '{ref_col}' not found.")
        
    requests = []
    
    # Identify target rows: Non-empty ref AND (missing Authors OR missing Title)
    # Check if cols exist
    cols_check = ['Authors', 'Title']
    for c in cols_check:
        if c not in df.columns:
            df[c] = None
            
    count = 0
    with open(JSONL_INPUT, 'w') as f:
        for idx, row in df.iterrows():
            ref = row.get(ref_col)
            if pd.isna(ref) or str(ref).strip() == "":
                continue
                
            # Skip if already done
            # Check if Authors/Title are present in source CSV?
            # User requirement: "Keep existing Authors/Title if already filled"
            authors_filled = pd.notna(row.get('Authors')) and str(row.get('Authors')).strip() != ""
            title_filled = pd.notna(row.get('Title')) and str(row.get('Title')).strip() != ""
            
            if authors_filled and title_filled:
                continue
                 
            # Create Custom ID
            n_id = row.get('n_id', 'unknown')
            custom_id = f"req_{n_id}_{idx}"
            
            # Construct Prompt
            prompt = f"""
            You are a bibliographic parsing expert. Extract the Authors and Title from the following reference.
            
            Reference: "{ref}"
            
            Return a strictly valid JSON object with:
            - "authors": string (exact author list) or null
            - "title": string (title only) or null
            - "confidence": float (0.0 to 1.0)
            - "justification": short string
            
            Do NOT return markdown formatting. Just raw JSON.
            """
            
            req = {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": MODEL,
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant that outputs JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.0,
                    "max_tokens": 500,
                    "response_format": {"type": "json_object"}
                }
            }
            
            f.write(json.dumps(req) + "\n")
            count += 1
            
    print(f"Generated {count} requests in {JSONL_INPUT}")
    return count

def submit_batch():
    client = get_client()
    state = load_state()
    
    # Check if active batch exists
    if state.get('batch_id') and state.get('status') not in ['completed', 'failed', 'expired', 'cancelled']:
        print(f"Batch already active: {state['batch_id']}. Skipping submission.")
        return state['batch_id']
        
    print("Uploading input file...")
    if not os.path.exists(JSONL_INPUT):
         print("Input file missing. Generating...")
         cnt = generate_jsonl()
         if cnt == 0:
             print("No requests generated. Nothing to do.")
             return None

    batch_input_file = client.files.create(
      file=open(JSONL_INPUT, "rb"),
      purpose="batch"
    )
    file_id = batch_input_file.id
    print(f"File uploaded: {file_id}")
    
    print("Creating batch job...")
    batch_job = client.batches.create(
      input_file_id=file_id,
      endpoint="/v1/chat/completions",
      completion_window="24h",
      metadata={"description": "extract_authors_title"}
    )
    
    state['batch_id'] = batch_job.id
    state['input_file_id'] = file_id
    state['status'] = 'submitted'
    state['submitted_at'] = time.time()
    save_state(state)
    
    print(f"Batch submitted! ID: {batch_job.id}")
    return batch_job.id

def poll_batch_until_completion(batch_id):
    client = get_client()
    state = load_state()
    
    print(f"Polling batch {batch_id}...")
    
    while True:
        try:
            batch = client.batches.retrieve(batch_id)
            status = batch.status
            
            # Print status update
            elapsed = time.time() - state.get('submitted_at', time.time())
            elapsed_str = str(timedelta(seconds=int(elapsed)))
            print(f"Batch {batch_id}: status={status} (elapsed {elapsed_str}) | processed: {batch.request_counts.completed}/{batch.request_counts.total}")
            
            # Update state
            state['status'] = status
            if batch.output_file_id:
                state['output_file_id'] = batch.output_file_id
            if batch.error_file_id:
                state['error_file_id'] = batch.error_file_id
            save_state(state)
            
            if status == 'completed':
                print("Batch completed successfully.")
                return True
            elif status in ['failed', 'expired', 'cancelled']:
                print(f"Batch failed with status: {status}")
                if batch.errors:
                    print(f"Errors: {batch.errors}")
                return False
                
            time.sleep(60) # Poll every 60s
            
        except Exception as e:
            print(f"Error polling: {e}")
            time.sleep(60)

def download_results(state):
    client = get_client()
    output_file_id = state.get('output_file_id')
    
    if not output_file_id:
        print("No output file ID found in state.")
        return False
        
    print(f"Downloading results from {output_file_id}...")
    content = client.files.content(output_file_id).read()
    with open(JSONL_OUTPUT, 'wb') as f:
        f.write(content)
    print(f"Results saved to {JSONL_OUTPUT}")
    return True

def merge_results():
    if not os.path.exists(JSONL_OUTPUT):
        print(f"Output file {JSONL_OUTPUT} not found.")
        return
        
    print(f"Merging results from {JSONL_OUTPUT}...")
    
    results_map = {}
    with open(JSONL_OUTPUT, 'r') as f:
        for line in f:
            try:
                res = json.loads(line)
                custom_id = res['custom_id']
                # req_nid_INDEX -> extract INDEX
                idx = int(custom_id.split('_')[-1]) 
                
                resp_body = res['response']['body']
                content = resp_body['choices'][0]['message']['content']
                extraction = json.loads(content)
                
                results_map[idx] = extraction
            except Exception as e:
                pass
                
    print(f"Loaded {len(results_map)} results.")
    
    print(f"Loading {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV, low_memory=False)
    
    # Ensure columns
    for col in ['Authors', 'Title', 'llm_authors_title_meta']:
        if col not in df.columns:
            df[col] = None
            
    # Update directly
    updated_count = 0
    for idx, extraction in results_map.items():
        if idx in df.index:
            # Only update if missing (optional safety, though input gen handled logic)
            # Just overwrite/fill based on extraction
            df.at[idx, 'Authors'] = extraction.get('authors')
            df.at[idx, 'Title'] = extraction.get('title')
            meta = {
                'confidence': extraction.get('confidence'),
                'justification': extraction.get('justification')
            }
            df.at[idx, 'llm_authors_title_meta'] = json.dumps(meta)
            updated_count += 1
            
    print(f"Updated {updated_count} rows.")
    
    # Reorder columns
    cols = df.columns.tolist()
    if 'Authors' in cols and 'Title' in cols:
         try:
             cols = [c for c in cols if c not in ['Authors', 'Title']]
             ref_idx = cols.index('Publication Reference')
             cols.insert(ref_idx + 1, 'Authors')
             cols.insert(ref_idx + 2, 'Title')
             df = df[cols]
         except: pass

    print(f"Saving final output to {OUTPUT_CSV}...")
    df.to_csv(OUTPUT_CSV, index=False)
    
    print("-" * 30)
    print("Batch processing completed successfully.")
    print(f"Rows processed: {updated_count}")
    print(f"Output written to: {OUTPUT_CSV}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true', help="Generate JSONL only")
    parser.add_argument('--submit-only', action='store_true', help="Submit batch only")
    parser.add_argument('--watch-only', action='store_true', help="Poll existing batch only")
    parser.add_argument('--merge-only', action='store_true', help="Merge existing results only")
    args = parser.parse_args()
    
    # Load state
    state = load_state()
    
    # --dry-run
    if args.dry_run:
        generate_jsonl()
        return

    # --merge-only
    if args.merge_only:
        merge_results()
        return

    # --watch-only
    if args.watch_only:
        if not state.get('batch_id'):
            print("No batch to watch.")
            return
        success = poll_batch_until_completion(state['batch_id'])
        if success:
             download_results(state)
             merge_results()
        return

    # DEFAULT / MAIN FLOW
    
    # 1. Generate Input if needed
    if not os.path.exists(JSONL_INPUT) and not state.get('batch_id'):
        count = generate_jsonl()
        if count == 0:
            print("No requests generated.")
            # If no requests, maybe check if we should merge anyway?
            return

    # 2. Submit Batch if needed
    batch_id = state.get('batch_id')
    if not batch_id or state.get('status') in ['failed', 'expired', 'cancelled']:
        batch_id = submit_batch()
        if not batch_id:
            return # Nothing submitted
        if args.submit_only:
             return
    
    # 3. Poll
    success = poll_batch_until_completion(batch_id)
    if not success:
        sys.exit(1)
        
    # 4. Download
    if not download_results(load_state()): # Reload state to get output_id
        sys.exit(1)
        
    # 5. Merge
    merge_results()

if __name__ == "__main__":
    main()
