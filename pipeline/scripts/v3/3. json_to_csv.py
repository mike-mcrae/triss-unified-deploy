import json
import csv
import os
import sys
import pandas as pd
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

# Files
BASE_DIR = str(_PIPELINE_ROOT / "1. data/1. raw/0. profiles")
JSON_FILE = os.path.join(BASE_DIR, "v3", "profiles_parsed.json")
SUMMARY_FILE = os.path.join(BASE_DIR, "profiles_summary.csv")
OUTPUT_CSV = os.path.join(BASE_DIR, "v3", "0.all_listed_pubs.csv")

def main():
    # 1. Load Summary for Lookup
    print(f"Loading summary from {SUMMARY_FILE}...")
    try:
        summary_df = pd.read_csv(SUMMARY_FILE, low_memory=False)
    except Exception as e:
        print(f"Error loading summary file: {e}")
        return

    # Create a lookup dictionary keyed by email
    # distinct emails?
    # We want a dict where Key=Email, Value={n_id, firstname, lastname, department, school, ...}
    # Let's convert to records
    summary_records = summary_df.to_dict('records')
    email_lookup = {}
    
    for row in summary_records:
        email = str(row.get('email')).strip().lower() # Normalize email
        if email and email != 'nan':
            email_lookup[email] = row
            
    print(f"Loaded {len(email_lookup)} profiles from summary.")

    # 2. Load JSON
    print(f"Loading JSON from {JSON_FILE}...")
    try:
        with open(JSON_FILE, 'r') as f:
            profiles_data = json.load(f)
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        return
        
    print(f"Loaded {len(profiles_data)} profiles from JSON.")

    # 3. Flatten and Merge
    all_rows = []
    
    # Define columns to extract from summary (metadata)
    # n_id,firstname,lastname,department,school,role,email,biography,research_interests
    # n_id,firstname,lastname,department,school,role,email,biography,research_interests
    meta_cols = ['n_id', 'firstname', 'lastname', 'department', 'school', 'role', 'email', 'article_id']
    
    total_pubs = 0
    matched_profiles = 0
    
    for profile in profiles_data:
        email_key = str(profile.get('email')).strip().lower()
        
        # Look up metadata
        metadata = email_lookup.get(email_key)
        
        if not metadata:
            # Fallback: maintain email from JSON, others blank
            # warning?
            # print(f"Warning: No summary metadata found for {email_key}")
            metadata = {k: None for k in meta_cols if k != 'email'}
            metadata['email'] = profile.get('email')
        else:
            matched_profiles += 1
            
        # Get publications
        pubs = profile.get('publications', [])
        if not pubs:
            pubs = []
            
        # Initialize pub counter for this profile
        pub_counter = 1
        
        for pub in pubs:
            row = {}
            # Add metadata
            for col in meta_cols:
                row[col] = metadata.get(col)
                
            # Add Article ID
            # "For n_id 1 their first paper is 1_1 their second paper is 1_2..."
            n_id = metadata.get('n_id')
            if n_id is not None:
                row['article_id'] = f"{n_id}_{pub_counter}"
            else:
                # Fallback if n_id missing (shouldn't happen with valid profiles)
                row['article_id'] = f"unknown_{pub_counter}"
            
            pub_counter += 1
            
            # Add publication data
            # Just dump all keys from pub object
            for k, v in pub.items():
                if k not in row: # Don't overwrite metadata if collision (unlikely)
                    row[k] = v
            
            all_rows.append(row)
            total_pubs += 1

    print(f"Matched {matched_profiles} profiles with metadata.")
    print(f"Total publications extracted: {total_pubs}")
    
    # 4. Write to CSV
    if not all_rows:
        print("No data to write.")
        return
        
    # Determine all fieldnames
    # Collect all keys from all rows to ensure we get everything
    # (But for CSV DictWriter, we need a fixed list. Let's get the union of keys)
    # To keep order nice: meta_cols first, then others
    all_keys = set()
    for r in all_rows:
        all_keys.update(r.keys())
        
    # Sort keys: meta_cols first, then the rest sorted alphabetically or explicitly
    # User requested: "n_id first name last name email department school etc and then all of the details about the publication"
    
    # Explicit order for Pub columns if we know them
    # "Peer Reviewed": "Y", "Year Descending": "2024", "Publication Reference": ..., "Publication Type", "Publication Source", "Senior Author", "Record Status", "Pers Pub Role", "Full Journal Name"
    # Let's verify common keys
    
    fieldnames = list(meta_cols)
    remaining = [k for k in all_keys if k not in meta_cols]
    fieldnames.extend(sorted(remaining))
    
    print(f"Writing to {OUTPUT_CSV}...")
    try:
        with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)
            
        print("Done.")
    except Exception as e:
        print(f"Error writing CSV: {e}")

if __name__ == "__main__":
    main()
