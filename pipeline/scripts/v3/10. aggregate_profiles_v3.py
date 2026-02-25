import json
import pandas as pd
import os
import sys
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
BASE_DIR = str(_PIPELINE_ROOT / "1. data/1. raw/0. profiles")
INPUT_JSON = os.path.join(BASE_DIR, "v3", "profiles_parsed.json")
OUTPUT_CSV = os.path.join(BASE_DIR, "v3", "profiles_summary.csv")
STAFF_CSV = os.path.join(BASE_DIR, "..", "staff_cleaned.csv") 

# Target Filters
TARGET_TYPES = [
    "Book", 
    "Book Chapter", 
    "Journal", 
    "Journal Article", 
    "Working Paper"
]

def pub_year_is_2019_plus(pub):
    """
    Check if publication year is >= 2019.
    Reuses logic from original script, but handles 'Year Descending' which might be str.
    """
    # Prefer 'Year Descending' as it's cleaner in V3 usually
    year = pub.get("Year Descending") or pub.get("Year") or ""
    year = str(year).strip()
    
    # Simple digit check
    if year.isdigit():
        return int(year) >= 2019
    return False

def count_v3_types(pubs):
    """
    Count publications matching TARGET_TYPES.
    Returns:
        - counts (dict): Keyed by Type and Type_2019
        - total_filtered (int)
        - total_filtered_2019 (int)
    """
    counts = {t: 0 for t in TARGET_TYPES}
    counts_2019 = {f"{t}_2019": 0 for t in TARGET_TYPES}
    
    total_filtered = 0
    total_filtered_2019 = 0
    
    for pub in pubs:
        if not isinstance(pub, dict):
            continue
            
        ptype = pub.get("Publication Type")
        if not ptype or ptype not in TARGET_TYPES:
            continue
            
        # It matches a target type
        counts[ptype] += 1
        total_filtered += 1
        
        # Check 2019+
        if pub_year_is_2019_plus(pub):
            counts_2019[f"{ptype}_2019"] += 1
            total_filtered_2019 += 1
            
    return counts, counts_2019, total_filtered, total_filtered_2019

def main():
    print(f"Loading JSON from {INPUT_JSON}...")
    try:
        with open(INPUT_JSON, 'r') as f:
            profiles = json.load(f)
    except Exception as e:
        print(f"Error loading JSON: {e}")
        return
        
    print(f"Loaded {len(profiles)} profiles.")
    
    summary_rows = []
    
    for profile in profiles:
        # Extract Metadata
        # We assume metadata is already in JSON as per previous checks
        row = {
            "n_id": profile.get("n_id"),
            "firstname": profile.get("firstname"),
            "lastname": profile.get("lastname"),
            "department": profile.get("department"),
            "school": profile.get("school"),
            "role": profile.get("role"),
            "email": profile.get("email"),
            "biography": profile.get("biography"),
            "research_interests": profile.get("research_interests"),
            "n_publications": len(profile.get("publications", []) or [])
        }
        
        # Calculate Counts
        pubs = profile.get("publications", []) or []
        counts, counts_2019, total, total_2019 = count_v3_types(pubs)
        
        # Add Aggregates
        row["total_filtered_pubs"] = total
        row["total_filtered_pubs_2019"] = total_2019
        
        # Add Specific Counts
        row.update(counts)
        row.update(counts_2019)
        
        summary_rows.append(row)
        
    # Create DataFrame
    df = pd.DataFrame(summary_rows)
    
    # Ensure Column Order
    # Metadata first
    meta_cols = ["n_id", "firstname", "lastname", "department", "school", "role", "email", "biography", "research_interests", "n_publications"]
    
    # Aggregates next
    agg_cols = ["total_filtered_pubs", "total_filtered_pubs_2019"]
    
    # Types next (sorted)
    type_cols = sorted(list(counts.keys()))
    type_cols_2019 = sorted(list(counts_2019.keys()))
    
    # Combine
    final_cols = meta_cols + agg_cols + type_cols + type_cols_2019
    
    # Reorder if columns exist
    existing_cols = [c for c in final_cols if c in df.columns]
    df = df[existing_cols]
    
    print(f"Saving summary to {OUTPUT_CSV}...")
    df.to_csv(OUTPUT_CSV, index=False)
    print("Done.")

if __name__ == "__main__":
    main()
