import pandas as pd
import json
import datetime
from collections import defaultdict
import os
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

BASE_DIR = str(_PIPELINE_ROOT / "1. data/3. Final")

FILES = {
    "all": f"{BASE_DIR}/2. All listed publications.csv",
    "2019_plus": f"{BASE_DIR}/3. All listed publications 2019 +.csv",
    "measured": f"{BASE_DIR}/4.All measured publications.csv"
}

def safe_read_csv(filepath):
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        return pd.read_csv(f, low_memory=False)

def get_canonical_type(pub_type):
    mapping = {
        "Book": "book",
        "Book Chapter": "book_chapter",
        "Journal Article": "journal_article",
        "Working Paper": "working_paper",
        "Other": "other"
    }
    # Assume strings, return 'other' for unfound
    if pd.isna(pub_type):
        return "other"
    
    canonical = mapping.get(str(pub_type).strip(), "other")
    return canonical

def process_file(coverage_name, filepath):
    print(f"\n--- Processing {coverage_name} ({filepath}) ---")
    df = safe_read_csv(filepath)
    
    # Required columns: n_id, department, school, Publication Type, sample_articles_chosen
    # file 4 missing sample_articles_chosen, default to 1 as specified "measured is all in sample"
    if 'sample_articles_chosen' not in df.columns:
        if coverage_name == "measured":
             df['sample_articles_chosen'] = 1
        else:
             raise ValueError(f"Missing sample_articles_chosen in {filepath}")
             
    # Ensure types
    df['n_id'] = pd.to_numeric(df['n_id'], errors='coerce')
    # Filter out invalid n_id
    df = df.dropna(subset=['n_id'])
    df['n_id'] = df['n_id'].astype(int).astype(str)
    
    df['department'] = df['department'].fillna("Unknown")
    df['school'] = df['school'].fillna("Unknown")
    df['sample_articles_chosen'] = pd.to_numeric(df['sample_articles_chosen'], errors='coerce').fillna(0).astype(int)
    
    # Structure definition
    def default_type_dict():
        return {
            "total": 0,
            "in_sample": 0,
            "share_in_sample": 0.0
        }
    
    n_id_data = defaultdict(lambda: {
        "department": "Unknown",
        "school": "Unknown",
        "total_publications": 0,
        "total_in_sample": 0,
        "share_in_sample": 0.0,
        "by_type": {} # { canonical_key: {label, total, in_sample, share} }
    })
    
    dept_data = defaultdict(lambda: {
        "school": "Unknown",
        "total_authors_set": set(),
        "total_publications": 0,
        "total_in_sample": 0,
        "share_in_sample": 0.0,
        "by_type": {}
    })
    
    school_data = defaultdict(lambda: {
        "total_departments_set": set(),
        "total_authors_set": set(),
        "total_publications": 0,
        "total_in_sample": 0,
        "share_in_sample": 0.0,
        "by_type": {}
    })

    # Used to track labels cleanly
    canonical_labels = {}
    
    for _, row in df.iterrows():
        nid = row['n_id']
        dept = row['department']
        school = row['school']
        raw_type = row['Publication Type']
        c_type = get_canonical_type(raw_type)
        if pd.isna(raw_type) or str(raw_type).strip() == "":
            raw_type = "Other"
        
        canonical_labels[c_type] = str(raw_type).strip() if c_type != "other" else "Other"
        in_samp = 1 if row['sample_articles_chosen'] == 1 else 0
        
        # --- UPDATE n_id ---
        n_id_data[nid]["department"] = dept
        n_id_data[nid]["school"] = school
        n_id_data[nid]["total_publications"] += 1
        n_id_data[nid]["total_in_sample"] += in_samp
        if c_type not in n_id_data[nid]["by_type"]:
            n_id_data[nid]["by_type"][c_type] = default_type_dict()
        n_id_data[nid]["by_type"][c_type]["total"] += 1
        n_id_data[nid]["by_type"][c_type]["in_sample"] += in_samp
        
        # --- UPDATE dept ---
        dept_data[dept]["school"] = school
        dept_data[dept]["total_authors_set"].add(nid)
        dept_data[dept]["total_publications"] += 1
        dept_data[dept]["total_in_sample"] += in_samp
        if c_type not in dept_data[dept]["by_type"]:
            dept_data[dept]["by_type"][c_type] = default_type_dict()
        dept_data[dept]["by_type"][c_type]["total"] += 1
        dept_data[dept]["by_type"][c_type]["in_sample"] += in_samp
        
        # --- UPDATE school ---
        school_data[school]["total_departments_set"].add(dept)
        school_data[school]["total_authors_set"].add(nid)
        school_data[school]["total_publications"] += 1
        school_data[school]["total_in_sample"] += in_samp
        if c_type not in school_data[school]["by_type"]:
            school_data[school]["by_type"][c_type] = default_type_dict()
        school_data[school]["by_type"][c_type]["total"] += 1
        school_data[school]["by_type"][c_type]["in_sample"] += in_samp

    # Clean up sets and compute shares
    # n_id
    total_pubs_from_nid = 0
    for nid, data in n_id_data.items():
        data["share_in_sample"] = round(data["total_in_sample"] / data["total_publications"], 4) if data["total_publications"] > 0 else 0.0
        total_pubs_from_nid += data["total_publications"]
        for ckey, tdata in data["by_type"].items():
            tdata["label"] = canonical_labels.get(ckey, "Other")
            tdata["share_in_sample"] = round(tdata["in_sample"] / tdata["total"], 4) if tdata["total"] > 0 else 0.0
            
    # dept
    total_pubs_from_dept = 0
    for dept, data in dept_data.items():
        data["total_authors"] = len(data.pop("total_authors_set"))
        data["share_in_sample"] = round(data["total_in_sample"] / data["total_publications"], 4) if data["total_publications"] > 0 else 0.0
        total_pubs_from_dept += data["total_publications"]
        for ckey, tdata in data["by_type"].items():
            tdata["label"] = canonical_labels.get(ckey, "Other")
            tdata["share_in_sample"] = round(tdata["in_sample"] / tdata["total"], 4) if tdata["total"] > 0 else 0.0

    # school
    total_pubs_from_school = 0
    for school, data in school_data.items():
        data["total_departments"] = len(data.pop("total_departments_set"))
        data["total_authors"] = len(data.pop("total_authors_set"))
        data["share_in_sample"] = round(data["total_in_sample"] / data["total_publications"], 4) if data["total_publications"] > 0 else 0.0
        total_pubs_from_school += data["total_publications"]
        for ckey, tdata in data["by_type"].items():
            tdata["label"] = canonical_labels.get(ckey, "Other")
            tdata["share_in_sample"] = round(tdata["in_sample"] / tdata["total"], 4) if tdata["total"] > 0 else 0.0

    # VALIDATION
    assert total_pubs_from_nid == total_pubs_from_dept, f"Mismatch pubs: {total_pubs_from_nid} vs {total_pubs_from_dept}"
    assert total_pubs_from_dept == total_pubs_from_school, f"Mismatch pubs: {total_pubs_from_dept} vs {total_pubs_from_school}"

    print(f"Validation passed for {coverage_name}.")
    print(f"Total Publications computed: {total_pubs_from_school}")
    print(f"n_ids: {len(n_id_data)}")
    print(f"departments: {len(dept_data)}")
    print(f"schools: {len(school_data)}")

    output = {
        "metadata": {
            "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
            "coverage": coverage_name,
            "units": ["n_id", "department", "school"]
        },
        "n_id": dict(n_id_data),
        "department": dict(dept_data),
        "school": dict(school_data)
    }

    outfile = f"{BASE_DIR}/counts_{coverage_name}.json"
    with open(outfile, "w", encoding='utf-8') as f:
        json.dump(output, f, indent=2)
    print(f"Saved: {outfile}")

if __name__ == "__main__":
    for coverage, file_path in FILES.items():
        process_file(coverage, file_path)
