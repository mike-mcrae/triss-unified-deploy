import pandas as pd
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

BASE_DIR = str(_PIPELINE_ROOT / "1. data/1. raw/0. profiles/v3")
MASTER_CSV = os.path.join(BASE_DIR, "0.all_listed_pubs.csv")
TARGET_CSV = os.path.join(BASE_DIR, "2c.all_listed_publications_2019_journal_like_doi.csv")

def normalize_text(s):
    if not isinstance(s, str):
        return ""
    return " ".join(s.lower().split())

def main():
    print("Loading Master...")
    df_m = pd.read_csv(MASTER_CSV, low_memory=False)
    print("Loading Target...")
    df_t = pd.read_csv(TARGET_CSV, low_memory=False)
    
    # Take first row of Target
    row_t = df_t.iloc[0]
    print("\n--- Target Row 0 ---")
    print(f"n_id: {repr(row_t.get('n_id'))}")
    print(f"Title: {repr(row_t.get('Title'))}")
    print(f"Year: {repr(row_t.get('Year Descending'))}")
    
    t_nid = str(row_t.get('n_id', '')).strip()
    t_title = normalize_text(row_t.get('Title', ''))
    
    t_year_raw = row_t.get('Year Descending', '')
    try:
        t_year = str(int(float(t_year_raw)))
    except:
        t_year = str(t_year_raw).strip()
        
    print(f"Target Key: ({repr(t_nid)}, {repr(t_title)}, {repr(t_year)})")
    
    # Search for this in Master
    # Brute force search
    print("\n--- Searching in Master ---")
    found_nid = False
    found_title = False
    
    for idx, row_m in df_m.iterrows():
        m_nid = str(row_m.get('n_id', '')).strip()
        m_title = normalize_text(row_m.get('Title', ''))
        
        m_year_raw = row_m.get('Year Descending', '')
        try:
            m_year = str(int(float(m_year_raw)))
        except:
            m_year = str(m_year_raw).strip()
            
        if m_nid == t_nid:
            found_nid = True
            if m_title == t_title:
                found_title = True
                print(f"Match found at index {idx}!")
                print(f"Master Key: ({repr(m_nid)}, {repr(m_title)}, {repr(m_year)})")
                print(f"Master Article ID: {row_m.get('article_id')}")
                if m_year != t_year:
                     print(f"MISMATCH YEAR: T={t_year} vs M={m_year}")
                return
                
    if found_nid:
        print("Found matching n_id but no matching Title.")
    else:
        print("No matching n_id found.")

if __name__ == "__main__":
    main()
