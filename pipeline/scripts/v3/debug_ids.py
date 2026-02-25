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
MASTER_CSV = os.path.join(BASE_DIR, "0.all_listed_pubs_fully_enriched.csv")
SOURCE_CSV = os.path.join(BASE_DIR, "2c.all_listed_publications_2019_filter.csv")
TARGET_CSV = os.path.join(BASE_DIR, "2c.all_listed_publications_2019_journal_like_doi.csv")

def main():
    print("Loading files...")
    df_m = pd.read_csv(MASTER_CSV, low_memory=False)
    df_s = pd.read_csv(SOURCE_CSV, low_memory=False)
    df_t = pd.read_csv(TARGET_CSV, low_memory=False)
    
    # Filter for n_id 74
    m_ROWS = df_m[df_m['n_id'] == 74]
    s_ROWS = df_s[df_s['n_id'] == 74]
    t_ROWS = df_t[df_t['n_id'] == 74]
    
    print(f"\nMaster (n_id=74): {len(m_ROWS)} rows")
    for idx, row in m_ROWS.iterrows():
        print(f"  ID: {row.get('article_id')} | Title: {str(row.get('Title'))[:30]}...")

    print(f"\nSource (n_id=74): {len(s_ROWS)} rows")
    for idx, row in s_ROWS.iterrows():
        print(f"  ID: {row.get('article_id')} | Title: {str(row.get('Title'))[:30]}...")

    print(f"\nTarget (n_id=74): {len(t_ROWS)} rows")
    for idx, row in t_ROWS.iterrows():
        print(f"  ID: {row.get('article_id')} | Title: {str(row.get('Title'))[:30]}...")
        
if __name__ == "__main__":
    main()
