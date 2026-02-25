import pandas as pd
import re
import difflib
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

BASE_DIR = str(_PIPELINE_ROOT / "1. data")
IN_DIR = f"{BASE_DIR}/1. raw/0. profiles/v3"
OUT_DIR = f"{BASE_DIR}/3. Final"

FILE_2E = f"{IN_DIR}/2e.all_listed_publications_2019_books_other_abstracts.csv"
FILE_2F = f"{IN_DIR}/2f.all_listed_publications_2019_journal_like_with_abstract.csv"

FILE_MASTER = f"{IN_DIR}/0.all_listed_pubs_fully_enriched.csv"
FILE_REMOVED = f"{IN_DIR}/2f.removed_rows_not_jas.csv"

FILE_OUT_MEASURED = f"{OUT_DIR}/4.All measured publications.csv"
FILE_OUT_MASTER = f"{OUT_DIR}/2. All listed publications.csv"

def fuzz_ratio(s1, s2):
    return difflib.SequenceMatcher(None, s1, s2).ratio() * 100.0

def clean_abstract(a):
    if pd.isna(a):
        return a
    a = str(a)
    a = re.sub(r'<[^>]*>', '', a)
    a = re.sub(r'[\r\n\t\s]+', ' ', a)
    a = a.strip()
    
    while True:
        m = re.match(r'(?i)^(\[paraphrased\]:|\*\*abstract\*\*|abstract:|abstract)(.*)', a)
        if m:
            a = m.group(2).strip()
        else:
            break
    return a

def norm_title(t):
    if pd.isna(t):
        return ""
    t = str(t).lower()
    t = re.sub(r'[^\w\s]', '', t)
    t = re.sub(r'\s+', ' ', t).strip()
    return t

def safe_read_csv(filepath):
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        return pd.read_csv(f, low_memory=False)

def step1():
    print("--- STEP 1 ---")
    df_2e = safe_read_csv(FILE_2E)
    df_2e['source_file'] = '2e.all_listed_publications_2019_books_other_abstracts.csv'
    df_2f = safe_read_csv(FILE_2F)
    df_2f['source_file'] = '2f.all_listed_publications_2019_journal_like_with_abstract.csv'

    df_combined = pd.concat([df_2e, df_2f], ignore_index=True)
    rows_before = len(df_combined)
    print(f"Initial concatenated rows: {rows_before}")

    df_combined['abstract'] = df_combined['abstract'].apply(clean_abstract)
    df_combined['title_norm'] = df_combined['Title'].apply(norm_title)
    df_combined['has_abstract'] = df_combined['abstract'].apply(lambda x: 1 if pd.notna(x) and len(str(x).strip()) > 0 else 0)

    print("Deduplicating...")
    duplicate_groups_count = 0
    drop_indices = set()

    for n_id, group in df_combined.groupby('n_id'):
        if len(group) == 1:
            continue
        
        indices = group.index.tolist()
        titles = group['title_norm'].tolist()
        has_abstracts = group['has_abstract'].tolist()
        
        components = []
        for k, idx in enumerate(indices):
            t1 = titles[k]
            h1 = has_abstracts[k]
            matched = False
            for comp in components:
                first_idx = comp[0][0]
                pos = indices.index(first_idx)
                t_first = titles[pos]
                
                if fuzz_ratio(t1, t_first) >= 95.0:
                    comp.append((idx, k, h1))
                    matched = True
                    break
            if not matched:
                components.append([(idx, k, h1)])
                
        for comp in components:
            if len(comp) > 1:
                duplicate_groups_count += 1
                comp.sort(key=lambda x: (-x[2], x[1]))
                for item in comp[1:]:
                    drop_indices.add(item[0])

    final_df = df_combined.drop(index=list(drop_indices))
    print(f"Number of rows before deduplication: {rows_before}")
    print(f"Number of duplicate groups: {duplicate_groups_count}")
    print(f"Number of rows dropped: {len(drop_indices)}")
    print(f"Final rows step 1: {len(final_df)}")

    os.makedirs(OUT_DIR, exist_ok=True)
    final_df.to_csv(FILE_OUT_MEASURED, index=False, encoding='utf-8')

def step2():
    print("\n--- STEP 2 ---")
    df_master = safe_read_csv(FILE_MASTER)
    df_removed = safe_read_csv(FILE_REMOVED)

    df_master['title_norm'] = df_master['Title'].apply(norm_title)
    df_removed['title_norm'] = df_removed['Title'].apply(norm_title)

    matched_count = 0
    df_removed_grouped = df_removed.groupby('n_id')

    for idx, row in df_master.iterrows():
        nid = row['n_id']
        if pd.isna(nid):
            continue
        if nid in df_removed_grouped.groups:
            rem_group = df_removed_grouped.get_group(nid)
            t_master = row['title_norm']
            for _, r_row in rem_group.iterrows():
                if fuzz_ratio(t_master, r_row['title_norm']) >= 95.0:
                    df_master.at[idx, 'Publication Type'] = r_row['Publication Type']
                    matched_count += 1
                    break

    print(f"Rows matched and updated from removed: {matched_count}")
    df_master.drop(columns=['title_norm'], inplace=True, errors='ignore')
    df_master.to_csv(FILE_OUT_MASTER, index=False, encoding='utf-8')

def step3():
    print("\n--- STEP 3 ---")
    df_master = safe_read_csv(FILE_OUT_MASTER)
    df_measured = safe_read_csv(FILE_OUT_MEASURED)

    df_master['title_norm'] = df_master['Title'].apply(norm_title)
    df_measured['title_norm'] = df_measured['Title'].apply(norm_title)
    df_master['sample_articles_chosen'] = 0

    df_measured_grouped = df_measured.groupby('n_id')
    flagged_count = 0
    total_master = len(df_master)

    for idx, row in df_master.iterrows():
        nid = row['n_id']
        if pd.isna(nid):
            continue
        if nid in df_measured_grouped.groups:
            meas_group = df_measured_grouped.get_group(nid)
            t_master = row['title_norm']
            for _, m_row in meas_group.iterrows():
                if fuzz_ratio(t_master, m_row['title_norm']) >= 95.0:
                    df_master.at[idx, 'sample_articles_chosen'] = 1
                    flagged_count += 1
                    break

    print(f"Total publications in master: {total_master}")
    print(f"Number flagged as sample_articles_chosen == 1: {flagged_count}")
    print(f"% string representation: {flagged_count/total_master*100:.2f}%")

    df_master.drop(columns=['title_norm'], inplace=True, errors='ignore')
    df_master.to_csv(FILE_OUT_MASTER, index=False, encoding='utf-8')

def step4():
    print("\n--- STEP 4 ---")
    df_master = safe_read_csv(FILE_OUT_MASTER)
    
    # Filter for year >= 2019. Note that Year Descending might have missing values.
    # Convert to numeric, coercing errors to NaN.
    df_master['Year Descending numeric'] = pd.to_numeric(df_master['Year Descending'], errors='coerce')
    
    df_filtered = df_master[df_master['Year Descending numeric'] >= 2019].copy()
    df_filtered.drop(columns=['Year Descending numeric'], inplace=True, errors='ignore')
    
    file_out_2019 = f"{OUT_DIR}/3. All listed publications 2019 +.csv"
    df_filtered.to_csv(file_out_2019, index=False, encoding='utf-8')
    print(f"Total rows in master: {len(df_master)}")
    print(f"Total rows filtered (Year >= 2019): {len(df_filtered)}")
    print(f"Saved to: {file_out_2019}")

if __name__ == '__main__':
    # step1()
    # step2()
    # step3()
    step4()
