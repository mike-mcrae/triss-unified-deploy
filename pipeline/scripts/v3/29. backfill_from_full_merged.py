import pandas as pd
from openai import OpenAI
import os
import re
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
TARGET_CSV = str(_PIPELINE_ROOT / "1. data/1. raw/0. profiles/v3/2e.all_listed_publications_2019_journal_like_no_abstract copy.csv")
SOURCE_CSV = str(_SHARED_DATA_ROOT / "3. final/3. profiles_publications_full_merged.csv")
OPENAI_MODEL = "gpt-4o-mini"

def normalize_title(text):
    if not isinstance(text, str): return ""
    return re.sub(r'\W+', '', text.lower())

def verify_abstract_with_llm(client, title, candidate_abstract):
    """
    Uses LLM to verify if the candidate abstract is a valid description of the article.
    """
    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful academic assistant."},
                {"role": "user", "content": f"""
I have a potential abstract for the following paper. Verify if it is a valid abstract for this title.

**Title**: {title}

**Candidate Abstract**:
{candidate_abstract[:2000]}

**Instructions:**
1. If this text is indeed the abstract (or a valid description/summary) of the paper, return it Cleaned (remove artifacts, 'Abstract:', etc.).
2. If this text is NOT an abstract (e.g. it's a login page, a different paper, or junk), return `INVALID_ABSTRACT`.
3. If it is empty or too short (<50 chars), return `INVALID_ABSTRACT`.

Return ONLY the cleaned abstract text or the string `INVALID_ABSTRACT`.
"""}
            ],
            max_tokens=600,
            temperature=0.0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"    ! OpenAI Error: {e}")
        return None

def main():
    print(f"Loading Target: {TARGET_CSV}")
    try:
        df_target = pd.read_csv(TARGET_CSV, low_memory=False, encoding='utf-8')
    except UnicodeDecodeError:
        df_target = pd.read_csv(TARGET_CSV, low_memory=False, encoding='latin1')

    print(f"Loading Source: {SOURCE_CSV}")
    try:
        df_source = pd.read_csv(SOURCE_CSV, low_memory=False, encoding='utf-8')
    except UnicodeDecodeError:
        df_source = pd.read_csv(SOURCE_CSV, low_memory=False, encoding='latin1')

    # Prepare Source Lookups
    print("Building Lookup Tables...")
    doi_lookup = {}
    title_lookup = {} # normalized_title -> abstract
    
    count = 0 
    for _, row in df_source.iterrows():
        # Get Abstract
        abstract = row.get('abstract', '')
        if not abstract or pd.isna(abstract):
            continue
            
        if len(str(abstract)) < 50:
            continue
            
        abstract = str(abstract)

        # 1. DOI Key
        doi = str(row.get('DOI', '')).strip().lower()
        if doi and doi != 'nan':
            doi_lookup[doi] = abstract
            
        # 2. Title Keys (title AND found_title)
        titles = []
        t1 = str(row.get('title', '')).strip()
        t2 = str(row.get('found_title', '')).strip()
        if t1 and t1 != 'nan': titles.append(t1)
        if t2 and t2 != 'nan': titles.append(t2)
        
        for t in titles:
            norm = normalize_title(t)
            if norm:
                title_lookup[norm] = abstract
        
        count += 1
    
    print(f"Source Lookups Built: {len(doi_lookup)} DOIs, {len(title_lookup)} Titles.")

    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: No OpenAI API Key found.")
        return

    client = OpenAI()
    
    # Identify Missing Abstracts
    mask = df_target['abstract'].fillna("").astype(str).apply(lambda x: len(x) < 20)
    indices = df_target[mask].index.tolist()
    
    print(f"Found {len(indices)} matches to attempt backfill.")
    
    success_count = 0
    
    for i, idx in enumerate(indices):
        row = df_target.loc[idx]
        target_title = str(row.get('Title', '')).strip()
        
        target_doi = str(row.get('crossref_doi', '')).strip().lower()
        if not target_doi or target_doi == 'nan':
            target_doi = str(row.get('Dig Ob Id', '')).strip().lower()
            if 'doi.org/' in target_doi:
                target_doi = target_doi.split('doi.org/')[-1]
        
        candidate = None
        match_type = ""
        
        # 1. Try DOI Match (Exact)
        if target_doi and target_doi in doi_lookup:
            candidate = doi_lookup[target_doi]
            match_type = "DOI"
        
        # 2. Try Title Match (Normalized Exact)
        if not candidate and len(target_title) > 10:
            norm_title = normalize_title(target_title)
            if norm_title in title_lookup:
                candidate = title_lookup[norm_title]
                match_type = "Title_Match"
        
        if candidate:
            print(f"[{i+1}/{len(indices)}] Match Found ({match_type}): {target_title[:40]}...")
            
            # Verify with LLM
            verified_abstract = verify_abstract_with_llm(client, target_title, candidate)
            
            if verified_abstract and verified_abstract != "INVALID_ABSTRACT":
                print(f"    -> Verified & Cleaned ({len(verified_abstract)} chars).")
                df_target.at[idx, 'abstract'] = verified_abstract
                df_target.at[idx, 'abstract_source'] = "backfill_full_merged"
                df_target.at[idx, 'scraped_status'] = "success_backfill_merged"
                success_count += 1
                
                # Save every 5
                if success_count % 5 == 0:
                    df_target.to_csv(TARGET_CSV, index=False)
                    print("    (Saved)")
            else:
                print("    -> LLM Rejected.")
        
    df_target.to_csv(TARGET_CSV, index=False)
    print(f"Done. Backfilled {success_count} abstracts.")

if __name__ == "__main__":
    main()
