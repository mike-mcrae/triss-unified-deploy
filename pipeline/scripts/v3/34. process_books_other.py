import pandas as pd
import os
import re
from openai import OpenAI
import traceback
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

# Paths
BASE_DIR = str(_PIPELINE_ROOT / "1. data")
TARGET_CSV = os.path.join(BASE_DIR, "1. raw/0. profiles/v3/2c.all_listed_publications_2019_books_other.csv")
SOURCE_FINAL = os.path.join(BASE_DIR, "1. raw/0. profiles/profiles_publications_final.csv")
SOURCE_FULL = str(_SHARED_DATA_ROOT / "3. final/3. profiles_publications_full_merged.csv")
OUTPUT_CSV = os.path.join(BASE_DIR, "1. raw/0. profiles/v3/2e.all_listed_publications_2019_books_other_abstracts.csv")

OPENAI_MODEL = "gpt-4o-mini"

def normalize_title(text):
    if not isinstance(text, str): return ""
    return re.sub(r'\W+', '', text.lower())

def verify_book_abstract(client, title, text):
    """
    Asks LLM if the text is a valid description/abstract of the book/chapter.
    """
    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful academic assistant."},
                {"role": "user", "content": f"""
I have a potential description for the following Book or Book Chapter. 
Verify if it is a valid abstract or description.

**Title**: {title}

**Candidate Text**:
{text[:2000]}

**Instructions:**
1. If this text describes the book/chapter content, return the Cleaned text.
2. If it is NOT a description (e.g. just metadata, a different book, login page), return `INVALID`.
3. If it is too short (<50 chars), return `INVALID`.

Return ONLY the cleaned text or `INVALID`.
"""}
            ],
            max_tokens=600,
            temperature=0.0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error: {e}")
        return None

def main():
    print(f"Loading Target: {os.path.basename(TARGET_CSV)}")
    try:
        df_target = pd.read_csv(TARGET_CSV, low_memory=False, encoding='utf-8')
    except:
        df_target = pd.read_csv(TARGET_CSV, low_memory=False, encoding='latin1')

    print("Loading Source 1: profiles_publications_final.csv")
    try:
        df_src1 = pd.read_csv(SOURCE_FINAL, low_memory=False, encoding='utf-8')
    except:
        df_src1 = pd.read_csv(SOURCE_FINAL, low_memory=False, encoding='latin1')

    print("Loading Source 2: profiles_publications_full_merged.csv")
    try:
        df_src2 = pd.read_csv(SOURCE_FULL, low_memory=False, encoding='utf-8')
    except:
        df_src2 = pd.read_csv(SOURCE_FULL, low_memory=False, encoding='latin1')

    # Build Lookups
    # Key: normalized_title -> { 'abstract': ..., 'doi': ..., 'url': ... }
    lookup = {}

    def add_to_lookup(df, title_col, doi_col, url_col, abstract_col):
        for _, row in df.iterrows():
            title = str(row.get(title_col, ''))
            norm = normalize_title(title)
            if not norm or len(norm) < 10: continue
            
            # Extract data
            data = {}
            if abstract_col in row and isinstance(row[abstract_col], str):
                data['abstract'] = row[abstract_col]
            if doi_col in row and isinstance(row[doi_col], str):
                data['doi'] = row[doi_col]
            if url_col in row and isinstance(row[url_col], str):
                data['url'] = row[url_col]
            
            # Merge if exists
            if norm in lookup:
                lookup[norm].update({k: v for k, v in data.items() if v and str(v).lower() != 'nan'})
            else:
                lookup[norm] = data

    print("Building Lookup Tables...")
    add_to_lookup(df_src1, 'found_title', 'doi', 'url', 'abstract')
    add_to_lookup(df_src1, 'publication_reference', 'doi', 'url', 'abstract') 

    add_to_lookup(df_src2, 'title', 'DOI', 'URL', 'abstract')
    add_to_lookup(df_src2, 'found_title', 'DOI', 'URL', 'abstract')

    print(f"Lookup Table Built: {len(lookup)} titles.")
    
    # Initialize client if key present, else warn
    client = None
    if os.environ.get("OPENAI_API_KEY"):
         client = OpenAI()
    else:
         print("WARNING: No OPENAI_API_KEY found. LLM verification will be skipped.")

    # Process Target
    df_target['abstract'] = ''
    df_target['abstract_source'] = ''
    df_target['main_url'] = ''
    
    count_abstracts = 0
    count_urls = 0
    
    print(f"Processing {len(df_target)} rows...")
    
    for idx, row in df_target.iterrows():
        title = str(row.get('Title', ''))
        norm = normalize_title(title)
        
        # 1. Recover URL/DOI if missing
        current_doi = str(row.get('Dig Ob Id', ''))
        current_url = str(row.get('Url', ''))
        
        match_data = lookup.get(norm)
        
        if match_data:
            # Recover DOI
            if (not current_doi or current_doi.lower() in ['nan', '-', '']) and match_data.get('doi'):
                df_target.at[idx, 'Dig Ob Id'] = match_data['doi']
                current_doi = match_data['doi']
                
            # Recover URL
            if (not current_url or current_url.lower() in ['nan', '-', '']) and match_data.get('url'):
                df_target.at[idx, 'Url'] = match_data['url']
                current_url = match_data['url']
                
            # Recover Abstract
            candidate_abstract = match_data.get('abstract')
            if candidate_abstract and isinstance(candidate_abstract, str) and len(candidate_abstract) > 50:
                if client:
                    # Verify Only if we haven't already processed it? (Assuming fresh run)
                    # For now, verify all candidates.
                    verified = verify_book_abstract(client, title, candidate_abstract)
                    if verified and verified != 'INVALID':
                        print(f"[{idx}] Abstract Verified: '{title[:30]}...'")
                        df_target.at[idx, 'abstract'] = verified
                        df_target.at[idx, 'abstract_source'] = 'book_recovery_llm'
                        count_abstracts += 1
                else:
                    # If no client, accept it blindly? Or skip?
                    # User request implied LLM verification is key.
                    # I'll enable blind acceptance if key missing? No, user wants verification.
                    print(f"Skipping verification for '{title[:20]}' (No API Key)")
        
        # Set Main URL
        main_url = ''
        if current_doi and current_doi.lower() not in ['nan', '-']:
             if 'doi.org' in current_doi: main_url = current_doi
             else: main_url = f"https://doi.org/{current_doi}"
        elif current_url and current_url.lower() not in ['nan', '-']:
             main_url = current_url
        
        df_target.at[idx, 'main_url'] = main_url
        if main_url: count_urls += 1
        
    print(f"Finished. Recovered {count_abstracts} abstracts and populated {count_urls} URLs.")
    df_target.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved to {os.path.basename(OUTPUT_CSV)}")

if __name__ == "__main__":
    try:
        print("Starting script execution...")
        main()
    except Exception as e:
        print(f"CRITICAL FAILIURE: {e}")
        traceback.print_exc()
