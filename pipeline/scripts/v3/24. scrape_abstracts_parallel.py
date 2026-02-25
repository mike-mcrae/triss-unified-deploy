import pandas as pd
from seleniumbase import SB
from bs4 import BeautifulSoup
import time
import os
import random
import math
import urllib.parse
from openai import OpenAI
from concurrent.futures import ProcessPoolExecutor, as_completed
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
BASE_DIR = str(_PIPELINE_ROOT / "1. data/1. raw/0. profiles/v3")
INPUT_CSV = os.path.join(BASE_DIR, "2e.all_listed_publications_2019_journal_like_no_abstract.csv")
OPENAI_MODEL = "gpt-4o-mini"
MAX_BODY_CHARS = 12000 # Increased for better paraphrasing context
NUM_WORKERS = 5

def clean_text(text):
    if not text: return ""
    return text.strip().replace('\n', ' ').replace('\r', ' ')

def get_abstract_from_llm(client, raw_text):
    """
    Sends raw text to GPT-4o-mini to extract OR paraphrase the abstract.
    """
    try:
        if len(raw_text) < 300:
            return None

        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful academic assistant skilled at summarizing research papers."},
                {"role": "user", "content": f"""
Please identify and extract the **Abstract** from the following text.

**Instructions:**
1. **Extraction**: If there is a clear "Abstract" section, extract it word-for-word.
2. **Paraphrasing (Fallback)**: If NO formal abstract exists, but the text contains an Introduction, Summary, or Conclusion that clearly describes the study's **Aims, Methods, and Results**, write a concise paragraph (150-250 words) summarizing the paper. 
    - Prefix paraphrased abstracts with: `[Paraphrased]: `
3. **Failure**: If the text is unrelated (e.g., login page, cookie banner, random footer text) or does not contain enough info to summarize the paper, return exactly: `NO_ABSTRACT_FOUND`.

**Raw Text:**
{raw_text[:MAX_BODY_CHARS]}
"""}
            ],
            max_tokens=800,
            temperature=0.3 # Slightly higher for paraphrasing creativity
        )
        content = response.choices[0].message.content.strip()
        
        if "NO_ABSTRACT_FOUND" in content:
            return None
        return content
        
    except Exception as e:
        print(f"    ! OpenAI Error: {e}")
        return None

def google_search_urls(sb, query):
    """Searches Google and returns the top 2 non-ad result URLs."""
    try:
        search_url = f"https://www.google.com/search?q={urllib.parse.quote(query)}"
        sb.uc_open_with_reconnect(search_url, reconnect_time=4)
        
        # Handle CAPTCHA if needed
        try:
             if sb.is_element_visible('iframe[src*="cloudflare"]') or sb.is_text_visible("Just a moment"):
                 sb.uc_gui_click_captcha()
                 time.sleep(3)
        except: pass
        
        time.sleep(random.uniform(2, 4))
        
        # Extract links
        links = []
        time.sleep(2)

        # Check title/content for blocks
        title = sb.get_title()

        # 1. CAPTCHA / Unusual Traffic
        if "Sorry" in title or sb.is_element_visible('iframe[src*="recaptcha"]') or sb.is_text_visible("unusual traffic"):
             print("    ! CAPTCHA/Block detected. In Non-headless mode, solve it manually if needed.")
             # GUI click only works if not headless
             try: sb.uc_gui_click_captcha()
             except: pass
             time.sleep(5)

        # 2. Consent Form (Region specific)
        if sb.is_element_visible('button#L2AGLb') or sb.is_element_visible('button#W0wltc'):
             print("    ! Consent Form detected. Clicking...")
             try: sb.click('button#L2AGLb')
             except: sb.click('button#W0wltc')
             time.sleep(3)

        time.sleep(2)

        # Robust Selectors
        selectors = [
            "div.g a[data-ved]",
            "div.yuRUbf a",
            "a > h3",
            "div#search a h3"
        ]

        for sel in selectors:
            try:
                elements = sb.find_elements(sel)
                for el in elements:
                    try:
                        href = el.get_attribute("href")
                        if not href and el.tag_name == "h3":
                             parent = el.find_element("./..")
                             href = parent.get_attribute("href")

                        if href and "google.com" not in href and href.startswith("http"):
                             if href not in links:
                                 links.append(href)
                    except: continue
                    if len(links) >= 3: break
            except: pass
            if links: break

    except Exception as e:
        print(f"    ! Search Error: {e}")
    return links

def scrape_and_process_url(sb, client, url, worker_id):
    """Helper to scrape a URL and try to get an abstract."""
    try:
        sb.uc_open_with_reconnect(url, reconnect_time=4)
        try:
             if sb.is_element_visible('iframe[src*="cloudflare"]') or sb.is_text_visible("Just a moment"):
                 sb.uc_gui_click_captcha()
                 time.sleep(3)
        except: pass

        time.sleep(random.uniform(3, 6))

        html = sb.get_page_source()
        soup = BeautifulSoup(html, 'html.parser')

        body_text = ""
        article = soup.find('article') or soup.find('div', role='main') or soup.find('main')
        if article:
            body_text = article.get_text(separator=' ', strip=True)
        elif soup.body:
            body_text = soup.body.get_text(separator=' ', strip=True)

        body_text = clean_text(body_text)

        if len(body_text) > 300:
             abstract = get_abstract_from_llm(client, body_text)
             return abstract, url

    except Exception as e:
        print(f"[Worker {worker_id}] Error scraping {url}: {e}")

    return None, None

def scrape_worker(worker_id, subset_indices, input_csv_path):
    print(f"[Worker {worker_id}] Starting... ({len(subset_indices)} rows)")

    df = pd.read_csv(input_csv_path, low_memory=False)
    # Ensure columns exist
    for col in ['url_found', 'scraped_status', 'abstract', 'abstract_source']:
        if col not in df.columns:
            df[col] = ""

    df_subset = df.loc[subset_indices].copy()

    if not os.environ.get("OPENAI_API_KEY"):
        print(f"[Worker {worker_id}] ERROR: No API Key")
        return None

    client = OpenAI()
    processed_count = 0
    results = []

    try:
        print(f"[Worker {worker_id}] Entering SB context...")
        with SB(uc=True, test=True, headless=False) as sb:
            print(f"[Worker {worker_id}] SB Context Active.")

            for idx in subset_indices:
                try:
                    row = df.loc[idx]

                    # 1. Source Resolution Priority
                    target_url = None
                    doi = str(row.get('crossref_doi', '')).strip()
                    dig_ob_id = str(row.get('Dig Ob Id', '')).strip()
                    url_col = str(row.get('Url', '')).strip()
                    tara = str(row.get('Tara Handle', '')).strip()

                    if doi and doi != "nan":
                        target_url = f"https://doi.org/{doi}"
                    elif dig_ob_id and dig_ob_id != "nan":
                         target_url = f"https://doi.org/{dig_ob_id}" if "doi.org" not in dig_ob_id else dig_ob_id
                    elif url_col and url_col != "nan" and url_col.startswith("http"):
                        target_url = url_col
                    elif tara and tara != "nan" and tara.startswith("http"):
                        target_url = tara

                    abstract = None
                    final_url = ""
                    source_type = ""
                    status = "failed"

                    # 2. Try Target URL
                    if target_url:
                        print(f"[Worker {worker_id}] [{idx}] Trying Direct URL: {target_url}")
                        abstract, final_url = scrape_and_process_url(sb, client, target_url, worker_id)
                        if abstract: source_type = "direct_url_llm"

                    # 3. Fallback: Google Search and DuckDuckGo
                    if not abstract:
                        title = str(row.get('Title', '')).strip()
                        author = str(row.get('Author', '')).strip()
                        if title and title != "nan":
                                     break
                    
                    if abstract:
                        print(f"[Worker {worker_id}] -> Success! ({len(abstract)} chars)")
                        status = "success"
                    else:
                        print(f"[Worker {worker_id}] -> Failed (All methods exhausted).")

                    # Store result
                    results.append({
                        'index': idx,
                        'abstract': abstract if abstract else "",
                        'abstract_source': source_type if source_type else "",
                        'scraped_status': status,
                        'url_found': final_url if final_url else ""
                    })
                    
                    processed_count += 1
                    if processed_count % 5 == 0:
                         save_temp_results(worker_id, results)
                         
                except Exception as e:
                    print(f"[Worker {worker_id}] Error on {idx}: {e}")
                    
    except Exception as e:
        print(f"[Worker {worker_id}] CRITICAL FAIL: {e}")
        
    temp_file = save_temp_results(worker_id, results)
    print(f"[Worker {worker_id}] Finished. Saved to {temp_file}")
    return temp_file

def save_temp_results(worker_id, results):
    if not results: return None
    temp_df = pd.DataFrame(results)
    temp_path = os.path.join(BASE_DIR, f"temp_worker_{worker_id}.csv")
    temp_df.to_csv(temp_path, index=False)
    return temp_path

def main():
    print(f"Loading {INPUT_CSV}...")
    if not os.path.exists(INPUT_CSV):
        print("File not found.")
        return

    # Fix: Ensure SeleniumBase log directory exists
    log_dir = os.path.join(os.getcwd(), "latest_logs")
    os.makedirs(log_dir, exist_ok=True)

    df = pd.read_csv(INPUT_CSV, low_memory=False)
    
    # Init columns
    for col in ['url_found', 'scraped_status', 'abstract', 'abstract_source']:
        if col not in df.columns:
            df[col] = ""
            
    df['abstract'] = df['abstract'].fillna("").astype(str)
    
    # Filter rows
    indices_to_process = df[df['abstract'].apply(lambda x: len(str(x)) < 20)].index.tolist()
    total_rows = len(indices_to_process)
    
    print(f"Found {total_rows} rows to scrape.")
    
    if total_rows == 0:
        print("Nothing to do.")
        return

    chunk_size = math.ceil(total_rows / NUM_WORKERS)
    chunks = [indices_to_process[i:i + chunk_size] for i in range(0, total_rows, chunk_size)]
    
    print(f"Launching {len(chunks)} workers (Approx {chunk_size} rows each)...")
    
    temp_files = []
    
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = []
        for i, chunk in enumerate(chunks):
            if not chunk: continue
            time.sleep(5) # Stagger
            futures.append(executor.submit(scrape_worker, i, chunk, INPUT_CSV))
            
        for future in as_completed(futures):
            res = future.result()
            if res:
                temp_files.append(res)
                
    print("\nAll workers finished. Merging results...")
    
    df_final = pd.read_csv(INPUT_CSV, low_memory=False)
    for col in ['url_found', 'scraped_status', 'abstract', 'abstract_source', 'Url']:
         if col not in df_final.columns:
            df_final[col] = ""
            
    updates_count = 0
    
    for tf in temp_files:
        if os.path.exists(tf):
            try:
                temp_df = pd.read_csv(tf)
                for _, row in temp_df.iterrows():
                    idx = int(row['index'])
                    if row['scraped_status'] == 'success':
                        df_final.at[idx, 'abstract'] = row['abstract']
                        df_final.at[idx, 'abstract_source'] = row['abstract_source']
                        df_final.at[idx, 'scraped_status'] = 'success'
                        df_final.at[idx, 'url_found'] = row['url_found']
                        # Update the main URL column if it was empty, as requested
                        if pd.isna(df_final.at[idx, 'Url']) or str(df_final.at[idx, 'Url']) == 'nan' or str(df_final.at[idx, 'Url']) == '':
                             df_final.at[idx, 'Url'] = row['url_found']
                             
                        updates_count += 1
                    elif row['scraped_status'] == 'failed':
                         df_final.at[idx, 'scraped_status'] = 'failed'
                os.remove(tf)
            except Exception as e:
                print(f"Error merging {tf}: {e}")
                
    print(f"Saving final CSV with {updates_count} new abstracts...")
    df_final.to_csv(INPUT_CSV, index=False)
    print("Done.")

if __name__ == "__main__":
    main()
