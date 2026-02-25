import pandas as pd
import os
import time
import random
import json
import sys
from seleniumbase import SB
from openai import OpenAI
from bs4 import BeautifulSoup
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
BASE_DIR = str(_PIPELINE_ROOT / "1. data/1. raw/0. profiles/v3")
TARGET_CSV = os.path.join(BASE_DIR, "2e.all_listed_publications_2019_books_other_abstracts.csv")
CHECKPOINT_FILE = os.path.join(BASE_DIR, "checkpoint_scrape.json")

OPENAI_MODEL = "gpt-4o-mini"

def clean_html(html_content):
    soup = BeautifulSoup(html_content, "html.parser")
    meta_desc = ""
    for meta in soup.find_all("meta"):
        if meta.get("name") == "description":
            meta_desc += f"Meta Description: {meta.get('content', '')}\n"
        if meta.get("property") == "og:description":
            meta_desc += f"OG Description: {meta.get('content', '')}\n"

    for element in soup(["script", "style", "noscript", "svg", "path", "footer", "nav"]):
        element.decompose()
    
    text = soup.get_text(separator="\n", strip=True)
    full_text = f"{meta_desc}\n\n{text}"
    return full_text[:120000]

def get_description_from_llm(client, title, page_text):
    try:
        user_content = f"""
I am looking for a description/abstract for the publication: "**{title}**".

Here is text from a webpage found via search:
--- START CONTENT ---
{page_text}
--- END CONTENT ---

**Task**:
1. Search the provided text for ANY description, summary, or abstract for the publication "**{title}**".
2. **IMPORTANT**: If the publication is a CHAPTER in a book, and the text contains a description of the BOOK, **extract the BOOK description** and preface it with "Book Description: ".
3. It might be a book description, a chapter summary, or a blurb.
4. If YES, extract it and return it (rewrite slightly if needed to be a standalone abstract).
5. If NO (e.g. login page, completely unrelated, generic home page), return `NO_CONTENT`.

**Output**:
Return ONLY the abstract text or `NO_CONTENT`.
"""
        # DEBUG: Save exact prompt
        with open("debug_last_prompt.txt", "w") as f:
            f.write(user_content)
        
        # DEBUG: Print to terminal as requested
        print(f"\n--- This is what LLM is Seeing: ---\n{user_content}\n-----------------------------------\n")

        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are a research assistant."},
                {"role": "user", "content": user_content}
            ],
            temperature=0.0,
            max_tokens=400
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"LLM Error: {e}")
        return None

def perform_google_search(sb, query, max_results=2):
    """
    Returns (links, status)
    Status: 'SUCCESS', 'CAPTCHA', 'ERROR'
    """
    try:
        print(f"    -> Google Search: {query}")
        from urllib.parse import quote_plus
        search_url = f"https://www.google.com/search?q={quote_plus(query)}"
        
        sb.uc_open(search_url)
        
        # Check for CAPTCHA elements
        captcha_present = False
        if sb.is_element_visible('iframe[src*="google.com/recaptcha"]'):
             print("    -> CAPTCHA Detected (iframe). Trying to click...")
             sb.uc_gui_click_captcha()
             captcha_present = True
        
        # "Our systems have detected unusual traffic..."
        if "unusual traffic" in sb.get_page_source() or sb.is_element_visible("#captcha-form"):
             print("    -> CAPTCHA Detected (Unusual Traffic Page).")
             captcha_present = True

        if sb.is_element_visible("button[aria-label='Accept all']"):
            sb.click("button[aria-label='Accept all']")
        elif sb.is_element_visible("div.QS5gu.sy4vM"):
            sb.click("div.QS5gu.sy4vM", delay=0.5)

        time.sleep(2 + random.random())
        
        # Validate if we actually got results
        links = []
        elements = sb.find_elements("h3")
        
        # If no results and we saw a captcha, consider it a HARD CAPTCHA failure
        if not elements and captcha_present:
             print("    -> blocked by CAPTCHA.")
             return [], 'CAPTCHA'

        for el in elements:
            try:
                parent = el.find_element("xpath", "./..")
                href = parent.get_attribute("href")
                if href and "google.com" not in href and href.startswith("http"):
                    links.append(href)
                    if len(links) >= max_results:
                        break
            except:
                continue
        
        return links, 'SUCCESS'

    except Exception as e:
        print(f"    -> Search Error: {e}")
        # Re-raise to trigger browser restart if it's a critical connection error
        if "Read timed out" in str(e) or "HTTPConnectionPool" in str(e):
             raise e
        return [], 'ERROR'

def main():
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY missing.")
        return

    print(f"Loading {os.path.basename(TARGET_CSV)}...")
    try:
        df = pd.read_csv(TARGET_CSV, low_memory=False, encoding='utf-8')
    except:
        df = pd.read_csv(TARGET_CSV, low_memory=False, encoding='latin1')

    # Target: Missing abstracts OR 'NO_CONTENT'
    mask = (df['abstract'].isna()) | (df['abstract'] == '') | (df['abstract'] == 'NO_CONTENT')
    target_indices = df[mask].index.tolist()
    print(f"Found {len(target_indices)} rows to search.")

    start_idx = 199 # Default to start from 200 onwards
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, "r") as f:
                ckpt = json.load(f)
                start_idx = ckpt.get("last_index", start_idx)
                print(f"Loaded checkpoint: skipping up to and including dataframe index {start_idx}")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")

    client = OpenAI()
    
    # SINGLE BROWSER INSTANCE
    consecutive_captcha_count = 0
    
    try:
        with SB(uc=True, test=True, headless=False) as sb:
            for idx in target_indices:
                if idx <= start_idx:
                    continue

                if consecutive_captcha_count >= 3:
                    print("\n!!! STOPPING: Encountered 3 consecutive Unsolvable CAPTCHAs. !!!")
                    break

                row = df.loc[idx]
                title = str(row.get('Title', ''))
                authors = str(row.get('Authors', ''))
                
                # Clear status
                df.at[idx, 'abstract'] = '' 
                df.at[idx, 'abstract_source'] = ''

                print(f"[{idx}] Processing: {title[:40]}...")
                
                # Search Logic
                urls = []
                status = 'ERROR'
                
                try:
                    # 1. Exact Match
                    first_author = authors.split(',')[0] if authors else ""
                    query_exact = f'"{title}" {first_author}'
                    urls, status = perform_google_search(sb, query_exact, max_results=2)
                    
                    if status == 'CAPTCHA':
                        consecutive_captcha_count += 1
                        print(f"    (Captcha Strike {consecutive_captcha_count}/3)")
                        if consecutive_captcha_count >= 3:
                            print("\n!!! STOPPING: Encountered 3 consecutive Unsolvable CAPTCHAs. !!!")
                            sys.exit(1)
                        time.sleep(10) # Wait a bit
                        continue # Skip to next item (or retry broad? usually safer to wait)
                    
                    # 2. Broad Match (if no urls and not blocked)
                    if not urls and status == 'SUCCESS':
                         query_broad = f'{title} {first_author}'.replace('"', '').replace("'", "")
                         print(f"    -> Exact match failed. Trying broad...")
                         urls, status = perform_google_search(sb, query_broad, max_results=2)

                         if status == 'CAPTCHA':
                            consecutive_captcha_count += 1
                            print(f"    (Captcha Strike {consecutive_captcha_count}/3)")
                            if consecutive_captcha_count >= 3:
                                print("\n!!! STOPPING: Encountered 3 consecutive Unsolvable CAPTCHAs. !!!")
                                sys.exit(1)
                            time.sleep(10)
                            continue

                except Exception as e:
                    print(f"    -> CRITICAL SEARCH ERROR: {e}")
                    # If browser crashed, we might need to break or restart loop. 
                    # For now, let's break to be safe as user requested "keep one browser".
                    break

                # If successful search (even if 0 results), reset captcha count
                consecutive_captcha_count = 0 

                if not urls:
                    print("    -> No URLs found.")
                    continue
                    
                found_abstract = False
                for i, url in enumerate(urls):
                    if "tcd.ie" in url:
                        print(f"    Skipping TCD URL: {url}")
                        continue
                    
                    print(f"    Visiting Result {i+1}: {url}")
                    try:
                        sb.uc_open(url)
                        time.sleep(2 + random.random()) 
                        
                        page_source = sb.get_page_source()
                        clean_text = clean_html(page_source)
                        desc = get_description_from_llm(client, title, clean_text)
                        
                        if desc and desc != 'NO_CONTENT' and "NO_CONTENT" not in desc:
                            print(f"    -> FOUND Abstract ({len(desc)} chars).")
                            df.at[idx, 'abstract'] = desc
                            df.at[idx, 'abstract_source'] = 'google_search_llm'
                            df.at[idx, 'main_url'] = url 
                            found_abstract = True
                            # Save immediately on success
                            df.to_csv(TARGET_CSV, index=False)
                            break 
                        else:
                            print("    -> LLM Rejected.")
                            # Save debug text to verify what was sent
                            debug_dir = os.path.join(BASE_DIR, "debug_scrapes_v3")
                            os.makedirs(debug_dir, exist_ok=True)
                            with open(os.path.join(debug_dir, f"fail_{idx}_{i}.txt"), "w") as f:
                                f.write(f"URL: {url}\n\n{clean_text}")
                            
                    except Exception as e:
                        print(f"    -> Error visiting {url}: {e}")
                        # Don't raise here, just skip URL
                
                if not found_abstract:
                    print("    -> Failed to extract abstract.")
                    
                # Update checkpoint after successfully concluding an item (no unhandled CAPTCHA)
                with open(CHECKPOINT_FILE, "w") as f:
                    json.dump({"last_index": idx}, f)

    except Exception as e:
        print(f"BROWSER CRASHED or STOPPED: {e}")
        traceback.print_exc()

    print("Done.")

if __name__ == "__main__":
    main()
