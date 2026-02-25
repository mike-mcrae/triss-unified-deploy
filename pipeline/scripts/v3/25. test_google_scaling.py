import pandas as pd
from seleniumbase import SB
from bs4 import BeautifulSoup
import time
import os
import random
import urllib.parse
from openai import OpenAI
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
MAX_BODY_CHARS = 10000

def clean_text(text):
    if not text: return ""
    return text.strip().replace('\n', ' ').replace('\r', ' ')

def get_abstract_from_llm(client, raw_text):
    print("    -> Sending %d chars to LLM..." % len(raw_text))
    try:
        if len(raw_text) < 300:
            print("    -> Text too short (<300 chars). Stopping.")
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
            temperature=0.3
        )
        content = response.choices[0].message.content.strip()
        
        if "NO_ABSTRACT_FOUND" in content:
            print("    -> LLM returned NO_ABSTRACT_FOUND.")
            return None
        return content
        
    except Exception as e:
        print(f"    ! OpenAI Error: {e}")
        return None

def google_search_urls(sb, query):
    links = []
    try:
        search_url = f"https://www.google.com/search?q={urllib.parse.quote(query)}"
        print(f"    Searching Google: {search_url}")
        sb.uc_open_with_reconnect(search_url, reconnect_time=4)
        time.sleep(2)
        
        # Check title/content for blocks
        title = sb.get_title()
        print(f"    Page Title: {title}")
        
        # 1. CAPTCHA / Unusual Traffic
        if "Sorry" in title or sb.is_element_visible('iframe[src*="recaptcha"]') or sb.is_text_visible("unusual traffic"):
             print("    ! CAPTCHA/Block detected. Attempting solve...")
             sb.uc_gui_click_captcha()
             time.sleep(5)
             
        # 2. Consent Form (Region specific)
        if sb.is_element_visible('button#L2AGLb') or sb.is_element_visible('button#W0wltc'):
             print("    ! Consent Form detected. Clicking...")
             try: sb.click('button#L2AGLb')
             except: sb.click('button#W0wltc')
             time.sleep(3)
        
        time.sleep(3)
        
        # Robust Selectors for 2024/2025 Google Layout
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
                        if not href:
                            # If we selected the h3, get parent href
                            if el.tag_name == "h3":
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
        
    if not links:
        print("    ! No links found. Saving screenshot and HTML...")
        clean_query = "".join(x for x in query if x.isalnum())[:20]
        sb.save_screenshot(f"google_fail_{clean_query}.png")
        with open(f"google_fail_{clean_query}.html", "w") as f:
            f.write(sb.get_page_source())
        
    return links

def main():
    print(f"Loading {INPUT_CSV}...")
    if not os.path.exists(INPUT_CSV):
        print("File not found.")
        return

    df = pd.read_csv(INPUT_CSV, low_memory=False)
    
    # Filter for MISSING abstracts
    df['abstract'] = df['abstract'].fillna("").astype(str)
    missing_df = df[df['abstract'].apply(lambda x: len(str(x)) < 20)].copy()
    
    if len(missing_df) == 0:
        print("No missing abstracts found!")
        return
        
    # Take top 10
    top_10 = missing_df.head(10).index.tolist()
    print(f"Testing top 10 missing rows: {top_10}")
    
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: No OpenAI API Key found.")
        return

    client = OpenAI()
    
    # Log dir
    log_dir = os.path.join(os.getcwd(), "latest_logs_debug")
    os.makedirs(log_dir, exist_ok=True)

    with SB(uc=True, test=True, headless=False) as sb:
        for idx in top_10:
            print("\n" + "="*50)
            row = df.loc[idx]
            title = str(row.get('Title', '')).strip()
            # Try 'Last Name', fallback to 'Author' or whatever column holds names
            lastname = str(row.get('Last Name', '')).strip()
            if not lastname or lastname == 'nan':
                 lastname = str(row.get('Author', '')).strip()
            
            print(f"[{idx}] Processing: '{title}' / '{lastname}'")
            
            if not title or title == 'nan':
                print("    ! Skipping (No Title)")
                continue
                
            query = f"{title} {lastname}"
            print(f"    Query: {query}")
            
            # 1. Try Google
            urls = []
            try:
                urls = google_search_urls(sb, query)
            except Exception as e:
                print(f"    ! Google Search crashed: {e}")
                
            # 2. Try DuckDuckGo if Google failed
            if not urls:
                print("    ! Google failed. Switching to DuckDuckGo...")
                try:
                    ddg_url = f"https://duckduckgo.com/?q={urllib.parse.quote(query)}"
                    sb.uc_open_with_reconnect(ddg_url, reconnect_time=4)
                    time.sleep(3)
                    
                    links = []
                    elements = sb.find_elements("a[data-testid='result-title-a']")
                    for el in elements:
                         href = el.get_attribute("href")
                         if href: links.append(href)
                         if len(links) >= 2: break
                    
                    if not links: # Fallback selector
                        elements = sb.find_elements("h2 a")
                        for el in elements:
                             href = el.get_attribute("href")
                             if href: links.append(href)
                             if len(links) >= 2: break
                             
                    urls = links
                    print(f"    DDG URLs: {urls}")
                except Exception as e:
                    print(f"    ! DDG Error: {e}")

            print(f"    Found URLs: {urls}")
            
            success = False
            for url in urls:
                print(f"    -> Scraping: {url}")
                try:
                    sb.uc_open_with_reconnect(url, reconnect_time=4)
                    time.sleep(4)
                    
                    html = sb.get_page_source()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    # Capture broad text
                    body_text = ""
                    article = soup.find('article') or soup.find('div', role='main') or soup.find('main')
                    if article:
                        body_text = article.get_text(separator=' ', strip=True)
                    elif soup.body:
                        body_text = soup.body.get_text(separator=' ', strip=True)
                    
                    body_text = clean_text(body_text)
                    print(f"    -> Extracted {len(body_text)} chars.")
                    
                    abstract = get_abstract_from_llm(client, body_text)
                    if abstract:
                        print(f"    -> SUCCESS:\n{abstract[:300]}...")
                        success = True
                        break
                    else:
                        print("    -> Failed to extract abstract.")
                        
                except Exception as e:
                    print(f"    ! Error scraping {url}: {e}")
            
            if not success:
                print("    ! FAILED (All URLs tried).")

if __name__ == "__main__":
    main()
