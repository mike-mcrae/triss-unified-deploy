from seleniumbase import SB
from bs4 import BeautifulSoup
from openai import OpenAI
import time
import os
import urllib.parse

# Configuration
OPENAI_MODEL = "gpt-4o-mini"
# The first entry details
TITLE = "The I-ASC Explanatory Model as a Support for AAC Assessment Planning A Case Report"
AUTHOR = "M.Smith"

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

def google_search(sb, query):
    print(f" -> Google Search: {query}")
    encoded_query = urllib.parse.quote_plus(query)
    search_url = f"https://www.google.com/search?q={encoded_query}"
    
    sb.uc_open_with_reconnect(search_url, reconnect_time=4)
    time.sleep(2)
    
    # Handle CAPTCHA if needed (manual intervention might be required if headless=False)
    if sb.is_element_visible('iframe[src*="google.com/recaptcha"]'):
        sb.uc_gui_click_captcha()
    
    # Extract results
    links = []
    soup = BeautifulSoup(sb.get_page_source(), "html.parser")
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.startswith("http") and "google.com" not in href:
            links.append(href)
            if len(links) >= 2:
                break
    return links

def get_description_from_llm(client, title, page_text):
    print(f"\n--- CALLING LLM for '{title}' ---")
    
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
    # Debug print
    print(f"\n--- This is what LLM is Seeing: ---\n{user_content[:2000]} ... [truncated] ... {user_content[-500:]}\n-----------------------------------\n")

    try:
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

def main():
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY missing.")
        return

    client = OpenAI()
    
    with SB(uc=True, test=True, headless=False) as sb:
        print(f"\n\n=======================================================")
        print(f"PROCESSING ENTRY 1: {TITLE}")
        print(f"=======================================================")
        
        # 1. Search
        query = f'"{TITLE}" {AUTHOR}'
        links = google_search(sb, query)
        print(f" -> Found Links: {links}")
        
        for i, url in enumerate(links):
            print(f"\n--- Outputting Result {i+1}: {url} ---")
            try:
                sb.uc_open_with_reconnect(url, reconnect_time=4)
                
                # NLM specific scroll
                if "nlm.nih.gov" in url:
                    print("  -> Scrolling page for NLM...")
                    sb.execute_script("window.scrollTo(0, 1000);")
                    time.sleep(2)
                
                 # Routledge specific scroll
                if "routledge.com" in url and sb.is_element_present(".accordion-button"):
                     print("  -> Scrolling to accordion...")
                     sb.execute_script("document.querySelector('.accordion-button').scrollIntoView();")
                     time.sleep(1)

                time.sleep(2)
                
                page_source = sb.get_page_source()
                clean_text = clean_html(page_source)
                
                result = get_description_from_llm(client, TITLE, clean_text)
                print(f"\n>>> FINAL LLM RESULT for Result {i+1}: {result}")
                
                if result and "NO_CONTENT" not in result:
                    print(f"SUCCESS! Found abstract in Result {i+1}. Stopping.")
                    break
                else:
                    print(f"Failed to find content in Result {i+1}.")
                    
            except Exception as e:
                print(f"Error visiting {url}: {e}")

if __name__ == "__main__":
    main()
