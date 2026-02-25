from seleniumbase import SB
from bs4 import BeautifulSoup
import time
import os

URLS = [
    "https://catalog.nlm.nih.gov/discovery/fulldisplay/alma9918487681406676/01NLM_INST:01NLM_INST",
    "https://www.routledge.com/Clinical-Cases-in-Augmentative-and-Alternative-Communication/Smith/p/book/9780367618285"
]

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

def main():
    print("Starting Debug Scrape for First Entry URLs...")
    with SB(uc=True, test=True, headless=False) as sb:
        for url in URLS:
            print(f"--- Visiting: {url} ---")
            try:
                sb.uc_open_with_reconnect(url, reconnect_time=4)
                
                # Human interaction (simulated)
                if sb.is_element_visible("button[aria-label='Accept all']"):
                    print("  Clicking Accept All (Button)...")
                    sb.click("button[aria-label='Accept all']")
                elif sb.is_element_visible("div.QS5gu.sy4vM"):
                    print("  Clicking Accept All (Div)...")
                    sb.click("div.QS5gu.sy4vM", delay=0.5)
                
                # Routledge specific: Scroll or click accordion?
                if "routledge.com" in url:
                    # Check for accordion
                    if sb.is_element_present(".accordion-button"):
                        print("  Found accordion button. Scrolling into view...")
                        sb.execute_script("document.querySelector('.accordion-button').scrollIntoView();")
                        time.sleep(1)

                time.sleep(3)
                
                page_source = sb.get_page_source()
                text = clean_html(page_source)
                
                print(f"  Extracted Text Length: {len(text)}")
                print("  --- PREVIEW (First 1000 chars) ---")
                print(text[:1000])
                print("  --- END PREVIEW ---")
                
                # Check for key phrase
                if "concise introduction" in text or "Explanatory Model" in text:
                    print("  SUCCESS: Key phrases found in text.")
                else:
                    print("  WARNING: Key phrases NOT found.")

            except Exception as e:
                print(f"  Error: {e}")

if __name__ == "__main__":
    main()
