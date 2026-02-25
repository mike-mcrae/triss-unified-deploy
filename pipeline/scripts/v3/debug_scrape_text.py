from seleniumbase import SB
from bs4 import BeautifulSoup
import time

URL = "https://www.routledge.com/The-Routledge-Handbook-of-Service-User-Involvement-in-Human-Services-Research-and-Education/McLaughlin-Beresford-Cameron-Casey-Duffy/p/book/9780367523565"

def clean_html(html_content):
    soup = BeautifulSoup(html_content, "html.parser")
    # match the logic in the main script
    for element in soup(["script", "style", "noscript", "svg", "path", "footer", "nav"]):
        element.decompose()
    text = soup.get_text(separator="\n", strip=True)
    return text

def main():
    print(f"Testing extraction for: {URL}")
    with SB(uc=True, test=True, headless=False) as sb:
        sb.uc_open_with_reconnect(URL, reconnect_time=4)
        time.sleep(5) # Wait for load
        
        # Check if we need to click "Show more" or similar?
        # Routledge often has a "Description" tab or section.
        
        source = sb.get_page_source()
        text = clean_html(source)
        
        print(f"Extracted Length: {len(text)}")
        print("--- START TEXT SAMPLE ---")
        print(text[:2000])
        print("--- END TEXT SAMPLE ---")
        
        with open("debug_scraped_text.txt", "w") as f:
            f.write(text)
        print("Full text saved to debug_scraped_text.txt")

if __name__ == "__main__":
    main()
