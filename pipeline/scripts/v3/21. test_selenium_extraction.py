from selenium.webdriver.common.by import By
import undetected_chromedriver as uc
from bs4 import BeautifulSoup
import time

def main():
    url = "https://doi.org/10.1016/j.intfin.2023.101882"
    print(f"Testing URL (undetected): {url}")

    options = uc.ChromeOptions()
    options.headless = False  # Headless is often detected even with UC, try visible first or keep headless=True if needed.
    # But for a server script, headless is preferred. Let's try headless=True first in UC terms.
    # Note: UC headless handling is different.
    # options.add_argument('--headless=new') 

    try:
        print("Initializing Undetected ChromeDriver...")
        # version_main allows specifying distinct chrome version if auto-detect fails, usually optional
        driver = uc.Chrome(options=options, headless=True, use_subprocess=False, version_main=144)
        
        print("Navigating...")
        driver.get(url)
        
        # Wait for redirect and render. ScienceDirect can be slow or have intermediate pages.
        time.sleep(10) 
        
        print(f"Current URL: {driver.current_url}")
        print(f"Page Title: {driver.title}")

        html = driver.page_source
        soup = BeautifulSoup(html, 'html.parser')
        
        # Abstract extraction logic
        # ScienceDirect often uses: div class="abstract author" or id="abstracts" or section class="abstract"
        abstract_text = ""
        
        # Method 1: Specific ScienceDirect classes
        abs_div = soup.find('div', class_='abstract') or \
                  soup.find('div', id='abstracts') or \
                  soup.find('section', class_='abstract') or \
                  soup.find('div', class_='author-abstract') # varying Elsevier formats
                  
        if abs_div:
            print("Found abstract container.")
            abstract_text = abs_div.get_text(strip=True)
        else:
            # Method 2: Meta description
            meta_desc = soup.find('meta', attrs={'name': 'description'})
            if meta_desc:
                print("Found meta description.")
                abstract_text = meta_desc.get('content', '')

        if len(abstract_text) > 50:
            print("\n--- Abstract Extracted ---")
            print(abstract_text)
        else:
            print("\n--- Failed to Extract Abstract ---")
            print("Dumping snippet of body text:")
            print(soup.body.get_text(separator=' ', strip=True)[:500] if soup.body else "No body content")

        driver.quit()

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
