from seleniumbase import SB
import time
from bs4 import BeautifulSoup

def main():
    url = "https://doi.org/10.1016/j.intfin.2023.101882"
    print(f"Testing URL (SeleniumBase UC): {url}")

    # uc=True enables Undetected-Chromedriver mode
    # test=True allows for some debug info if needed, but we use context manager
    # headless=False is usually required for UC mode to be effective against tough bots, 
    # but we can try headless=True if on a server. However, user is on Mac local, so visible is fine/better for testing.
    # We'll use visible mode to see if it works, then toggle headless.
    
    with SB(uc=True, test=True, headless=False) as sb:
        print("Navigating with reconnect...")
        # uc_open_with_reconnect is a helper to bypass initial detection
        sb.uc_open_with_reconnect(url, reconnect_time=4)
        
        print("Handling potential CAPTCHA...")
        # This attempts to click the Cloudflare/Turnstile CAPTCHA if found
        try:
            sb.uc_gui_click_captcha()
        except:
            print("No CAPTCHA found or already solved.")

        print("Waiting for page load...")
        time.sleep(5) 
        
        # Check if we are still blocked
        title = sb.get_title()
        print(f"Page Title: {title}")
        
        if "Just a moment" in title or "Robot" in title:
             print("Still blocked. Trying one more CAPTCHA click...")
             sb.uc_gui_click_captcha()
             time.sleep(5)

        html = sb.get_page_source()
        soup = BeautifulSoup(html, 'html.parser')
        
        # Abstract extraction (Reuse logic)
        abstract_text = ""
        abs_div = soup.find('div', class_='abstract') or \
                  soup.find('div', id='abstracts') or \
                  soup.find('section', class_='abstract') or \
                  soup.find('div', class_='author-abstract')
                  
        if abs_div:
            print("Found abstract container.")
            abstract_text = abs_div.get_text(strip=True)
        else:
            meta_desc = soup.find('meta', attrs={'name': 'description'})
            if meta_desc:
                print("Found meta description.")
                abstract_text = meta_desc.get('content', '')

        if len(abstract_text) > 50:
            print("\n--- Abstract Extracted ---")
            print(abstract_text[:500] + "...")
        else:
            print("\n--- Failed to Extract Abstract ---")
            print("Snippet:", soup.body.get_text(separator=' ', strip=True)[:300] if soup.body else "No body")

if __name__ == "__main__":
    main()
