from seleniumbase import SB
import time

def main():
    with SB(uc=True, test=True, headless=False) as sb:
        print("Opening Google...")
        sb.uc_open_with_reconnect("https://www.google.com/search?q=test+search+query", reconnect_time=4)
        
        # Handle Consent
        if sb.is_element_visible("button[aria-label='Accept all']"):
            sb.click("button[aria-label='Accept all']")
        elif sb.is_element_visible("div.QS5gu.sy4vM"): 
            sb.click("div.QS5gu.sy4vM", delay=0.5)

        time.sleep(3)
        
        print("\n--- Testing Selectors ---")
        
        # Selector 1: Standard div.g a
        links1 = sb.find_elements("div.g a")
        print(f"Selector 'div.g a' found {len(links1)} elements.")
        for i, el in enumerate(links1[:3]):
            print(f"  {i}: {el.get_attribute('href')}")

        # Selector 2: h3 parent
        # Find h3, then go up to 'a'
        # xpath: //h3/parent::a
        links2 = sb.find_elements("xpath://h3/parent::a")
        print(f"Selector '//h3/parent::a' found {len(links2)} elements.")
        for i, el in enumerate(links2[:3]):
            print(f"  {i}: {el.get_attribute('href')}")
            
        # Selector 3: div#search a
        links3 = sb.find_elements("div#search a")
        print(f"Selector 'div#search a' found {len(links3)} elements.")
        # Filter for h3 children? 
        valid_links = []
        for el in links3:
            href = el.get_attribute('href')
            if href and 'google' not in href and href.startswith('http'):
                # Check if it contains h3?
                try:
                    el.find_element(by='tag name', value='h3')
                    valid_links.append(href)
                except:
                    pass
        print(f"  Filtered (has h3 child): {len(valid_links)}")
        for i, l in enumerate(valid_links[:3]):
            print(f"  {i}: {l}")

if __name__ == "__main__":
    main()
