import requests
from urllib.parse import quote

API_URL = "https://api.openalex.org/works"
MAILTO = "mike.mcrae25@gmail.com"
TITLE = "Occipital-temporal cortical tuning to semantic and affective features of natural images predicts associated behavioral responses"

def test():
    # Use filter instead of search
    query = quote(TITLE)
    url = f"{API_URL}?filter=title.search:{query}&per-page=5&mailto={MAILTO}"
    print(f"URL: {url}")
    
    try:
        response = requests.get(url, timeout=10)
        print(f"Status: {response.status_code}")
        print(f"Headers: {response.headers}")
        
        limit = response.headers.get('X-RateLimit-Limit', 'Unknown')
        remaining = response.headers.get('X-RateLimit-Remaining', 'Unknown')
        credits_used = response.headers.get('X-RateLimit-Credits-Used', 'Unknown')
        credits_required = response.headers.get('X-RateLimit-Credits-Required', 'Unknown')
        
        print(f"Limit: {limit}")
        print(f"Remaining: {remaining}")
        print(f"Credits Used: {credits_used}")
        print(f"Credits Required: {credits_required}")
        
        if response.status_code != 200:
            print(f"Body: {response.text}")
        else:
            data = response.json()
            print(f"Results: {len(data.get('results', []))}")
            if data.get('results'):
                print(f"Top: {data['results'][0].get('title')}")
                
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test()
