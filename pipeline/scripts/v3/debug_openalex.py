import requests
import re
from urllib.parse import quote

API_URL = "https://api.openalex.org/works"
MAILTO = "mike.mcrae25@gmail.com"
TITLE = 'Vicarious Liability for Independent Contractors: Orthodox Principles "On the Move"?'

def clean_title(title):
    """
    OpenAlex-safe title normalization.
    Removes punctuation that causes backend 500s.
    """
    title = title.lower()
    title = re.sub(r'[":?]', '', title)   # known OpenAlex crashers
    title = re.sub(r'\s+', ' ', title)
    return title.strip()

def query_openalex(title):
    clean = clean_title(title)

    # Progressive fallback: full → truncated
    variants = [
        clean,
        " ".join(clean.split()[:8]),
        " ".join(clean.split()[:5]),
    ]

    for v in variants:
        url = (
            f"{API_URL}"
            f"?filter=title.search:{quote(v)}"
            f"&per-page=5"
            f"&mailto={MAILTO}"
        )

        print(f"\nTrying: {v}")
        print(f"URL: {url}")

        try:
            r = requests.get(url, timeout=15)
            print(f"Status: {r.status_code}")

            if r.status_code == 200:
                data = r.json()
                results = data.get("results", [])
                print(f"Results: {len(results)}")
                if results:
                    print(f"Top hit: {results[0].get('title')}")
                return data

            # If OpenAlex backend crashes, try shorter query
            if r.status_code in (500, 502, 503):
                print("OpenAlex backend error, retrying with shorter title…")
                continue

            # Any other status: stop
            print(f"Body: {r.text}")
            break

        except Exception as e:
            print(f"Request error: {e}")
            break

    print("No successful OpenAlex response.")
    return None

if __name__ == "__main__":
    query_openalex(TITLE)
