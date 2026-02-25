import requests
import re
import textwrap

SEMANTIC_SCHOLAR_API_KEY = None  # optional

HEADERS_SS = {
    "User-Agent": "doi-lookup/1.0",
}
if SEMANTIC_SCHOLAR_API_KEY:
    HEADERS_SS["x-api-key"] = SEMANTIC_SCHOLAR_API_KEY


def print_abstract(text, max_chars=1500):
    if not text:
        print("  (no abstract)")
        return
    print("\n" + "-" * 80)
    print(textwrap.fill(text[:max_chars], width=100))
    if len(text) > max_chars:
        print("\n  [truncated]")
    print("-" * 80)


def crossref_by_doi(doi):
    r = requests.get(f"https://api.crossref.org/works/{doi}", timeout=15)
    if r.status_code != 200:
        return None
    return r.json()["message"]


def crossref_by_title(title, author=None):
    params = {"query.title": title, "rows": 3}
    if author:
        params["query.author"] = author
    r = requests.get("https://api.crossref.org/works", params=params, timeout=15)
    if r.status_code != 200:
        return []
    return r.json()["message"]["items"]


def openalex_by_doi(doi):
    r = requests.get(f"https://api.openalex.org/works/https://doi.org/{doi}", timeout=15)
    if r.status_code != 200:
        return None
    return r.json()


def openalex_by_title(title):
    r = requests.get(
        "https://api.openalex.org/works",
        params={"search": title, "per-page": 3},
        timeout=15,
    )
    if r.status_code != 200:
        return []
    return r.json()["results"]


def semantic_scholar_by_doi(doi):
    url = f"https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}"
    r = requests.get(url, headers=HEADERS_SS,
                     params={"fields": "title,abstract,url"}, timeout=15)
    if r.status_code != 200:
        return None
    return r.json()


def semantic_scholar_by_title(title):
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    r = requests.get(
        url,
        headers=HEADERS_SS,
        params={"query": title, "limit": 3, "fields": "title,abstract,url"},
        timeout=15,
    )
    if r.status_code != 200:
        return []
    return r.json()["data"]


# -------------------------
# Interactive CLI
# -------------------------
print("\nChoose input type:")
print("1 = DOI")
print("2 = Title + Author")
print("3 = Title only")

choice = input("Enter choice (1/2/3): ").strip()

doi = None
title = None
author = None

if choice == "1":
    doi = input("Enter DOI: ").strip()

elif choice == "2":
    title = input("Enter title: ").strip()
    author = input("Enter author (last name ok): ").strip()

elif choice == "3":
    title = input("Enter title: ").strip()

else:
    print("Invalid choice.")
    exit()


print("\n" + "=" * 80)

# -------------------------
# Crossref
# -------------------------
print("\nCROSSREF")
if doi:
    cr = crossref_by_doi(doi)
    if cr and cr.get("abstract"):
        print(f"DOI: {cr.get('DOI')}")
        print_abstract(cr.get("abstract"))
    else:
        print("No abstract found.")
else:
    results = crossref_by_title(title, author)
    if results:
        for i, r in enumerate(results, 1):
            print(f"\nMatch {i}: DOI {r.get('DOI')}")
            print_abstract(r.get("abstract"))
    else:
        print("No matches.")

# -------------------------
# OpenAlex
# -------------------------
print("\nOPENALEX")
if doi:
    oa = openalex_by_doi(doi)
    if oa and oa.get("abstract"):
        print(f"DOI: {oa.get('doi')}")
        print_abstract(oa.get("abstract"))
    else:
        print("No abstract found.")
else:
    results = openalex_by_title(title)
    if results:
        for i, r in enumerate(results, 1):
            print(f"\nMatch {i}: DOI {r.get('doi')}")
            print_abstract(r.get("abstract"))
    else:
        print("No matches.")

# -------------------------
# Semantic Scholar
# -------------------------
print("\nSEMANTIC SCHOLAR")
if doi:
    ss = semantic_scholar_by_doi(doi)
    if ss and ss.get("abstract"):
        print(f"URL: {ss.get('url')}")
        print_abstract(ss.get("abstract"))
    else:
        print("No abstract found.")
else:
    results = semantic_scholar_by_title(title)
    if results:
        for i, r in enumerate(results, 1):
            print(f"\nMatch {i}: URL {r.get('url')}")
            print_abstract(r.get("abstract"))
    else:
        print("No matches.")

print("\nDone.\n")
