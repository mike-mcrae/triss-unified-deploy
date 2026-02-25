from seleniumbase import SB
from bs4 import BeautifulSoup
from openai import OpenAI
import time
import os

# Configuration
OPENAI_MODEL = "gpt-4o-mini"
TARGET_TITLE = "The I-ASC Explanatory Model as a Support for AAC Assessment Planning A Case Report"

URLS = [
    {
        "source": "NLM",
        "url": "https://catalog.nlm.nih.gov/discovery/fulldisplay/alma9918487681406676/01NLM_INST:01NLM_INST"
    },
    {
        "source": "Routledge",
        "url": "https://www.routledge.com/Clinical-Cases-in-Augmentative-and-Alternative-Communication/Smith/p/book/9780367618285"
    }
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

def get_description_from_llm(client, title, page_text):
    print(f"\n--- CALLING LLM for '{title}' ---")
    
    # MODIFIED PROMPT to be explicit about Book vs Chapter
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
    # Print the "Head" of the content to verify it's getting there
    print(f"--- TEXT SENT TO LLM (First 1500 chars) ---")
    print(user_content[:1500])
    print("... [truncated] ...")
    print(f"--- TEXT SENT TO LLM (Last 500 chars) ---")
    print(user_content[-500:])
    print("-------------------------------------------")

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
        for item in URLS:
            print(f"\n\n=======================================================")
            print(f"PROCESSING: {item['source']}")
            print(f"URL: {item['url']}")
            print(f"=======================================================")
            
            try:
                sb.uc_open_with_reconnect(item['url'], reconnect_time=4)
                
                # NLM specific scroll
                if "nlm.nih.gov" in item['url']:
                    print("  -> Scrolling page for NLM...")
                    sb.execute_script("window.scrollTo(0, 1000);")
                    time.sleep(2)
                
                # Routledge specific scroll
                if "routledge.com" in item['url'] and sb.is_element_present(".accordion-button"):
                     print("  -> Scrolling to accordion...")
                     sb.execute_script("document.querySelector('.accordion-button').scrollIntoView();")
                     time.sleep(1)

                time.sleep(2)
                
                page_source = sb.get_page_source()
                clean_text = clean_html(page_source)
                
                print(f"  -> Extracted Text Length: {len(clean_text)}")
                
                # Check for "Summary" or "Description" keywords manually before LLM
                match_summary = "Summary" in clean_text
                match_desc = "Description" in clean_text
                print(f"  -> Contains 'Summary'? {match_summary}")
                print(f"  -> Contains 'Description'? {match_desc}")
                
                # Call LLM
                result = get_description_from_llm(client, TARGET_TITLE, clean_text)
                print(f"\n>>> LLM RESULT: {result}")
                
            except Exception as e:
                print(f"Error processing {item['source']}: {e}")

if __name__ == "__main__":
    main()
