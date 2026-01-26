import requests
from collections import defaultdict
import time
from config import API_KEY

def search_semantic_scholar(query, coarse_domain, limit=5, year=None):
    url = "http://api.semanticscholar.org/graph/v1/snippet/search"
    # Valid Semantic Scholar domains
    valid_domains = {
        "Computer Science", "Medicine", "Chemistry", "Biology", "Materials Science",
        "Physics", "Geology", "Psychology", "Art", "History", "Geography",
        "Sociology", "Business", "Political Science", "Economics", "Philosophy",
        "Mathematics", "Engineering", "Environmental Science",
        "Agricultural and Food Sciences", "Education", "Law", "Linguistics"
    }
    query_params = {"query": query, "limit": limit}

    if coarse_domain in valid_domains:
        query_params["fieldsOfStudy"] = coarse_domain
    if year is not None:
        query_params["year"] = f"-{year-1}"

    headers = {"x-api-key": API_KEY}
    
    print(f"\t -Searching Semantic Scholar for query: {query} in domain: {coarse_domain}.")

    while True:
        response = requests.get(url, params=query_params, headers=headers, allow_redirects=True)
        
        if response.status_code == 200:
            break
        elif response.status_code == 429:
            retry_after = int(response.headers.get('Retry-After', 5))
            print(f"Rate limited. Retrying after {retry_after} seconds...")
            time.sleep(retry_after)
        else:
            response.raise_for_status()
    
    if response.status_code == 200:
        return_response = response.json()['data']
        if len(return_response) == 0:
            print(f"\t -No results found for query: {query} in domain: {coarse_domain}.")
        return return_response
    else:
        response.raise_for_status()

def fetch_paper_details(paper_id):
    url = f"http://api.semanticscholar.org/graph/v1/paper/CorpusId:{paper_id}"
    params = {
        "fields": "title,abstract"
    }
    headers = {"x-api-key": API_KEY}
    
    while True:
        response = requests.get(url, params=params, headers=headers, allow_redirects=True)
        
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 429:
            retry_after = int(response.headers.get('Retry-After', 10))
            print(f"Rate limited. Retrying after {retry_after} seconds...")
            time.sleep(retry_after)
        else:
            response.raise_for_status()

def collect_snippets(response):
    snippets = defaultdict(list)
    for item in response:
        title = item.get('paper', {}).get('title', '')
        text = item.get('snippet', {}).get('text', '')
        if title.strip() == text.strip():
            paper_info = fetch_paper_details(item.get('paper', {}).get('corpusId', ''))
            text = paper_info.get('abstract', '')
        
        if text:
            snippets[title].append(text.strip())
        else:
            continue
    
    return snippets