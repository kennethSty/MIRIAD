from Bio import Entrez
import requests
from enum import Enum
from typing import List, Dict

from meddxagent.ddxdriver.logger import log

ADDED_EXTRA = 10

class Corpus(Enum):
    PUBMED = "PubMed"
    WIKIPEDIA = "Wikipedia"
    MIRIAD = "Miriad"


def api_search(
    query: str,
    top_k: int = 5,
    corpus_name: str = Corpus.PUBMED.value,
) -> List[Dict[str, str]]:
    if corpus_name == Corpus.PUBMED.value:
        return _search_pubmed(query=query, top_k=top_k)
    elif corpus_name == Corpus.WIKIPEDIA.value:
        return _search_wikipedia(query=query, top_k=top_k)
    elif corpus_name == Corpus.MIRIAD.value:
        return _search_miriad(query=query, top_k=top_k)

def format_search_result(result):
    """Format a single result into a string."""
    if not all(key in result for key in ["title", "content"]):
        log.warning("Result formatted incorrectly, returning nothing")
        log.warning(f"Result {str(result)}")
        exit()
    else:
        return f"Title: {result['title']}\nContent:\n{result['content']}"

def _search_miriad(
    query: str,
    top_k: int = 5,
    email: str = "your.email@example.com",
) -> List[Dict[str, str]]:
    """
    Search Miriad for a keyword and return the formatted question and answer pairs of the top k results.

    Args:
    - query (str): The search query.
    - top_k (int): The number of top results to return.

    Returns:
    - list of dict: A list of dictionaries with 'question' and 'answer' for each result.
    """
    raise NotImplementedError 


def _search_pubmed(
    query: str,
    top_k: int = 5,
    min_abstract_length: int = 100,
    email: str = "your.email@example.com",
) -> List[Dict[str, str]]:
    """
    Search PubMed for a query and return the titles and formatted abstracts of the top k results.
    Skip results with abstracts shorter than a specified length.

    Args:
    - query (str): The search query.
    - top_k (int): The number of top results to return.
    - min_abstract_length (int): The minimum length of the abstract to accept.
    - email (str): Email address to use for Entrez.

    Returns:
    - list of dict: A list of dictionaries with 'title' and 'abstract' for each result.
    """
    Entrez.email = email

    # Modify query to include only free full-text articles
    query = f"{query} AND free full text[sb]"
    # query = f"{query} AND ((pmc cc by-sa license[filter]) OR (pmc cc by license[filter]) OR (pmc cc0 license[filter]))"

    # Search PubMed for the query
    search_handle = Entrez.esearch(db="pubmed", term=query, retmax=top_k + ADDED_EXTRA)
    search_results = Entrez.read(search_handle)
    search_handle.close()
    id_list = search_results["IdList"]

    results = []

    # Fetch details for each article by PubMed ID
    for pmid in id_list:
        if len(results) >= top_k:
            break

        fetch_handle = Entrez.efetch(db="pubmed", id=pmid, rettype="abstract", retmode="xml")
        fetch_results = Entrez.read(fetch_handle)
        fetch_handle.close()

        article_results = fetch_results["PubmedArticle"]
        if not article_results:
            continue

        article = article_results[0]
        title = article["MedlineCitation"]["Article"]["ArticleTitle"]

        if not article or not title:
            continue
        # Extract and preserve all sections of the abstract
        abstract_sections = (
            article["MedlineCitation"]["Article"].get("Abstract", {}).get("AbstractText", [])
        )

        # Construct the full abstract by concatenating sections with their labels
        abstract = ""
        for section in abstract_sections:
            section_label = section.attributes.get("Label", "") if section.attributes else ""
            abstract += f"{section_label}: {section}\n\n" if section_label else f"{section}\n\n"

        # Skip this abstract if it's too short
        if len(abstract) < min_abstract_length:
            continue

        results.append({"title": title.strip(), "content": abstract.strip()})

    return results

def _search_wikipedia(
    query: str, top_k: int = 5, min_summary_length: int = 100
) -> List[Dict[str, str]]:
    """
    Search Wikipedia for a query and return the titles and summaries of the top k results.
    Skips results with summaries shorter than a specified length.

    Args:
        query (str): The search query.
        top_k (int): The number of top results to return.
        min_summary_length (int): The minimum length of the summary to accept.

    Returns:
        List[Dict[str, str]]: A list of dictionaries with 'title' and 'summary' keys.
    """

    url = "https://en.wikipedia.org/w/api.php"
    max_query_length = 300  # Wikipedia's max search length
    headers = {
        "User-Agent": "WikipediaSearchEx/1.0 (contact: example@example.com)"
    }

    # Truncate the query if it's too long
    if len(query) > max_query_length:
        query = query[:max_query_length]
        print(f"Query too long. Truncated to: {query}")

    # Perform the search request
    try:
        search_response = requests.get(
            url,
            params={
                "action": "query",
                "list": "search",
                "srsearch": query,
                "srlimit": top_k + 3,  # fetch extra in case of short summaries
                "format": "json",
            },
            headers=headers,
            timeout=10,
        )
        search_response.raise_for_status()
        search_data = search_response.json()
    except Exception as e:
        print(f"Search request failed: {e}")
        return []

    search_results = search_data.get("query", {}).get("search", [])
    results = []

    # Fetch summaries for each result
    for result in search_results:
        if len(results) >= top_k:
            break

        title = result["title"]

        try:
            page_response = requests.get(
                url,
                params={
                    "action": "query",
                    "prop": "extracts",
                    "titles": title,
                    "exintro": True,
                    "explaintext": True,
                    "format": "json",
                },
                headers=headers,
                timeout=10,
            )
            page_response.raise_for_status()
            page_data = page_response.json()
        except Exception as e:
            print(f"Failed to fetch page '{title}': {e}")
            continue

        page_id = next(iter(page_data["query"]["pages"]))
        page_summary = page_data["query"]["pages"][page_id].get("extract", "")

        # Skip too-short summaries
        if len(page_summary) < min_summary_length:
            continue

        results.append({
            "title": title.strip(),
            "content": page_summary.strip()
        })

    return results

