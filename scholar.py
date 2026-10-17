"""
scholar.py - make sure every search includes some Google Scholar pages.

How it works:
  1. Runs a normal web search AND a Google Scholar search (both via Serper).
  2. Ranks everything together with your algorithm.rank_results().
  3. If fewer than `min_scholar` Scholar pages made the final list, the
     best-matching Scholar pages are swapped in for the lowest-ranked web
     results, so each search shows at least 1-2 academic sources.

Each Scholar result also carries its year, publication info and citation
count, which you can display as a credibility signal.

Cost: each search now uses 2 Serper queries (one web, one Scholar).

Usage (same folder as algorithm.py):
    export SERPER_API_KEY="your_key"
    python3 scholar.py "school uniforms should be compulsory"
    python3 scholar.py "global warming" --min-scholar 1 --limit 8

Use it from other code:
    from scholar import search_with_scholar
    results, info = search_with_scholar("global warming")
"""
import argparse
import sys
from urllib.parse import urlparse

import requests

import algorithm

SCHOLAR_URL = "https://google.serper.dev/scholar"


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _is_scholar(page):
    return page.get("source") == "scholar"


def _norm_url(url):
    return (url or "").strip().lower().rstrip("/")


def _domain(url):
    domain = urlparse(url or "").netloc.lower()
    return domain[4:] if domain.startswith("www.") else domain


# --------------------------------------------------------------------------
# Google Scholar search (Serper /scholar endpoint)
# --------------------------------------------------------------------------
def search_scholar(query, count=10):
    """Return Google Scholar results in the same dict shape as search_web()."""
    if not algorithm.SERPER_API_KEY:
        raise RuntimeError(
            "Missing API key. Set SERPER_API_KEY in your terminal "
            "before running the program."
        )

    try:
        count = max(1, min(int(count), 20))
    except (TypeError, ValueError):
        count = 10

    response = requests.post(
        SCHOLAR_URL,
        headers={
            "X-API-KEY": algorithm.SERPER_API_KEY,
            "Content-Type": "application/json",
        },
        json={"q": query, "gl": "gb", "hl": "en"},
        timeout=20,
    )
    if not response.ok:
        raise RuntimeError(
            f"Serper Scholar API error {response.status_code}: {response.text}"
        )

    data = response.json()

    pages = []
    for item in data.get("organic") or []:
        if len(pages) >= count:
            break
        url = item.get("link") or ""
        if not url:
            continue

        publication = algorithm._clean(item.get("publicationInfo"))
        snippet = algorithm._clean(item.get("snippet")) or publication
        year = item.get("year")

        pages.append({
            "title": algorithm._clean(item.get("title")),
            "snippet": snippet,
            "url": url,
            "domain": _domain(url),
            # publicationInfo = authors, journal, year, publisher
            "publisher": publication,
            "published": str(year) if year else "",
            "underexposure": 0.0,
            # Scholar-specific fields
            "source": "scholar",
            "cited_by": item.get("citedBy"),
            "year": year,
            "publication_info": publication,
        })

    return pages


# --------------------------------------------------------------------------
# Guarantee a minimum number of Scholar pages
# --------------------------------------------------------------------------
def _ensure_scholar(query, ranked, scholar_pages, limit, min_scholar,
                    relaxed_relevance, relaxed_quality):
    """Swap in the best remaining Scholar pages until min_scholar is met."""
    need = min_scholar - sum(1 for p in ranked if _is_scholar(p))
    if need <= 0 or not scholar_pages:
        return ranked

    used = {_norm_url(p.get("url")) for p in ranked}
    pool = [p for p in scholar_pages if _norm_url(p.get("url")) not in used]
    if not pool:
        return ranked

    # Rank the leftover Scholar pages with slightly relaxed thresholds, so a
    # short academic snippet doesn't stop us from showing any source at all.
    extras = algorithm.rank_results(
        query, pool, limit=need,
        min_relevance=relaxed_relevance, min_quality=relaxed_quality,
    )
    if not extras:
        return ranked
    for page in extras:
        page["forced_scholar"] = True

    result = list(ranked)

    # Make room by dropping the lowest-ranked non-Scholar results.
    to_remove = max(0, len(result) + len(extras) - limit)
    i = len(result) - 1
    while to_remove > 0 and i >= 0:
        if not _is_scholar(result[i]):
            result.pop(i)
            to_remove -= 1
        i -= 1

    return result + extras


def search_with_scholar(
    query,
    count=20,
    limit=10,
    min_scholar=2,
    scholar_count=10,
    scholar_min_relevance=0.20,
    scholar_min_quality=0.40,
    **rank_kwargs,
):
    """
    Web search + Google Scholar search, ranked together, with at least
    `min_scholar` Scholar pages in the final list when any are available.

    Returns (results, info). `info` explains what happened, e.g. a warning if
    Scholar could not be reached or returned too few pages.
    """
    try:
        limit = max(1, min(int(limit), 50))
    except (TypeError, ValueError):
        limit = 10
    min_scholar = max(0, min(int(min_scholar), limit))

    info = {"warning": None, "scholar_fetched": 0, "scholar_in_results": 0}

    web_pages = algorithm.search_web(query, count=count)

    scholar_pages = []
    if min_scholar > 0:
        try:
            scholar_pages = search_scholar(query, count=scholar_count)
        except (requests.RequestException, RuntimeError) as error:
            info["warning"] = f"Google Scholar search failed: {error}"
    info["scholar_fetched"] = len(scholar_pages)

    # Merge, preferring the Scholar version when a URL appears in both.
    scholar_urls = {_norm_url(p["url"]) for p in scholar_pages}
    merged = scholar_pages + [
        p for p in web_pages if _norm_url(p.get("url")) not in scholar_urls
    ]

    ranked = algorithm.rank_results(query, merged, limit=limit, **rank_kwargs)
    final = _ensure_scholar(
        query, ranked, scholar_pages, limit, min_scholar,
        scholar_min_relevance, scholar_min_quality,
    )

    info["scholar_in_results"] = sum(1 for p in final if _is_scholar(p))
    if (info["warning"] is None and min_scholar
            and info["scholar_in_results"] < min_scholar):
        info["warning"] = (
            f"Only {info['scholar_in_results']} Google Scholar page(s) "
            f"available (wanted {min_scholar})."
        )

    return final, info


# --------------------------------------------------------------------------
# Command-line output
# --------------------------------------------------------------------------
def _short(text, n):
    text = " ".join((text or "").split())
    return text if len(text) <= n else text[: n - 1] + "…"


def print_results(query, results, info, limit):
    line = "=" * 90
    print(f"\n{line}")
    print(f"Query: {query}")
    print(f"{len(results)} results, {info['scholar_in_results']} from Google Scholar")
    print(line)

    if info["warning"]:
        print(f"Note: {info['warning']}\n")

    if not results:
        print("No pages passed the filters. Try lowering min_relevance/min_quality.")
        return

    for i, page in enumerate(results, start=1):
        tag = "[Scholar]" if _is_scholar(page) else "         "
        print(f"{i:>2}. {tag} {_short(page['title'], 70)}")
        print(f"               {page['url']}")
        if _is_scholar(page):
            bits = []
            if page.get("publication_info"):
                bits.append(_short(page["publication_info"], 70))
            if page.get("cited_by") is not None:
                bits.append(f"cited by {page['cited_by']}")
            if bits:
                print(f"               {' | '.join(bits)}")
        print(f"               relevance {page['relevance']:.2f}"
              f"{'  (added to guarantee a Scholar source)' if page.get('forced_scholar') else ''}")
    print(f"{line}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Search the web and guarantee some Google Scholar sources."
    )
    parser.add_argument("query", nargs="*", help="search query")
    parser.add_argument("--limit", type=int, default=10, help="results to show (default 10)")
    parser.add_argument("--min-scholar", type=int, default=2,
                        help="minimum Google Scholar pages (default 2)")
    args = parser.parse_args()

    query = " ".join(args.query).strip() or input(
        "What would you like to search for? "
    ).strip()
    if not query:
        print("Please enter a search query.")
        return 1

    try:
        results, info = search_with_scholar(
            query, limit=args.limit, min_scholar=args.min_scholar
        )
    except requests.RequestException as error:
        print(f"Search API request failed: {error}")
        return 1
    except (ValueError, RuntimeError) as error:
        print(f"Error: {error}")
        return 1

    print_results(query, results, info, args.limit)
    return 0


if __name__ == "__main__":
    sys.exit(main())