"""
News scraper for collecting articles about companies.

Strategy (tiered, all free):
  1. Google News RSS — reliable, no API key, returns ~10-20 results per query
  2. Bing News RSS   — backup feed
  3. Direct scraping  — headline extraction from the 40 curated sources

Each article is stored as a dict with:
  - title, snippet, source, url, published_date, company_name, search_term
"""
import feedparser
import requests
import time
import random
import hashlib
import json
import os
import re
from datetime import datetime, timedelta
from urllib.parse import quote_plus, urljoin
from typing import List, Dict, Optional
from bs4 import BeautifulSoup
from dataclasses import dataclass, asdict, field
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.data_loader import Company, NewsSource


# ── Article data class ─────────────────────────────────────────
@dataclass
class Article:
    title: str
    snippet: str
    source: str
    url: str
    published_date: str
    company_name: str
    search_term: str
    scrape_method: str          # "google_rss", "bing_rss", "direct_scrape"
    article_id: str = ""

    def __post_init__(self):
        if not self.article_id:
            raw = f"{self.url}{self.title}"
            self.article_id = hashlib.md5(raw.encode()).hexdigest()

    def to_dict(self):
        return asdict(self)


# ── Google News RSS Scraper ────────────────────────────────────
def _fetch_rss_with_timeout(url: str, timeout: int = None) -> object:
    """Fetch RSS feed with a proper timeout (feedparser has no timeout support)."""
    timeout = timeout or config.REQUEST_TIMEOUT
    try:
        resp = requests.get(
            url,
            timeout=(5, timeout),  # (connect_timeout, read_timeout)
            headers={"User-Agent": config.USER_AGENT},
        )
        resp.raise_for_status()
        # Pass text content (not bytes) so feedparser doesn't re-fetch the URL
        return feedparser.parse(resp.text)
    except Exception as e:
        print(f"    [TIMEOUT/ERR] {e}")
        result = feedparser.util.FeedParserDict()
        result['entries'] = []
        return result


def scrape_google_news_rss(
    company: Company,
    last_n_days: int = None,
) -> List[Article]:
    """
    Fetch articles via Google News RSS for each search term of a company.
    """
    last_n_days = last_n_days or config.SCRAPE_LAST_N_DAYS
    articles = []
    seen_urls = set()

    for term in company.search_terms:
        query = quote_plus(f"{term} startup India")
        url = (
            f"https://news.google.com/rss/search?"
            f"q={query}+when:{last_n_days}d&hl=en-IN&gl=IN&ceid=IN:en"
        )
        try:
            feed = _fetch_rss_with_timeout(url)
            for entry in feed.entries[:config.MAX_ARTICLES_PER_COMPANY]:
                link = entry.get("link", "")
                if link in seen_urls:
                    continue
                seen_urls.add(link)

                # Parse date
                pub_date = ""
                if hasattr(entry, "published_parsed") and entry.published_parsed:
                    pub_date = time.strftime("%Y-%m-%d", entry.published_parsed)

                # Extract source from title (Google News format: "Title - Source")
                title = entry.get("title", "")
                source_name = "Google News"
                if " - " in title:
                    parts = title.rsplit(" - ", 1)
                    title = parts[0].strip()
                    source_name = parts[1].strip()

                articles.append(Article(
                    title=title,
                    snippet=_clean_html(entry.get("summary", "")),
                    source=source_name,
                    url=link,
                    published_date=pub_date,
                    company_name=company.name,
                    search_term=term,
                    scrape_method="google_rss",
                ))
        except Exception as e:
            print(f"  [WARN] Google RSS failed for '{term}': {e}")

        _random_delay()

    return articles


# ── Bing News RSS Scraper ──────────────────────────────────────
def scrape_bing_news_rss(company: Company) -> List[Article]:
    """Bing News RSS as a backup source."""
    articles = []
    seen_urls = set()

    for term in company.search_terms:
        query = quote_plus(f"{term} startup India")
        url = f"https://www.bing.com/news/search?q={query}&format=rss"
        try:
            feed = _fetch_rss_with_timeout(url)
            for entry in feed.entries[:30]:
                link = entry.get("link", "")
                if link in seen_urls:
                    continue
                seen_urls.add(link)

                pub_date = ""
                if hasattr(entry, "published_parsed") and entry.published_parsed:
                    pub_date = time.strftime("%Y-%m-%d", entry.published_parsed)

                articles.append(Article(
                    title=entry.get("title", ""),
                    snippet=_clean_html(entry.get("summary", "")),
                    source=entry.get("source", {}).get("title", "Bing News"),
                    url=link,
                    published_date=pub_date,
                    company_name=company.name,
                    search_term=term,
                    scrape_method="bing_rss",
                ))
        except Exception as e:
            print(f"  [WARN] Bing RSS failed for '{term}': {e}")

        _random_delay()

    return articles


# ── Direct Source Scraper ──────────────────────────────────────
def scrape_curated_source(
    company: Company,
    source: NewsSource,
) -> List[Article]:
    """
    Try to find articles about a company on a specific curated news source.
    Uses site-specific Google search as the most reliable approach.
    """
    articles = []
    term = company.name

    # Use Google site-search to find articles on this specific source
    domain = source.url.replace("https://", "").replace("http://", "").split("/")[0]
    query = quote_plus(f"site:{domain} {term}")
    rss_url = (
        f"https://news.google.com/rss/search?"
        f"q={query}+when:{config.SCRAPE_LAST_N_DAYS}d&hl=en-IN&gl=IN&ceid=IN:en"
    )

    try:
        feed = _fetch_rss_with_timeout(rss_url)
        for entry in feed.entries[:10]:
            pub_date = ""
            if hasattr(entry, "published_parsed") and entry.published_parsed:
                pub_date = time.strftime("%Y-%m-%d", entry.published_parsed)

            title = entry.get("title", "")
            if " - " in title:
                title = title.rsplit(" - ", 1)[0].strip()

            articles.append(Article(
                title=title,
                snippet=_clean_html(entry.get("summary", "")),
                source=source.name,
                url=entry.get("link", ""),
                published_date=pub_date,
                company_name=company.name,
                search_term=term,
                scrape_method="direct_scrape",
            ))
    except Exception as e:
        print(f"  [WARN] Source scrape failed for {source.name}/{term}: {e}")

    return articles


# ── Master scrape function ─────────────────────────────────────
def scrape_company(
    company: Company,
    sources: List[NewsSource] = None,
    use_google: bool = True,
    use_bing: bool = False,
    use_sources: bool = False,  # disabled by default (slow for 50 × 40)
) -> List[Article]:
    """
    Run all scraping methods for a single company.
    Returns de-duplicated articles.
    """
    all_articles: List[Article] = []

    if use_google:
        print(f"  [Google RSS] {company.name}...")
        all_articles.extend(scrape_google_news_rss(company))

    if use_bing:
        print(f"  [Bing RSS]   {company.name}...")
        all_articles.extend(scrape_bing_news_rss(company))

    if use_sources and sources:
        # Only scrape top 5 most relevant sources (by sector match)
        relevant = _match_sources_to_sector(company.sector, sources)[:5]
        for src in relevant:
            print(f"  [Source]     {company.name} @ {src.name}...")
            all_articles.extend(scrape_curated_source(company, src))
            _random_delay()

    # De-duplicate by title similarity
    return _deduplicate(all_articles)


def scrape_all_companies(
    companies: List[Company],
    sources: List[NewsSource] = None,
    **kwargs,
) -> Dict[str, List[Article]]:
    """
    Scrape news for all companies. Returns {company_name: [articles]}.
    """
    results = {}
    total = len(companies)

    for i, company in enumerate(companies, 1):
        print(f"\n[{i}/{total}] Scraping: {company.name} ({company.sector})")
        arts = scrape_company(company, sources, **kwargs)
        results[company.name] = arts
        print(f"  → Found {len(arts)} articles")

    return results


# ── Persistence ────────────────────────────────────────────────
def save_articles(
    results: Dict[str, List[Article]],
    path: str = None,
):
    """Save scraped articles to JSON."""
    path = path or os.path.join(config.RAW_DIR, "scraped_articles.json")
    data = {
        name: [a.to_dict() for a in arts]
        for name, arts in results.items()
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    total = sum(len(v) for v in data.values())
    print(f"\nSaved {total} articles for {len(data)} companies → {path}")
    return path


def load_articles(path: str = None) -> Dict[str, List[Article]]:
    """Load previously scraped articles."""
    path = path or os.path.join(config.RAW_DIR, "scraped_articles.json")
    with open(path) as f:
        data = json.load(f)
    return {
        name: [Article(**a) for a in arts]
        for name, arts in data.items()
    }


# ── Helpers ────────────────────────────────────────────────────
def _clean_html(text: str) -> str:
    """Strip HTML tags from RSS snippets."""
    if not text:
        return ""
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text(separator=" ", strip=True)


def _random_delay():
    """Respectful delay between requests."""
    lo, hi = config.REQUEST_DELAY
    time.sleep(random.uniform(lo, hi))


def _deduplicate(articles: List[Article]) -> List[Article]:
    """Remove duplicate articles based on URL and title hash."""
    seen = set()
    unique = []
    for a in articles:
        key = a.article_id
        if key not in seen:
            seen.add(key)
            unique.append(a)
    return unique


def _match_sources_to_sector(sector: str, sources: List[NewsSource]) -> List[NewsSource]:
    """Heuristically rank sources by relevance to a sector."""
    sector_lower = sector.lower()
    scored = []
    for src in sources:
        focus_lower = src.focus.lower()
        score = 0
        # Exact sector word match
        for word in sector_lower.split():
            if len(word) > 3 and word in focus_lower:
                score += 2
        # General startup/tech sources always relevant
        if any(kw in focus_lower for kw in ["startup", "funding", "venture", "tech"]):
            score += 1
        scored.append((score, src))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [s for _, s in scored]


# ── CLI entry point ────────────────────────────────────────────
if __name__ == "__main__":
    from src.data_loader import load_startups, load_mncs, load_news_sources

    companies = load_startups() + load_mncs()
    sources = load_news_sources()

    # Quick test with just 2 companies
    test_companies = companies[:2]
    results = scrape_all_companies(test_companies, sources)
    save_articles(results)
