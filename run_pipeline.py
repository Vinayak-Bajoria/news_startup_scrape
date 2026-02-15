#!/usr/bin/env python3
"""Run the full scrape + split + ML pipeline with incremental saves."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from src.data_loader import load_all_companies, load_news_sources, save_company_registry
from src.news_scraper import scrape_company, save_articles, Article
from src.splitter import split_companies, make_cv_folds, save_splits, print_split_summary

# Step 1: Split
companies = load_all_companies()
train, val, test = split_companies(companies)
cv_folds = make_cv_folds(train)
save_splits(train, val, test, cv_folds)
save_company_registry(companies)
print_split_summary(train, val, test, cv_folds)

# Step 2: Scrape ALL companies (with incremental save)
# Load existing results if any (resume support)
cache_path = os.path.join(config.RAW_DIR, "scraped_articles.json")
results = {}
if os.path.exists(cache_path):
    with open(cache_path) as f:
        cached = json.load(f)
    for name, arts in cached.items():
        results[name] = [Article(**a) for a in arts]
    print(f"\nResuming: {len(results)} companies already scraped")

sources = load_news_sources()
total = len(companies)
for i, company in enumerate(companies, 1):
    if company.name in results:
        print(f"[{i}/{total}] SKIP (cached): {company.name} ({len(results[company.name])} articles)")
        continue
    print(f"\n[{i}/{total}] Scraping: {company.name} ({company.sector})")
    arts = scrape_company(company, sources, use_bing=False, use_sources=False)
    results[company.name] = arts
    print(f"  → Found {len(arts)} articles")
    # Save incrementally every 5 companies
    if i % 5 == 0:
        save_articles(results)

# Final save
save_articles(results)

# Summary
total_arts = sum(len(v) for v in results.values())
zero_count = sum(1 for v in results.values() if len(v) == 0)
print(f"\nSummary: {total_arts} articles for {len(results)} companies")
print(f"Companies with 0 articles: {zero_count}")
for name, arts in sorted(results.items(), key=lambda x: len(x[1]), reverse=True):
    print(f"  {name:30s}: {len(arts):3d} articles")
