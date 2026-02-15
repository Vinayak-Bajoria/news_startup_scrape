#!/usr/bin/env python3
"""Run ML pipeline on previously scraped articles."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.pipeline import run_ml_pipeline
from src.news_scraper import load_articles

articles = load_articles()
print(f"Loaded {sum(len(v) for v in articles.values())} articles for {len(articles)} companies")

results = run_ml_pipeline(
    articles,
    run_tfidf=True,
    run_semantic=False,
    run_sentiment=False,
    run_ner=False,
    run_topics=True,
    run_zero_shot=False,
)

print(f"\nDone! Results for {len(results.get('companies', {}))} companies")
