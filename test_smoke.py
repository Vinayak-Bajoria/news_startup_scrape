#!/usr/bin/env python3
"""Smoke test all modules and ML functions."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 1. Data loading
from src.data_loader import load_startups, load_mncs, load_news_sources, Company
s = load_startups()
m = load_mncs()
print(f"OK  Startups: {len(s)}, MNCs: {len(m)}")

# 2. Articles
from src.news_scraper import load_articles, Article
arts = load_articles()
total = sum(len(v) for v in arts.values())
print(f"OK  Articles: {total} for {len(arts)} companies")

# 3. Splits
from src.splitter import load_splits
splits = load_splits()
print(f"OK  Splits: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")

# 4. ML results
with open("data/processed/ml_results.json") as f:
    ml = json.load(f)
print(f"OK  ML results: {len(ml.get('companies', {}))} companies, {ml.get('corpus_size', 0)} corpus")

# 5. Test TF-IDF on real data (like Page 3 does)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
zepto_arts = [a.to_dict() if isinstance(a, Article) else a for a in arts.get("Zepto", [])]
texts = [f"{a.get('title','')} {a.get('snippet','')}" for a in zepto_arts]
if texts:
    vec = TfidfVectorizer(max_features=3000, stop_words="english")
    mat = vec.fit_transform(texts)
    qv = vec.transform(["Zepto quick commerce"])
    sims = cosine_similarity(qv, mat).flatten()
    top = sims.argsort()[-3:][::-1]
    print(f"OK  TF-IDF: top 3 scores = {[round(float(sims[i]),3) for i in top]}")

# 6. Test live scrape (like Page 4 does)
from src.news_scraper import scrape_google_news_rss
test_company = Company(name="Zomato", founding_year="2008", sector="foodtech",
                       founders="Deepinder Goyal", description="Food delivery",
                       company_type="search", search_terms=["Zomato"])
live_arts = scrape_google_news_rss(test_company, last_n_days=7)
print(f"OK  Live scrape 'Zomato': {len(live_arts)} articles (7 days)")

print("\n=== ALL TESTS PASSED ===")
print("Pages ready:")
print("  Page 3: /ML_News_Analysis  — Dataset Analysis with company dropdown + timeline")
print("  Page 4: /Live_Search       — Live search any company + ML comparison")
print("\nRun: streamlit run bday_home.py")
