"""
End-to-end pipeline: scrape → split → run ML models → save results.

Can be run as a CLI script or imported by the Streamlit app.
"""
import os, json, time, sys
import numpy as np
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.data_loader import (
    load_startups, load_mncs, load_news_sources,
    load_all_companies, save_company_registry,
)
from src.news_scraper import scrape_all_companies, save_articles, load_articles, Article
from src.splitter import (
    split_companies, make_cv_folds, save_splits,
    load_splits, print_split_summary,
)
from src.ml_models import (
    TFIDFSearcher, SemanticSearcher, TopicModeler,
    analyze_sentiment, aggregate_sentiment,
    extract_entities, aggregate_entities,
    zero_shot_classify,
    compare_algorithms_for_company,
)


def run_scrape_pipeline(
    scrape: bool = True,
    use_bing: bool = True,
    use_sources: bool = False,
) -> Dict[str, List[Article]]:
    """Step 1: Collect news articles for all companies."""
    print("\n" + "=" * 60)
    print("STEP 1: NEWS COLLECTION")
    print("=" * 60)

    if not scrape and os.path.exists(os.path.join(config.RAW_DIR, "scraped_articles.json")):
        print("Loading previously scraped articles...")
        return load_articles()

    companies = load_all_companies()
    sources = load_news_sources()
    results = scrape_all_companies(
        companies, sources,
        use_bing=use_bing, use_sources=use_sources,
    )
    save_articles(results)
    return results


def run_split_pipeline():
    """Step 2: Split companies into train/val/test."""
    print("\n" + "=" * 60)
    print("STEP 2: TRAIN / VAL / TEST SPLIT")
    print("=" * 60)

    companies = load_all_companies()
    train, val, test = split_companies(companies)
    cv_folds = make_cv_folds(train)
    print_split_summary(train, val, test, cv_folds)
    save_splits(train, val, test, cv_folds)
    save_company_registry(companies)
    return train, val, test, cv_folds


def run_ml_pipeline(
    articles_by_company: Dict[str, list],
    run_tfidf: bool = True,
    run_semantic: bool = False,     # set False by default (needs sentence-transformers)
    run_sentiment: bool = False,    # set False by default (needs transformers)
    run_ner: bool = False,
    run_topics: bool = True,
    run_zero_shot: bool = False,
) -> dict:
    """
    Step 3: Run ML algorithms on collected articles.
    Returns a results dict with per-company and aggregate findings.
    """
    print("\n" + "=" * 60)
    print("STEP 3: ML ANALYSIS")
    print("=" * 60)

    # Flatten all articles into a single corpus
    all_articles = []
    for company_name, arts in articles_by_company.items():
        for a in arts:
            if isinstance(a, Article):
                a = a.to_dict()
            all_articles.append(a)

    if not all_articles:
        print("No articles found. Generating synthetic dataset for demo...")
        all_articles = _generate_synthetic_articles(articles_by_company)

    texts = [f"{a['title']} {a.get('snippet', '')}" for a in all_articles]
    print(f"Total articles in corpus: {len(all_articles)}")

    results = {"corpus_size": len(all_articles), "companies": {}}

    # ── 1. TF-IDF ──────────────────────────────────────────────
    tfidf_searcher = None
    if run_tfidf and all_articles:
        print("\n[1/6] TF-IDF Keyword Search...")
        tfidf_searcher = TFIDFSearcher()
        tfidf_searcher.fit(all_articles)
        tfidf_searcher.save()
        results["tfidf_top_terms"] = tfidf_searcher.get_top_terms(20)
        print(f"  Vocabulary size: {len(tfidf_searcher.vectorizer.vocabulary_)}")

    # ── 2. Semantic Search ─────────────────────────────────────
    semantic_searcher = None
    if run_semantic and all_articles:
        print("\n[2/6] Semantic Search (Sentence-Transformers)...")
        semantic_searcher = SemanticSearcher()
        semantic_searcher.fit(all_articles)
        semantic_searcher.save()

    # ── 5. Topic Modeling ──────────────────────────────────────
    topic_modeler = None
    if run_topics and len(all_articles) >= 20:
        print("\n[5/6] Topic Modeling (LDA)...")
        topic_modeler = TopicModeler()
        topic_modeler.fit(texts)
        topic_modeler.save()
        topics = topic_modeler.get_topics(10)
        results["topics"] = {
            f"topic_{i}": words for i, words in enumerate(topics)
        }
        print(f"  Discovered {len(topics)} topics")
        for i, words in enumerate(topics):
            print(f"    Topic {i}: {', '.join(words[:5])}")

    # ── Per-company analysis ───────────────────────────────────
    for company_name, arts in articles_by_company.items():
        if not arts:
            results["companies"][company_name] = {
                "article_count": 0, "status": "no_articles"
            }
            continue

        company_arts = [a.to_dict() if isinstance(a, Article) else a for a in arts]
        company_texts = [f"{a['title']} {a.get('snippet', '')}" for a in company_arts]
        company_result = {"article_count": len(company_arts)}

        # TF-IDF retrieval test
        if tfidf_searcher:
            tfidf_hits = tfidf_searcher.search(company_name, top_k=20)
            company_result["tfidf"] = {
                "retrieved": len(tfidf_hits),
                "top_scores": [round(h.get("tfidf_score", 0), 3) for h in tfidf_hits[:5]],
            }

        # Semantic retrieval test
        if semantic_searcher:
            sem_hits = semantic_searcher.search(company_name, top_k=20)
            company_result["semantic"] = {
                "retrieved": len(sem_hits),
                "top_scores": [round(h.get("semantic_score", 0), 3) for h in sem_hits[:5]],
            }

        # Comparison
        if tfidf_searcher and semantic_searcher:
            company_result["comparison"] = compare_algorithms_for_company(
                company_name,
                tfidf_hits,
                sem_hits,
            )

        # Sentiment
        if run_sentiment and company_texts:
            print(f"  [Sentiment] {company_name}...")
            sentiments = analyze_sentiment(company_texts)
            company_result["sentiment"] = aggregate_sentiment(sentiments)

        # NER
        if run_ner and company_texts:
            print(f"  [NER] {company_name}...")
            entities = extract_entities(company_texts)
            company_result["entities"] = aggregate_entities(entities)

        # Topics
        if topic_modeler and company_texts:
            dominant = topic_modeler.get_dominant_topic(company_texts)
            topic_dist = dict(zip(*np.unique(dominant, return_counts=True)))
            company_result["topics"] = {
                f"topic_{k}": int(v) for k, v in topic_dist.items()
            }

        # Zero-shot
        if run_zero_shot and company_texts:
            print(f"  [Zero-Shot] {company_name}...")
            zs = zero_shot_classify(company_texts[:10])  # limit for speed
            company_result["zero_shot"] = {
                "categories": dict(
                    sorted(
                        {r["top_label"]: r["top_score"] for r in zs}.items(),
                        key=lambda x: x[1], reverse=True,
                    )
                )
            }

        results["companies"][company_name] = company_result

    # Save results
    results_path = os.path.join(config.PROCESSED_DIR, "ml_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nML results saved → {results_path}")

    return results


def _generate_synthetic_articles(articles_by_company: dict) -> list:
    """
    Generate synthetic articles for demo/testing when scraping returns empty.
    Uses company descriptions and sector info to create realistic-looking data.
    """
    from src.data_loader import load_all_companies
    companies = load_all_companies()
    company_map = {c.name: c for c in companies}

    synthetic = []
    templates = [
        "{name} raises funding in {sector} space",
        "{name} launches new product, expanding in {sector}",
        "Investors eye {name} as {sector} market grows",
        "{name} partners with industry leaders for growth",
        "{name}: {description}",
        "How {name} is disrupting the {sector} industry",
        "{name} appoints new leadership to drive expansion",
        "Report: {sector} startup {name} sees rapid user growth",
    ]

    import random
    random.seed(config.RANDOM_SEED)

    for company_name in articles_by_company.keys():
        c = company_map.get(company_name)
        if not c:
            continue
        n_articles = random.randint(2, 15)  # Simulate sparse/dense coverage
        for j in range(n_articles):
            template = random.choice(templates)
            title = template.format(
                name=c.name, sector=c.sector, description=c.description[:60],
            )
            synthetic.append({
                "title": title,
                "snippet": c.description,
                "source": random.choice(["YourStory", "Inc42", "ET Tech", "Mint",
                                         "TechCrunch", "Entrackr", "Moneycontrol"]),
                "url": "",
                "published_date": "2026-01-15",
                "company_name": company_name,
                "search_term": company_name,
                "scrape_method": "synthetic",
                "article_id": f"syn_{company_name}_{j}",
            })

    print(f"  Generated {len(synthetic)} synthetic articles for {len(articles_by_company)} companies")
    return synthetic


def load_ml_results(path: str = None) -> dict:
    """Load previously computed ML results."""
    path = path or os.path.join(config.PROCESSED_DIR, "ml_results.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


# ── Full pipeline ──────────────────────────────────────────────
def run_full_pipeline(scrape: bool = True, heavy_ml: bool = False):
    """Run the complete ETL + ML pipeline."""
    # Step 1: Scrape
    articles = run_scrape_pipeline(scrape=scrape)

    # Step 2: Split
    train, val, test, cv_folds = run_split_pipeline()

    # Step 3: ML (lightweight by default)
    results = run_ml_pipeline(
        articles,
        run_tfidf=True,
        run_semantic=heavy_ml,
        run_sentiment=heavy_ml,
        run_ner=heavy_ml,
        run_topics=True,
        run_zero_shot=heavy_ml,
    )

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print(f"  Companies: {len(articles)}")
    print(f"  Total articles: {sum(len(v) for v in articles.values())}")
    print(f"  Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    print(f"  CV folds: {len(cv_folds)}")
    print("=" * 60)

    return articles, results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-scrape", action="store_true", help="Skip scraping, use cached data")
    parser.add_argument("--heavy", action="store_true", help="Run all ML models (needs GPU/time)")
    args = parser.parse_args()

    run_full_pipeline(scrape=not args.no_scrape, heavy_ml=args.heavy)
