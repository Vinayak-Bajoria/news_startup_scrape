"""
Six ML algorithm "flavors" for news analysis.

Each produces a different view of the same articles, demonstrating how
algorithm choice affects results — especially for sparse startup news.

Algorithms
──────────
1. TF-IDF Keyword Search     — exact lexical retrieval
2. Semantic Search            — embedding-based similarity
3. Sentiment Analysis         — tone of coverage
4. Named Entity Recognition   — entity extraction
5. Topic Modeling (LDA)       — unsupervised theme clustering
6. Zero-Shot Classification   — label articles with no training data

All models are designed to be company-agnostic: they never memorise the
50 training company names.  Instead they learn patterns (relevance signals,
sentiment cues, entity structures, topic distributions) that transfer to
any new company the user types into the search bar.
"""
import os, json, pickle, re
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict

# NLP / ML
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
)

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# ── Lazy-loaded heavy models (loaded once on first call) ──────
_models_cache: Dict[str, Any] = {}


def _get_sentence_transformer():
    if "st" not in _models_cache:
        from sentence_transformers import SentenceTransformer
        _models_cache["st"] = SentenceTransformer(config.EMBEDDING_MODEL)
    return _models_cache["st"]


def _get_sentiment_pipeline():
    if "sent" not in _models_cache:
        from transformers import pipeline
        _models_cache["sent"] = pipeline(
            "sentiment-analysis",
            model=config.SENTIMENT_MODEL,
            truncation=True,
            max_length=512,
        )
    return _models_cache["sent"]


def _get_ner_pipeline():
    if "ner" not in _models_cache:
        from transformers import pipeline
        _models_cache["ner"] = pipeline(
            "ner",
            model=config.NER_MODEL,
            aggregation_strategy="simple",
            device=-1,
        )
    return _models_cache["ner"]


def _get_zero_shot_pipeline():
    if "zs" not in _models_cache:
        from transformers import pipeline
        _models_cache["zs"] = pipeline(
            "zero-shot-classification",
            model=config.ZERO_SHOT_MODEL,
            device=-1,
        )
    return _models_cache["zs"]


# ═══════════════════════════════════════════════════════════════
# 1. TF-IDF KEYWORD SEARCH
# ═══════════════════════════════════════════════════════════════
class TFIDFSearcher:
    """
    Classic bag-of-words retrieval.
    
    Trains a TF-IDF matrix on ALL collected articles.  At query time,
    transforms the query and retrieves by cosine similarity.

    ✅ Fast, interpretable, no GPU
    ❌ Misses paraphrases & indirect mentions
    """

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
        )
        self.tfidf_matrix = None
        self.corpus_meta: List[dict] = []   # parallel list of article metadata

    def fit(self, articles: List[dict]):
        """Fit TF-IDF on a list of article dicts with 'title' and 'snippet'."""
        texts = [f"{a['title']} {a.get('snippet', '')}" for a in articles]
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)
        self.corpus_meta = articles

    def search(self, query: str, top_k: int = 20) -> List[dict]:
        """Return top-k articles most similar to the query."""
        q_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self.tfidf_matrix).flatten()
        top_idx = sims.argsort()[-top_k:][::-1]
        results = []
        for idx in top_idx:
            if sims[idx] > 0:
                r = dict(self.corpus_meta[idx])
                r["tfidf_score"] = float(sims[idx])
                results.append(r)
        return results

    def get_top_terms(self, n: int = 20) -> List[str]:
        """Top terms by average TF-IDF weight."""
        if self.tfidf_matrix is None:
            return []
        mean_tfidf = self.tfidf_matrix.mean(axis=0).A1
        top_idx = mean_tfidf.argsort()[-n:][::-1]
        feature_names = self.vectorizer.get_feature_names_out()
        return [(feature_names[i], float(mean_tfidf[i])) for i in top_idx]

    def save(self, path: str = None):
        path = path or os.path.join(config.MODELS_DIR, "tfidf_model.pkl")
        with open(path, "wb") as f:
            pickle.dump({"vectorizer": self.vectorizer, "matrix": self.tfidf_matrix,
                          "meta": self.corpus_meta}, f)

    def load(self, path: str = None):
        path = path or os.path.join(config.MODELS_DIR, "tfidf_model.pkl")
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.vectorizer = data["vectorizer"]
        self.tfidf_matrix = data["matrix"]
        self.corpus_meta = data["meta"]


# ═══════════════════════════════════════════════════════════════
# 2. SEMANTIC SEARCH  (Sentence-Transformers)
# ═══════════════════════════════════════════════════════════════
class SemanticSearcher:
    """
    Embedding-based retrieval using sentence-transformers.

    ✅ Finds paraphrases, synonyms, indirect mentions
    ✅ Works even when company name isn't in the article
    ❌ Slower, needs embeddings pre-computed
    """

    def __init__(self):
        self.embeddings = None
        self.corpus_meta: List[dict] = []

    def fit(self, articles: List[dict]):
        """Compute embeddings for all articles."""
        model = _get_sentence_transformer()
        texts = [f"{a['title']} {a.get('snippet', '')}" for a in articles]
        self.embeddings = model.encode(texts, show_progress_bar=True,
                                        batch_size=64, normalize_embeddings=True)
        self.corpus_meta = articles

    def search(self, query: str, top_k: int = 20) -> List[dict]:
        model = _get_sentence_transformer()
        q_emb = model.encode([query], normalize_embeddings=True)
        sims = cosine_similarity(q_emb, self.embeddings).flatten()
        top_idx = sims.argsort()[-top_k:][::-1]
        results = []
        for idx in top_idx:
            if sims[idx] > 0.1:
                r = dict(self.corpus_meta[idx])
                r["semantic_score"] = float(sims[idx])
                results.append(r)
        return results

    def save(self, path: str = None):
        path = path or os.path.join(config.MODELS_DIR, "semantic_model.pkl")
        with open(path, "wb") as f:
            pickle.dump({"embeddings": self.embeddings, "meta": self.corpus_meta}, f)

    def load(self, path: str = None):
        path = path or os.path.join(config.MODELS_DIR, "semantic_model.pkl")
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.embeddings = data["embeddings"]
        self.corpus_meta = data["meta"]


# ═══════════════════════════════════════════════════════════════
# 3. SENTIMENT ANALYSIS
# ═══════════════════════════════════════════════════════════════
def analyze_sentiment(texts: List[str], batch_size: int = 32) -> List[dict]:
    """
    Run sentiment analysis on a list of texts.
    Returns list of {label, score} dicts.

    ✅ Reveals tone of coverage (positive/negative/neutral)
    ✅ Can compare startup sentiment vs MNC sentiment
    """
    pipe = _get_sentiment_pipeline()
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        # Truncate long texts
        batch = [t[:500] for t in batch]
        preds = pipe(batch)
        results.extend(preds)
    return results


def aggregate_sentiment(sentiments: List[dict]) -> dict:
    """Aggregate sentiment predictions into summary statistics."""
    if not sentiments:
        return {"positive": 0, "negative": 0, "neutral": 0, "avg_score": 0}
    label_map = {"positive": "positive", "negative": "negative", "neutral": "neutral",
                 "POSITIVE": "positive", "NEGATIVE": "negative", "NEUTRAL": "neutral",
                 "LABEL_0": "negative", "LABEL_1": "neutral", "LABEL_2": "positive"}
    counts = Counter()
    total_score = 0
    for s in sentiments:
        label = label_map.get(s.get("label", ""), "neutral")
        counts[label] += 1
        total_score += s.get("score", 0)
    n = len(sentiments)
    return {
        "positive": counts["positive"],
        "negative": counts["negative"],
        "neutral": counts["neutral"],
        "positive_pct": round(counts["positive"] / n * 100, 1),
        "negative_pct": round(counts["negative"] / n * 100, 1),
        "neutral_pct": round(counts["neutral"] / n * 100, 1),
        "avg_confidence": round(total_score / n, 3),
        "total_articles": n,
    }


# ═══════════════════════════════════════════════════════════════
# 4. NAMED ENTITY RECOGNITION  (NER)
# ═══════════════════════════════════════════════════════════════
def extract_entities(texts: List[str]) -> List[List[dict]]:
    """
    Extract named entities from texts.

    ✅ Finds co-mentioned companies, people, locations
    ✅ Reveals ecosystem connections
    ✅ Company-agnostic — doesn't need to know company names
    """
    pipe = _get_ner_pipeline()
    all_entities = []
    for text in texts:
        text = text[:512]  # model max length
        try:
            ents = pipe(text)
            all_entities.append([
                {
                    "word": e["word"],
                    "entity": e["entity_group"],
                    "score": round(float(e["score"]), 3),
                }
                for e in ents
            ])
        except Exception:
            all_entities.append([])
    return all_entities


def aggregate_entities(all_entities: List[List[dict]]) -> dict:
    """Aggregate entities across all articles for a company."""
    entity_counts: Dict[str, Counter] = defaultdict(Counter)
    for entities in all_entities:
        for e in entities:
            entity_counts[e["entity"]][e["word"]] += 1

    return {
        entity_type: dict(counter.most_common(15))
        for entity_type, counter in entity_counts.items()
    }


# ═══════════════════════════════════════════════════════════════
# 5. TOPIC MODELING  (LDA)
# ═══════════════════════════════════════════════════════════════
class TopicModeler:
    """
    LDA topic modeling to discover what themes dominate startup news
    vs MNC news.

    ✅ Unsupervised — discovers themes without labels
    ✅ Shows whether startups are covered for 'funding' while MNCs for 'earnings'
    """

    def __init__(self, n_topics: int = None):
        self.n_topics = n_topics or config.LDA_NUM_TOPICS
        self.vectorizer = TfidfVectorizer(
            max_features=5000, stop_words="english",
            max_df=0.9, min_df=3,
        )
        self.lda = LatentDirichletAllocation(
            n_components=self.n_topics,
            random_state=config.RANDOM_SEED,
            max_iter=20,
        )
        self.is_fitted = False

    def fit(self, texts: List[str]):
        """Fit LDA on the corpus."""
        dtm = self.vectorizer.fit_transform(texts)
        self.lda.fit(dtm)
        self.is_fitted = True

    def transform(self, texts: List[str]) -> np.ndarray:
        """Get topic distribution for new texts."""
        dtm = self.vectorizer.transform(texts)
        return self.lda.transform(dtm)

    def get_topics(self, n_words: int = 10) -> List[List[str]]:
        """Return top words for each topic."""
        feature_names = self.vectorizer.get_feature_names_out()
        topics = []
        for topic_idx, topic in enumerate(self.lda.components_):
            top_words = [feature_names[i] for i in topic.argsort()[:-n_words - 1:-1]]
            topics.append(top_words)
        return topics

    def get_dominant_topic(self, texts: List[str]) -> List[int]:
        """Return the dominant topic index for each text."""
        dist = self.transform(texts)
        return dist.argmax(axis=1).tolist()

    def save(self, path: str = None):
        path = path or os.path.join(config.MODELS_DIR, "lda_model.pkl")
        with open(path, "wb") as f:
            pickle.dump({"vectorizer": self.vectorizer, "lda": self.lda,
                          "n_topics": self.n_topics}, f)

    def load(self, path: str = None):
        path = path or os.path.join(config.MODELS_DIR, "lda_model.pkl")
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.vectorizer = data["vectorizer"]
        self.lda = data["lda"]
        self.n_topics = data["n_topics"]
        self.is_fitted = True


# ═══════════════════════════════════════════════════════════════
# 6. ZERO-SHOT CLASSIFICATION
# ═══════════════════════════════════════════════════════════════
def zero_shot_classify(
    texts: List[str],
    categories: List[str] = None,
) -> List[dict]:
    """
    Classify articles into news categories without any training data.

    ✅ No labelled data needed — perfect for startups with little data
    ✅ Flexible — change categories on the fly
    ❌ Slow (large model), lower accuracy than fine-tuned
    """
    categories = categories or config.NEWS_CATEGORIES
    pipe = _get_zero_shot_pipeline()
    results = []
    for text in texts:
        text = text[:500]  # truncate
        try:
            pred = pipe(text, candidate_labels=categories, multi_label=False)
            results.append({
                "top_label": pred["labels"][0],
                "top_score": round(float(pred["scores"][0]), 3),
                "all_labels": dict(zip(pred["labels"], [round(s, 3) for s in pred["scores"]])),
            })
        except Exception:
            results.append({"top_label": "Unknown", "top_score": 0, "all_labels": {}})
    return results


# ═══════════════════════════════════════════════════════════════
# EVALUATION METRICS  (for comparing algorithms)
# ═══════════════════════════════════════════════════════════════
def compute_retrieval_metrics(
    retrieved_articles: List[dict],
    company_name: str,
) -> dict:
    """
    Evaluate how well a retrieval method found RELEVANT articles.
    Uses a simple heuristic: article is relevant if company name
    appears in title or snippet (ground-truth proxy).
    """
    if not retrieved_articles:
        return {"precision": 0, "recall": 0, "f1": 0, "total_retrieved": 0}

    relevant = 0
    name_lower = company_name.lower()
    name_words = set(name_lower.split())

    for a in retrieved_articles:
        text = f"{a.get('title', '')} {a.get('snippet', '')}".lower()
        # Check if any meaningful word of the company name appears
        if name_lower in text or any(w in text for w in name_words if len(w) > 3):
            relevant += 1

    precision = relevant / len(retrieved_articles) if retrieved_articles else 0
    return {
        "precision": round(precision, 3),
        "total_retrieved": len(retrieved_articles),
        "relevant_retrieved": relevant,
    }


def compare_algorithms_for_company(
    company_name: str,
    tfidf_results: List[dict],
    semantic_results: List[dict],
) -> dict:
    """Side-by-side comparison of TF-IDF vs Semantic search."""
    return {
        "company": company_name,
        "tfidf": compute_retrieval_metrics(tfidf_results, company_name),
        "semantic": compute_retrieval_metrics(semantic_results, company_name),
    }
