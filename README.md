# 📰 Newsssyyy — Startup News Intelligence

A Streamlit app that scrapes, analyzes, and compares news coverage of **50 Indian startups + 5 MNCs** using **6 different ML algorithms**. Search any company in real-time or explore pre-computed dataset results.

## Features

- **Live Search** — Type any company name, pick a timeline (1–30 days), and see real-time ML analysis from Google News RSS
- **Dataset Analysis** — Pre-computed results for 50 startups + 5 MNCs with side-by-side algorithm comparison
- **Best-First Tabs** — Algorithms are automatically ranked; the best-performing one appears first
- **6 ML Algorithms**: TF-IDF Search, Keyword Match, Source Diversity, Temporal Analysis, Topic Extraction, Composite Coverage Score
- **Startup vs MNC Comparison** — Head-to-head bar charts across all metrics

## Project Structure

```
news_startup_scrape/
├── Newsssyyy_Home.py          # Home page — Live Search
├── pages/
│   └── 2_Dataset_Analysis.py  # Dataset Analysis page
├── src/
│   ├── data_loader.py         # Load startup/MNC data from Excel
│   ├── news_scraper.py        # Google News RSS scraper
│   ├── splitter.py            # Train/val/test split logic
│   ├── ml_models.py           # TF-IDF & LDA model training
│   └── pipeline.py            # Full scrape + ML pipeline
├── data/
│   ├── raw/                   # Scraped articles JSON
│   ├── processed/             # ML results, company registry
│   ├── models/                # Saved TF-IDF & LDA models
│   └── splits/                # Train/val/test split definitions
├── assets/
│   └── Indian Startups (1).xlsx  # Source dataset
├── config.py                  # Path & parameter configuration
├── run_pipeline.py            # Run full scrape pipeline
├── run_ml.py                  # Run ML model training
├── test_smoke.py              # End-to-end smoke test
├── requirements.txt
└── README.md
```

## Setup

```bash
# Clone
git clone https://github.com/Vinayak-Bajoria/news_startup_scrape.git
cd news_startup_scrape

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run Newsssyyy_Home.py
```

## Pipeline (optional — data is pre-computed)

```bash
# Scrape news for all 55 companies
python run_pipeline.py

# Train TF-IDF & LDA models on scraped data
python run_ml.py
```

## Tech Stack

- **Streamlit** — Dashboard UI
- **scikit-learn** — TF-IDF vectorization, LDA topic modeling
- **Plotly** — Interactive charts (bar, pie, gauge, timeline)
- **feedparser + BeautifulSoup** — Google News RSS scraping
- **pandas** — Data wrangling

## Dataset

- **50 Indian startups** across fintech, edtech, healthtech, foodtech, logistics, and more
- **5 MNCs** (Google, Microsoft, Amazon, Apple, Meta) as comparison baselines
- **634 articles** scraped across 55 companies
- **Train/Val/Test split**: 32 / 9 / 9 companies (stratified by sector)