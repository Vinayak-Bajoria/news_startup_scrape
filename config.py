"""
Central configuration for the ML News Analysis pipeline.
"""
import os

# ── Paths ──────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
MODELS_DIR = os.path.join(DATA_DIR, "models")
SPLITS_DIR = os.path.join(DATA_DIR, "splits")
EXCEL_PATH = os.path.join(ASSETS_DIR, "Indian Startups (1).xlsx")

# ── Scraping ───────────────────────────────────────────────────
SCRAPE_LAST_N_DAYS = 30
MAX_ARTICLES_PER_COMPANY = 100
REQUEST_TIMEOUT = 15          # seconds
REQUEST_DELAY = (0.5, 1.5)    # random delay range between requests
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)

# ── Train / Test / Validation split ───────────────────────────
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
CV_FOLDS = 5
RANDOM_SEED = 42

# ── Reference MNCs (for comparison baseline) ──────────────────
REFERENCE_MNCS = [
    {"name": "Tata Consultancy Services", "short": "TCS", "sector": "IT Services"},
    {"name": "Infosys", "short": "Infosys", "sector": "IT Services"},
    {"name": "Reliance Industries", "short": "Reliance", "sector": "Conglomerate"},
    {"name": "Wipro", "short": "Wipro", "sector": "IT Services"},
    {"name": "HCL Technologies", "short": "HCL Tech", "sector": "IT Services"},
]

# ── ML Model configs ──────────────────────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"     # Sentence-Transformers model
NER_MODEL = "dslim/bert-base-NER"
SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
ZERO_SHOT_MODEL = "facebook/bart-large-mnli"
LDA_NUM_TOPICS = 8

# ── News categories for classification ────────────────────────
NEWS_CATEGORIES = [
    "Funding & Investment",
    "Product Launch",
    "Partnership & Collaboration",
    "Acquisition & Merger",
    "Leadership Change",
    "Revenue & Financials",
    "Regulation & Policy",
    "Controversy & Criticism",
    "Expansion & Growth",
    "Awards & Recognition",
]

# Ensure directories exist
for d in [DATA_DIR, RAW_DIR, PROCESSED_DIR, MODELS_DIR, SPLITS_DIR]:
    os.makedirs(d, exist_ok=True)
