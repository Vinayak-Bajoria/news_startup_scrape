"""
Company-level train / validation / test splitter with stratification by sector.

Why company-level splits?
─────────────────────────
If we split at the article level, the model sees articles about company X in
training AND test. It learns to recognise *company names*, not generalisable
patterns.  By holding out entire companies, the test set simulates the real use-
case: "a user types a brand-new company name the model has never seen."

Split allocation
────────────────
  Train  : 70 % of companies  (~35 startups)
  Val    : 15 %               (~ 8 startups)
  Test   : 15 %               (~ 7 startups)

Cross-validation: 5-fold within the training set (also company-level).

Stratification is by *coarsened* sector so each fold gets a mix of fintech,
healthtech, D2C, etc.
"""
import os, json, random
from typing import List, Dict, Tuple
from collections import defaultdict
from sklearn.model_selection import StratifiedKFold
import numpy as np
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.data_loader import Company


# ── Sector coarsening ──────────────────────────────────────────
# Many sectors have only 1 company.  We coarsen to ~8 buckets so
# stratified splitting has enough members per stratum.

_SECTOR_MAP = {
    "fintech":   ["Fintech", "Fintech / Social Trading", "Fintech / WealthTech",
                  "AI / Fintech", "InsurTech"],
    "healthtech": ["HealthTech", "HealthTech / Wearables", "EdTech / HealthTech"],
    "d2c":       ["D2C Aggregator", "D2C Aggregator (Thrasio-style)", "D2C Fashion",
                  "D2C Beauty", "D2C Footwear", "D2C Food", "F&B / D2C"],
    "edtech":    ["EdTech"],
    "ai_saas":   ["AI / SaaS", "SaaS / GRC"],
    "commerce":  ["Quick Commerce", "E-commerce Enabler", "FoodTech"],
    "deep_tech": ["Consumer Electronics", "CleanTech", "Location Tech",
                  "AgriTech", "Tech / App Store"],
    "other":     ["Social Community", "Creator Economy", "Spiritual Tech",
                  "Gaming / Community", "TravelTech", "Senior Tech",
                  "PropTech", "Wedding Tech", "HRTech / Jobs"],
}


def coarsen_sector(sector: str) -> str:
    """Map a fine-grained sector label to a coarse bucket."""
    for coarse, fine_list in _SECTOR_MAP.items():
        if sector in fine_list:
            return coarse
    return "other"


# ── Splitting logic ────────────────────────────────────────────
def split_companies(
    companies: List[Company],
    train_ratio: float = None,
    val_ratio: float = None,
    test_ratio: float = None,
    seed: int = None,
) -> Tuple[List[Company], List[Company], List[Company]]:
    """
    Split companies into train / val / test by stratified sector.
    Mutates each Company.split field in-place.
    Returns (train, val, test) lists.
    """
    train_ratio = train_ratio or config.TRAIN_RATIO
    val_ratio = val_ratio or config.VAL_RATIO
    test_ratio = test_ratio or config.TEST_RATIO
    seed = seed or config.RANDOM_SEED

    # Filter to startups only (MNCs are always in "reference", not split)
    startups = [c for c in companies if c.company_type == "startup"]
    mncs = [c for c in companies if c.company_type == "mnc"]

    # Group by coarse sector
    sector_groups: Dict[str, List[Company]] = defaultdict(list)
    for c in startups:
        sector_groups[coarsen_sector(c.sector)].append(c)

    random.seed(seed)

    train, val, test = [], [], []

    for sector, group in sector_groups.items():
        random.shuffle(group)
        n = len(group)
        n_test = max(1, round(n * test_ratio))
        n_val = max(1, round(n * val_ratio))
        n_train = n - n_test - n_val
        if n_train < 1:
            # Very small group — put at least 1 in train
            n_train = 1
            n_val = max(0, n - n_train - n_test)

        for c in group[:n_train]:
            c.split = "train"
            train.append(c)
        for c in group[n_train:n_train + n_val]:
            c.split = "val"
            val.append(c)
        for c in group[n_train + n_val:]:
            c.split = "test"
            test.append(c)

    # MNCs are marked as "reference" — available in all splits
    for c in mncs:
        c.split = "reference"

    return train, val, test


# ── Cross-Validation folds (within training set) ──────────────
def make_cv_folds(
    train_companies: List[Company],
    n_folds: int = None,
    seed: int = None,
) -> List[Tuple[List[str], List[str]]]:
    """
    Create K-fold CV indices within the training set, stratified by sector.
    Returns list of (train_names, val_names) tuples.
    """
    n_folds = n_folds or config.CV_FOLDS
    seed = seed or config.RANDOM_SEED

    names = np.array([c.name for c in train_companies])
    labels = np.array([coarsen_sector(c.sector) for c in train_companies])

    # StratifiedKFold needs at least n_folds per class; handle small classes
    unique, counts = np.unique(labels, return_counts=True)
    min_count = counts.min()
    effective_folds = min(n_folds, min_count) if min_count < n_folds else n_folds

    if effective_folds < 2:
        # Fall back to simple K-fold if classes too small
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        indices_iter = kf.split(names)
    else:
        skf = StratifiedKFold(n_splits=effective_folds, shuffle=True, random_state=seed)
        indices_iter = skf.split(names, labels)

    folds = []
    for train_idx, val_idx in indices_iter:
        folds.append((names[train_idx].tolist(), names[val_idx].tolist()))

    return folds


# ── Persistence ────────────────────────────────────────────────
def save_splits(
    train: List[Company],
    val: List[Company],
    test: List[Company],
    cv_folds: List[Tuple[List[str], List[str]]] = None,
    path: str = None,
):
    """Save split assignments to JSON for reproducibility."""
    path = path or os.path.join(config.SPLITS_DIR, "splits.json")

    data = {
        "train": [c.name for c in train],
        "val": [c.name for c in val],
        "test": [c.name for c in test],
        "train_sectors": {c.name: coarsen_sector(c.sector) for c in train},
        "val_sectors": {c.name: coarsen_sector(c.sector) for c in val},
        "test_sectors": {c.name: coarsen_sector(c.sector) for c in test},
    }
    if cv_folds:
        data["cv_folds"] = [
            {"fold_train": t, "fold_val": v}
            for t, v in cv_folds
        ]

    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Splits saved → {path}")
    return path


def load_splits(path: str = None) -> dict:
    """Load saved splits."""
    path = path or os.path.join(config.SPLITS_DIR, "splits.json")
    with open(path) as f:
        return json.load(f)


# ── Pretty print ───────────────────────────────────────────────
def print_split_summary(train, val, test, cv_folds=None):
    """Display a readable split summary."""
    print("\n" + "=" * 60)
    print("COMPANY-LEVEL TRAIN / VAL / TEST SPLIT")
    print("=" * 60)

    for label, group in [("TRAIN", train), ("VAL", val), ("TEST", test)]:
        sectors = defaultdict(list)
        for c in group:
            sectors[coarsen_sector(c.sector)].append(c.name)
        print(f"\n{label} ({len(group)} companies):")
        for s, names in sorted(sectors.items()):
            print(f"  [{s:12s}] {', '.join(names)}")

    if cv_folds:
        print(f"\nCROSS-VALIDATION: {len(cv_folds)} folds within training set")
        for i, (t, v) in enumerate(cv_folds):
            print(f"  Fold {i+1}: train={len(t)}, val={len(v)} → val companies: {v}")

    print("=" * 60)


# ── CLI entry point ────────────────────────────────────────────
if __name__ == "__main__":
    from src.data_loader import load_startups, load_mncs

    companies = load_startups() + load_mncs()
    train, val, test = split_companies(companies)
    cv_folds = make_cv_folds(train)
    print_split_summary(train, val, test, cv_folds)
    save_splits(train, val, test, cv_folds)
