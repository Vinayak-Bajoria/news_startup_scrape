"""
Loads startup data and news sources from the Excel file.
Provides a unified company registry used by all downstream modules.
"""
import pandas as pd
from dataclasses import dataclass, field, asdict
from typing import List, Optional
import json, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# ── Data classes ───────────────────────────────────────────────
@dataclass
class Company:
    name: str
    founding_year: str
    sector: str
    founders: str
    description: str
    company_type: str = "startup"       # "startup" or "mnc"
    split: Optional[str] = None         # "train", "val", "test"
    search_terms: List[str] = field(default_factory=list)

    def __post_init__(self):
        # Auto-generate search terms for scraping
        if not self.search_terms:
            terms = [self.name]
            # Add shorter / alternate names (only if meaningful, > 3 chars)
            if "." in self.name:
                cleaned = self.name.replace(".", "")
                if len(cleaned) > 3:
                    terms.append(cleaned)
            self.search_terms = list(set(terms))

    def to_dict(self):
        return asdict(self)


@dataclass
class NewsSource:
    name: str
    url: str
    focus: str

    def to_dict(self):
        return {"name": self.name, "url": self.url, "focus": self.focus}


# ── Loader functions ──────────────────────────────────────────
def load_startups() -> List[Company]:
    """Load the 50 startups from the Excel file."""
    df = pd.read_excel(
        config.EXCEL_PATH,
        sheet_name="Indian Startups 2020-2025",
    )
    companies = []
    for _, row in df.iterrows():
        companies.append(Company(
            name=str(row["Startup Name"]).strip(),
            founding_year=str(row["Founding Year"]).strip(),
            sector=str(row["Industry / Sector"]).strip(),
            founders=str(row["Founders"]).strip(),
            description=str(row["Description"]).strip(),
            company_type="startup",
        ))
    return companies


def load_mncs() -> List[Company]:
    """Load reference MNC companies for comparison."""
    return [
        Company(
            name=m["name"],
            founding_year="N/A",
            sector=m["sector"],
            founders="N/A",
            description=f"Reference MNC: {m['name']}",
            company_type="mnc",
            search_terms=[m["name"], m["short"]],
        )
        for m in config.REFERENCE_MNCS
    ]


def load_news_sources() -> List[NewsSource]:
    """Load the 40 curated news sources from the Excel file."""
    df = pd.read_excel(config.EXCEL_PATH, sheet_name="Sheet1")
    sources = []
    for _, row in df.iterrows():
        sources.append(NewsSource(
            name=str(row["Website Name"]).strip(),
            url=str(row["Website URL"]).strip(),
            focus=str(row["Primary Focus & Specialty"]).strip(),
        ))
    return sources


def load_all_companies() -> List[Company]:
    """Combined list: 50 startups + MNC baselines."""
    return load_startups() + load_mncs()


def save_company_registry(companies: List[Company], path: str = None):
    """Persist the company registry as JSON."""
    path = path or os.path.join(config.PROCESSED_DIR, "company_registry.json")
    with open(path, "w") as f:
        json.dump([c.to_dict() for c in companies], f, indent=2)
    return path


def load_company_registry(path: str = None) -> List[Company]:
    """Load previously saved registry."""
    path = path or os.path.join(config.PROCESSED_DIR, "company_registry.json")
    with open(path) as f:
        data = json.load(f)
    return [Company(**d) for d in data]


# ── Quick test ─────────────────────────────────────────────────
if __name__ == "__main__":
    startups = load_startups()
    mncs = load_mncs()
    sources = load_news_sources()
    print(f"Loaded {len(startups)} startups, {len(mncs)} MNCs, {len(sources)} news sources")
    print(f"\nSample startup: {startups[0]}")
    print(f"\nSample source: {sources[0]}")
