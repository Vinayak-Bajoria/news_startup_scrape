"""
📊 Dataset Analysis — Pre-computed ML results for 50 startups + 5 MNCs.

Features:
  • Company dropdown to select from the dataset
  • Timeline filter: 1-7 days, 15 days, 30 days
  • Side-by-side view of how each ML algorithm interprets the same company's news
  • Startup vs MNC comparison charts
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from collections import Counter
import json, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.data_loader import load_startups, load_mncs
from src.splitter import load_splits
from src.news_scraper import load_articles, Article

st.set_page_config(page_title="Newsssyyy — Dataset", page_icon="📊", layout="wide")


# ── Cached data loaders ────────────────────────────────────────
@st.cache_data
def get_startups():
    return load_startups()

@st.cache_data
def get_mncs():
    return load_mncs()

@st.cache_data
def get_all_articles():
    try:
        raw = load_articles()
        result = {}
        for name, arts in raw.items():
            result[name] = [a.to_dict() if isinstance(a, Article) else a for a in arts]
        return result
    except FileNotFoundError:
        return {}

@st.cache_data
def get_splits():
    try:
        return load_splits()
    except FileNotFoundError:
        return None


def filter_articles_by_days(articles: list, days: int) -> list:
    """Filter articles to only those published within the last N days."""
    if not articles or days >= 30:
        return articles
    cutoff = datetime.now() - timedelta(days=days)
    filtered = []
    for a in articles:
        pub = a.get("published_date", "")
        if not pub:
            continue
        try:
            pub_dt = datetime.strptime(pub[:10], "%Y-%m-%d")
            if pub_dt >= cutoff:
                filtered.append(a)
        except ValueError:
            continue
    return filtered


def run_tfidf_on_articles(articles: list, query: str) -> list:
    """Run TF-IDF search on filtered articles."""
    if not articles:
        return []
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    texts = [f"{a.get('title', '')} {a.get('snippet', '')}" for a in articles]
    try:
        vec = TfidfVectorizer(max_features=5000, stop_words="english", ngram_range=(1, 2))
        mat = vec.fit_transform(texts)
        q_vec = vec.transform([query])
        sims = cosine_similarity(q_vec, mat).flatten()
        results = []
        for idx in sims.argsort()[::-1]:
            if sims[idx] > 0.01:
                r = dict(articles[idx])
                r["tfidf_score"] = round(float(sims[idx]), 4)
                results.append(r)
        return results[:20]
    except Exception:
        return []


def run_keyword_match(articles: list, query: str) -> list:
    """Simple keyword presence check — baseline comparison."""
    query_lower = query.lower()
    query_words = set(w for w in query_lower.split() if len(w) > 2)
    results = []
    for a in articles:
        text = f"{a.get('title', '')} {a.get('snippet', '')}".lower()
        matches = sum(1 for w in query_words if w in text)
        if matches > 0:
            r = dict(a)
            r["keyword_score"] = round(matches / max(len(query_words), 1), 3)
            results.append(r)
    return sorted(results, key=lambda x: x["keyword_score"], reverse=True)[:20]


def run_source_analysis(articles: list) -> dict:
    if not articles:
        return {"sources": {}, "diversity": 0, "top_source": ("N/A", 0)}
    source_counts = Counter(a.get("source", "Unknown") for a in articles)
    return {
        "sources": dict(source_counts.most_common(15)),
        "diversity": len(source_counts),
        "top_source": source_counts.most_common(1)[0] if source_counts else ("N/A", 0),
    }


def run_date_analysis(articles: list) -> dict:
    if not articles:
        return {"dates": {}, "date_range": "N/A", "total_days": 0}
    date_counts = Counter()
    for a in articles:
        pub = a.get("published_date", "")
        if pub:
            date_counts[pub[:10]] += 1
    sorted_dates = sorted(date_counts.keys())
    return {
        "dates": dict(sorted(date_counts.items())),
        "date_range": f"{sorted_dates[0]} to {sorted_dates[-1]}" if sorted_dates else "N/A",
        "total_days": len(date_counts),
    }


def run_topic_keywords(articles: list) -> dict:
    if len(articles) < 3:
        return {"topics": [], "error": "Too few articles for topic extraction"}
    from sklearn.feature_extraction.text import TfidfVectorizer
    texts = [f"{a.get('title', '')} {a.get('snippet', '')}" for a in articles]
    try:
        vec = TfidfVectorizer(max_features=1000, stop_words="english", max_df=0.9, min_df=1)
        mat = vec.fit_transform(texts)
        feature_names = vec.get_feature_names_out()
        mean_tfidf = mat.mean(axis=0).A1
        top_idx = mean_tfidf.argsort()[-20:][::-1]
        return {"topics": [(feature_names[i], round(float(mean_tfidf[i]), 4)) for i in top_idx]}
    except Exception as e:
        return {"topics": [], "error": str(e)}


# ═══════════════════════════════════════════════════════════════
# PAGE LAYOUT
# ═══════════════════════════════════════════════════════════════
st.title("📊 Dataset Analysis — ML Model Comparison")
st.markdown("Select a company from the dataset and adjust the timeline to see how **different ML algorithms** produce different results.")

articles_data = get_all_articles()
startups = get_startups()
mncs = get_mncs()
all_companies = startups + mncs
company_names = [c.name for c in all_companies]

# ── Sidebar ────────────────────────────────────────────────────
st.sidebar.header("🔧 Controls")

selected_company = st.sidebar.selectbox(
    "🏢 Select Company", company_names, index=0,
    help="Pick any company from our dataset of 50 startups + 5 MNCs",
)

timeline_options = {
    "1 day": 1, "2 days": 2, "3 days": 3, "4 days": 4,
    "5 days": 5, "6 days": 6, "7 days": 7,
    "15 days": 15, "30 days (all)": 30,
}
selected_timeline = st.sidebar.select_slider(
    "📅 News Timeline", options=list(timeline_options.keys()), value="30 days (all)",
)
days = timeline_options[selected_timeline]

company_obj = next((c for c in all_companies if c.name == selected_company), None)
if company_obj:
    st.sidebar.divider()
    st.sidebar.subheader(f"ℹ️ {company_obj.name}")
    st.sidebar.markdown(f"**Type**: {'🏢 MNC' if company_obj.company_type == 'mnc' else '🚀 Startup'}")
    st.sidebar.markdown(f"**Sector**: {company_obj.sector}")
    st.sidebar.markdown(f"**Founded**: {company_obj.founding_year}")
    splits = get_splits()
    if splits:
        for s in ["train", "val", "test"]:
            if company_obj.name in splits.get(s, []):
                st.sidebar.markdown(f"**Split**: `{s.upper()}`")
                break

# ── Filtered articles ──────────────────────────────────────────
all_arts = articles_data.get(selected_company, [])
filtered = filter_articles_by_days(all_arts, days)

m1, m2, m3, m4 = st.columns(4)
m1.metric("📰 Total (30d)", len(all_arts))
m2.metric(f"📰 ({selected_timeline})", len(filtered))
src_info = run_source_analysis(filtered)
m3.metric("📡 Sources", src_info["diversity"])
dt_info = run_date_analysis(filtered)
m4.metric("📅 News Days", dt_info["total_days"])

# ═══════════════════════════════════════════════════════════════
# ML TABS
# ═══════════════════════════════════════════════════════════════
st.header(f"🤖 ML Results: {selected_company}")
st.caption(f"Timeline: last {days} day(s) — {len(filtered)} articles")

if not filtered:
    st.warning(f"**No articles found** for **{selected_company}** in the last **{days} day(s)**. "
               "Try expanding the timeline or compare with an MNC.")
    st.info("**Key Insight**: When articles = 0, all algorithms fail equally. "
            "But with 1-3 articles, they diverge: TF-IDF ranks poorly, Keyword match over-matches, "
            "Topic Modeling fails, and Coverage Score collapses.")
else:
    # ── Pre-compute ALL results for ranking ────────────────────
    tfidf_res = run_tfidf_on_articles(filtered, selected_company)
    kw_res = run_keyword_match(filtered, selected_company)
    topics = run_topic_keywords(filtered)
    topic_list = topics.get("topics", [])
    kw_pct = len(kw_res) / len(filtered) * 100 if filtered else 0

    volume_score = min(len(filtered) / 50 * 100, 100)
    diversity_score = min(src_info["diversity"] / 10 * 100, 100)
    temporal_score = min(dt_info["total_days"] / max(days, 1) * 100, 100)
    relevance_score = (len(tfidf_res) / max(len(filtered), 1)) * 100 if filtered else 0
    composite = round((volume_score + diversity_score + temporal_score + relevance_score) / 4, 1)

    # Score each algorithm 0-100
    tfidf_avg = sum(r["tfidf_score"] for r in tfidf_res) / max(len(tfidf_res), 1)
    algo_scores = {
        "tfidf":    min((tfidf_avg * 200) + min(len(tfidf_res) / max(len(filtered), 1) * 50, 50), 100),
        "keyword":  kw_pct if kw_pct <= 80 else max(100 - kw_pct, 20),
        "sources":  min(src_info["diversity"] / 10 * 100, 100),
        "temporal": min(dt_info["total_days"] / max(days, 1) * 100, 100),
        "topics":   min(len(topic_list) / 10 * 100, 100) if topic_list else 0,
        "coverage": composite,
    }

    tab_defs = [
        ("tfidf",    "📝 TF-IDF Search",   algo_scores["tfidf"]),
        ("keyword",  "🎯 Keyword Match",   algo_scores["keyword"]),
        ("sources",  "📊 Source Analysis",  algo_scores["sources"]),
        ("temporal", "📅 Temporal",         algo_scores["temporal"]),
        ("topics",   "🏷️ Topics",           algo_scores["topics"]),
        ("coverage", "📈 Coverage Score",   algo_scores["coverage"]),
    ]
    tab_defs.sort(key=lambda x: x[2], reverse=True)
    best_algo = tab_defs[0]

    st.success(f"🏆 **Best algorithm: {best_algo[1]}** — scored {best_algo[2]:.0f}/100 on this company")

    tab_labels = [f"{td[1]} ({td[2]:.0f})" for td in tab_defs]
    tabs = st.tabs(tab_labels)

    for tab_widget, (key, label, score) in zip(tabs, tab_defs):
        with tab_widget:
            if key == "tfidf":
                st.subheader("📝 TF-IDF (Term Frequency–Inverse Document Frequency)")
                st.markdown("Weighs terms by importance across the corpus. **Misses paraphrases and indirect mentions.**")
                c1, c2 = st.columns([1, 1])
                with c1:
                    st.metric("Matching articles", len(tfidf_res))
                    if tfidf_res:
                        st.metric("Avg score", f"{tfidf_avg:.3f}")
                with c2:
                    if tfidf_res:
                        sdf = pd.DataFrame([{"Article": r["title"][:50], "Score": r["tfidf_score"]} for r in tfidf_res[:10]])
                        fig = px.bar(sdf, x="Score", y="Article", orientation="h", title="Top TF-IDF Matches",
                                     color="Score", color_continuous_scale="Viridis")
                        fig.update_layout(height=350, yaxis=dict(autorange="reversed"))
                        st.plotly_chart(fig, use_container_width=True)
                if tfidf_res:
                    with st.expander("View articles"):
                        for r in tfidf_res[:10]:
                            st.markdown(f"**{r['title']}** — score: {r['tfidf_score']:.4f}")
                            st.caption(f"{r.get('source','N/A')} | {r.get('published_date','')}")

            elif key == "keyword":
                st.subheader("🎯 Simple Keyword Match")
                st.markdown("Pure string matching — no ML. Shows the baseline all algorithms should beat.")
                c1, c2 = st.columns([1, 1])
                with c1:
                    st.metric("Matches", len(kw_res))
                    if filtered:
                        st.metric("Match rate", f"{kw_pct:.0f}%")
                with c2:
                    if tfidf_res or kw_res:
                        cdf = pd.DataFrame({"Method": ["Keyword", "TF-IDF"], "Found": [len(kw_res), len(tfidf_res)]})
                        fig = px.bar(cdf, x="Method", y="Found", title="Keyword vs TF-IDF", color="Method", text="Found")
                        st.plotly_chart(fig, use_container_width=True)
                if kw_res:
                    with st.expander("View matches"):
                        for r in kw_res[:10]:
                            st.markdown(f"**{r['title']}** — match: {r['keyword_score']:.3f}")

            elif key == "sources":
                st.subheader("📊 Source Diversity Analysis")
                st.markdown("MNCs appear in 20+ sources; niche startups might appear in 0-2.")
                if src_info["sources"]:
                    c1, c2 = st.columns([1, 1])
                    with c1:
                        sdf = pd.DataFrame([{"Source": k, "Articles": v} for k, v in src_info["sources"].items()])
                        fig = px.pie(sdf, names="Source", values="Articles", title=f"Sources for {selected_company}")
                        st.plotly_chart(fig, use_container_width=True)
                    with c2:
                        st.metric("Diversity", src_info["diversity"])
                        st.metric("Top source", f"{src_info['top_source'][0]} ({src_info['top_source'][1]})")
                else:
                    st.warning("No sources — company has zero coverage")

            elif key == "temporal":
                st.subheader("📅 Temporal Distribution")
                st.markdown("Bursty = single event. Spread = strong presence.")
                if dt_info["dates"]:
                    ddf = pd.DataFrame([{"Date": k, "Articles": v} for k, v in dt_info["dates"].items()])
                    ddf["Date"] = pd.to_datetime(ddf["Date"])
                    fig = px.bar(ddf, x="Date", y="Articles", title=f"Timeline: {selected_company}",
                                 color="Articles", color_continuous_scale="Blues")
                    st.plotly_chart(fig, use_container_width=True)
                    if len(dt_info["dates"]) <= 2:
                        st.warning("⚡ Bursty coverage — likely tied to a single event")
                    elif len(dt_info["dates"]) >= 10:
                        st.success("📈 Consistent coverage — strong media presence")
                else:
                    st.warning("No date data available")

            elif key == "topics":
                st.subheader("🏷️ Topic Keywords")
                st.markdown("Top terms extracted via TF-IDF. Unreliable with < 5 articles.")
                if topic_list:
                    tdf = pd.DataFrame(topic_list, columns=["Keyword", "Importance"])
                    fig = px.bar(tdf.head(15), x="Importance", y="Keyword", orientation="h",
                                 title=f"Top Keywords: {selected_company}",
                                 color="Importance", color_continuous_scale="Sunset")
                    fig.update_layout(yaxis=dict(autorange="reversed"))
                    st.plotly_chart(fig, use_container_width=True)
                    if len(filtered) < 5:
                        st.warning("⚠️ < 5 articles — topic extraction is unreliable")
                else:
                    st.error(topics.get("error", "Cannot extract topics"))

            elif key == "coverage":
                st.subheader("📈 Composite Coverage Score")
                st.markdown("Combines volume, source diversity, temporal spread, and TF-IDF relevance into 0-100.")
                c1, c2 = st.columns([1, 1])
                with c1:
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number", value=composite,
                        title={"text": "Coverage Score"},
                        gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#4ECDC4"},
                               "steps": [{"range": [0, 30], "color": "#FF6B6B"},
                                         {"range": [30, 60], "color": "#FFD93D"},
                                         {"range": [60, 100], "color": "#6BCB77"}]}))
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
                with c2:
                    for lbl, sc in {"📰 Volume": volume_score, "📡 Diversity": diversity_score,
                                      "📅 Temporal": temporal_score, "🎯 Relevance": relevance_score}.items():
                        st.progress(min(sc / 100, 1.0), text=f"{lbl}: {sc:.0f}/100")
                    if composite < 30:
                        st.error("🔴 Poor coverage — ML models unreliable")
                    elif composite < 60:
                        st.warning("🟡 Moderate — algorithms will disagree")
                    else:
                        st.success("🟢 Strong — all algorithms produce meaningful results")


# ═══════════════════════════════════════════════════════════════
# STARTUP vs MNC COMPARISON
# ═══════════════════════════════════════════════════════════════
st.divider()
st.header("⚡ Startup vs MNC Comparison")

cc1, cc2 = st.columns(2)
with cc1:
    startup_names = [c.name for c in startups if len(articles_data.get(c.name, [])) > 0]
    compare_startup = st.selectbox("Startup", startup_names, key="cmp_s") if startup_names else None
with cc2:
    compare_mnc = st.selectbox("MNC", [c.name for c in mncs], key="cmp_m")

if compare_startup and compare_mnc:
    s_arts = filter_articles_by_days(articles_data.get(compare_startup, []), days)
    m_arts = filter_articles_by_days(articles_data.get(compare_mnc, []), days)
    cmp = pd.DataFrame({
        "Metric": ["Articles", "Sources", "News Days", "TF-IDF Hits", "Keyword Hits"],
        compare_startup: [len(s_arts), run_source_analysis(s_arts)["diversity"],
                          run_date_analysis(s_arts)["total_days"],
                          len(run_tfidf_on_articles(s_arts, compare_startup)),
                          len(run_keyword_match(s_arts, compare_startup))],
        compare_mnc: [len(m_arts), run_source_analysis(m_arts)["diversity"],
                      run_date_analysis(m_arts)["total_days"],
                      len(run_tfidf_on_articles(m_arts, compare_mnc)),
                      len(run_keyword_match(m_arts, compare_mnc))],
    })
    fig = go.Figure()
    fig.add_trace(go.Bar(name=compare_startup, x=cmp["Metric"], y=cmp[compare_startup], marker_color="#4ECDC4"))
    fig.add_trace(go.Bar(name=compare_mnc, x=cmp["Metric"], y=cmp[compare_mnc], marker_color="#FF6B6B"))
    fig.update_layout(barmode="group", title="Head-to-Head", height=400)
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(cmp, use_container_width=True)

# ── Split overview ─────────────────────────────────────────────
st.divider()
st.header("🔬 Dataset Split Overview")
splits = get_splits()
if splits:
    hdata = []
    for sn in ["train", "val", "test"]:
        for cn in splits.get(sn, []):
            hdata.append({"Company": cn, "Split": sn.upper(),
                          "Articles": len(filter_articles_by_days(articles_data.get(cn, []), days)),
                          "Sector": splits.get(f"{sn}_sectors", {}).get(cn, "other")})
    hdf = pd.DataFrame(hdata)
    if not hdf.empty:
        fig = px.bar(hdf, x="Company", y="Articles", color="Split",
                     title=f"Articles per Company by Split (last {days} days)",
                     color_discrete_map={"TRAIN": "#4ECDC4", "VAL": "#FFD93D", "TEST": "#FF6B6B"}, height=500)
        fig.update_layout(xaxis_tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
        summary = hdf.groupby("Split").agg(
            Companies=("Company", "count"), Total=("Articles", "sum"),
            Avg=("Articles", "mean"), Zero=("Articles", lambda x: (x == 0).sum())).round(1)
        st.dataframe(summary, use_container_width=True)
