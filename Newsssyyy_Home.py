"""
Newsssyyy — Live Search home page.
Type any company name, pick a timeline, and see 6 ML algorithms compared in real-time.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from collections import Counter
import json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from src.data_loader import Company
from src.news_scraper import scrape_google_news_rss, Article

st.set_page_config(
    page_title="Newsssyyy",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Helper functions ───────────────────────────────────────────
def filter_by_days(articles: list, days: int) -> list:
    if not articles or days >= 30:
        return articles
    cutoff = datetime.now() - timedelta(days=days)
    filtered = []
    for a in articles:
        pub = a.get("published_date", "")
        if not pub:
            filtered.append(a)
            continue
        try:
            if datetime.strptime(pub[:10], "%Y-%m-%d") >= cutoff:
                filtered.append(a)
        except ValueError:
            filtered.append(a)
    return filtered


def tfidf_search(articles: list, query: str) -> list:
    if len(articles) < 2:
        return [dict(a, tfidf_score=1.0) for a in articles]
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    texts = [f"{a.get('title','')} {a.get('snippet','')}" for a in articles]
    try:
        vec = TfidfVectorizer(max_features=3000, stop_words="english", ngram_range=(1, 2))
        mat = vec.fit_transform(texts)
        q = vec.transform([query])
        sims = cosine_similarity(q, mat).flatten()
        results = []
        for i in sims.argsort()[::-1]:
            if sims[i] > 0.01:
                results.append(dict(articles[i], tfidf_score=round(float(sims[i]), 4)))
        return results[:20]
    except Exception:
        return []


def keyword_search(articles: list, query: str) -> list:
    words = set(w.lower() for w in query.split() if len(w) > 2)
    results = []
    for a in articles:
        text = f"{a.get('title','')} {a.get('snippet','')}".lower()
        hits = sum(1 for w in words if w in text)
        if hits > 0:
            results.append(dict(a, keyword_score=round(hits / max(len(words), 1), 3)))
    return sorted(results, key=lambda x: x["keyword_score"], reverse=True)[:20]


def source_analysis(articles: list) -> dict:
    if not articles:
        return {"sources": {}, "diversity": 0}
    counts = Counter(a.get("source", "Unknown") for a in articles)
    return {"sources": dict(counts.most_common(15)), "diversity": len(counts)}


def date_analysis(articles: list) -> dict:
    if not articles:
        return {"dates": {}, "total_days": 0}
    counts = Counter(a.get("published_date", "")[:10] for a in articles if a.get("published_date"))
    return {"dates": dict(sorted(counts.items())), "total_days": len(counts)}


def topic_extraction(articles: list) -> list:
    if len(articles) < 3:
        return []
    from sklearn.feature_extraction.text import TfidfVectorizer
    texts = [f"{a.get('title','')} {a.get('snippet','')}" for a in articles]
    try:
        vec = TfidfVectorizer(max_features=500, stop_words="english", max_df=0.9, min_df=1)
        mat = vec.fit_transform(texts)
        names = vec.get_feature_names_out()
        means = mat.mean(axis=0).A1
        top = means.argsort()[-15:][::-1]
        return [(names[i], round(float(means[i]), 4)) for i in top]
    except Exception:
        return []


# ═══════════════════════════════════════════════════════════════
# PAGE
# ═══════════════════════════════════════════════════════════════
st.title("📰 Newsssyyy")
st.caption("Real-time news intelligence — powered by ML")
st.markdown("""
Type **any company name** below, pick a timeline, and instantly see how
**6 different ML algorithms** analyze the news coverage.
""")

# ── Controls ───────────────────────────────────────────────────
col_search, col_timeline = st.columns([3, 1])

with col_search:
    query = st.text_input(
        "🏢 Company Name",
        placeholder="e.g., Zomato, Flipkart, Byju's, PhonePe, any company...",
        help="Type any company name — we'll search Google News live",
    )

with col_timeline:
    timeline_map = {
        "1 day": 1, "2 days": 2, "3 days": 3, "4 days": 4,
        "5 days": 5, "6 days": 6, "7 days": 7,
        "15 days": 15, "30 days": 30,
    }
    timeline = st.selectbox("📅 Timeline", list(timeline_map.keys()), index=8)
    days = timeline_map[timeline]

# ── Fetch and analyze ──────────────────────────────────────────
if query:
    company = Company(
        name=query.strip(),
        founding_year="N/A",
        sector="User Query",
        founders="N/A",
        description=f"User searched for: {query}",
        company_type="search",
        search_terms=[query.strip()],
    )

    with st.spinner(f"🔎 Fetching news for **{query}** (last {days} days)..."):
        raw_articles = scrape_google_news_rss(company, last_n_days=days)
        articles = [a.to_dict() for a in raw_articles]

    # ── Metrics row ────────────────────────────────────────────
    st.divider()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("📰 Articles Found", len(articles))
    src = source_analysis(articles)
    m2.metric("📡 Unique Sources", src["diversity"])
    dt = date_analysis(articles)
    m3.metric("📅 Days with News", dt["total_days"])
    m4.metric("🕐 Timeline", f"{days} day(s)")

    if not articles:
        st.error(f"""
        **No articles found** for **"{query}"** in the last {days} day(s).

        This could mean:
        - The company has very low media coverage (common for early-stage startups)
        - The name is too generic or ambiguous
        - Google News didn't index articles for this term recently

        💡 **Try**: expanding the timeline to 30 days, or using the company's full official name.
        """)
    else:
        # ── Pre-compute ALL algorithm results ──────────────────
        tfidf_res = tfidf_search(articles, query)
        kw_res = keyword_search(articles, query)
        topics = topic_extraction(articles)
        pct = len(kw_res) / len(articles) * 100 if articles else 0

        # Coverage sub-scores
        v_score = min(len(articles) / 50 * 100, 100)
        d_score = min(src["diversity"] / 10 * 100, 100)
        t_score = min(dt["total_days"] / max(days, 1) * 100, 100)
        r_score = (len(tfidf_res) / max(len(articles), 1)) * 100 if articles else 0
        composite = round((v_score + d_score + t_score + r_score) / 4, 1)

        # ── Score each algorithm 0-100 for ranking ─────────────
        algo_scores = {}
        # TF-IDF: average relevance score * weight for number of results
        tfidf_avg = sum(r["tfidf_score"] for r in tfidf_res) / max(len(tfidf_res), 1)
        algo_scores["tfidf"] = min((tfidf_avg * 200) + min(len(tfidf_res) / max(len(articles), 1) * 50, 50), 100)
        # Keyword: match rate, but penalize over-matching
        kw_rate = pct
        algo_scores["keyword"] = kw_rate if kw_rate <= 80 else max(100 - kw_rate, 20)
        # Source diversity
        algo_scores["sources"] = min(src["diversity"] / 10 * 100, 100)
        # Temporal spread
        algo_scores["temporal"] = min(dt["total_days"] / max(days, 1) * 100, 100)
        # Topics: binary success + keyword count
        algo_scores["topics"] = min(len(topics) / 10 * 100, 100) if topics else 0
        # Coverage: composite itself
        algo_scores["coverage"] = composite

        # ── Build tab definitions in best-first order ──────────
        tab_defs = [
            ("tfidf",    "📝 TF-IDF",   algo_scores["tfidf"]),
            ("keyword",  "🎯 Keyword",  algo_scores["keyword"]),
            ("sources",  "📊 Sources",  algo_scores["sources"]),
            ("temporal", "📅 Timeline", algo_scores["temporal"]),
            ("topics",   "🏷️ Topics",   algo_scores["topics"]),
            ("coverage", "📈 Score",    algo_scores["coverage"]),
        ]
        tab_defs.sort(key=lambda x: x[2], reverse=True)

        best_algo = tab_defs[0]
        st.header(f"🤖 ML Analysis Results for: {query}")
        st.success(f"🏆 **Best algorithm: {best_algo[1]}** — scored {best_algo[2]:.0f}/100 on this company")

        tab_labels = [f"{td[1]} ({td[2]:.0f})" for td in tab_defs]
        tabs = st.tabs(tab_labels)

        # ── Render each tab based on its key ───────────────────
        for tab_widget, (key, label, score) in zip(tabs, tab_defs):
            with tab_widget:
                if key == "tfidf":
                    st.subheader("📝 TF-IDF Retrieval")
                    col_desc, col_verdict = st.columns([2, 1])
                    with col_desc:
                        st.markdown("""
                        **How it works**: Builds a term-frequency matrix across all fetched articles,
                        then ranks by cosine similarity to the company name query.

                        **Strength**: Fast, interpretable, works well when articles mention the exact name.
                        **Weakness**: Fails on paraphrases like "the quick commerce giant" instead of "Zepto".
                        """)
                    with col_verdict:
                        st.metric("Relevant articles", len(tfidf_res))
                        if tfidf_res:
                            st.metric("Avg relevance", f"{tfidf_avg:.3f}")
                            if tfidf_avg > 0.3:
                                st.success("✅ Strong signal")
                            elif tfidf_avg > 0.1:
                                st.warning("⚠️ Moderate signal")
                            else:
                                st.error("🔴 Weak signal")
                    if tfidf_res:
                        df = pd.DataFrame([{"Title": r["title"][:60], "Score": r["tfidf_score"],
                                            "Source": r.get("source", "")} for r in tfidf_res[:10]])
                        fig = px.bar(df, x="Score", y="Title", color="Source", orientation="h",
                                     title="Top TF-IDF Results")
                        fig.update_layout(height=400, yaxis=dict(autorange="reversed"))
                        st.plotly_chart(fig, use_container_width=True)
                        with st.expander("📄 Full article list"):
                            for r in tfidf_res:
                                st.markdown(f"**{r['title']}** — score: {r['tfidf_score']}")
                                st.caption(f"{r.get('source', '')} | {r.get('published_date', '')}")
                                if r.get("url"):
                                    st.markdown(f"[Read article]({r['url']})")
                                st.markdown("---")

                elif key == "keyword":
                    st.subheader("🎯 Simple Keyword Match")
                    col_desc, col_verdict = st.columns([2, 1])
                    with col_desc:
                        st.markdown("""
                        **How it works**: Checks if any word from the company name appears in the article text.
                        No ML at all — pure string matching.

                        **Strength**: Never misses an exact mention.
                        **Weakness**: Massively over-matches generic names (e.g., "Even", "Seven", "Jar").
                        """)
                    with col_verdict:
                        st.metric("Matches", len(kw_res))
                        st.metric("Match %", f"{pct:.0f}%")
                        if pct > 80:
                            st.warning("⚠️ Likely over-matching")
                        elif pct > 30:
                            st.success("✅ Reasonable match rate")
                        else:
                            st.info("ℹ️ Selective matching")
                    compare_df = pd.DataFrame({
                        "Algorithm": ["TF-IDF", "Keyword Match"],
                        "Articles Found": [len(tfidf_res), len(kw_res)],
                        "Precision Proxy": [
                            round(sum(r["tfidf_score"] for r in tfidf_res) / max(len(tfidf_res), 1), 3),
                            round(sum(r["keyword_score"] for r in kw_res) / max(len(kw_res), 1), 3),
                        ],
                    })
                    fig = px.bar(compare_df, x="Algorithm", y="Articles Found",
                                 color="Algorithm", text="Articles Found",
                                 title="TF-IDF vs Keyword: How Many Results?",
                                 color_discrete_map={"TF-IDF": "#4ECDC4", "Keyword Match": "#FFD93D"})
                    st.plotly_chart(fig, use_container_width=True)
                    with st.expander("📄 All articles"):
                        for a in articles:
                            st.markdown(f"**{a['title']}**")
                            st.caption(f"{a.get('source', '')} | {a.get('published_date', '')}")
                            if a.get('url'):
                                st.markdown(f"[Read article]({a['url']})")
                            st.markdown("---")

                elif key == "sources":
                    st.subheader("📊 Source Diversity Analysis")
                    col_desc, col_verdict = st.columns([2, 1])
                    with col_desc:
                        st.markdown("""
                        **How it works**: Counts how many different news sources covered this company.
                        High diversity = widely recognized. Low = niche or echo chamber.

                        MNCs typically appear in 15-25 sources. Early startups: 0-3.
                        """)
                    with col_verdict:
                        st.metric("Unique sources", src["diversity"])
                        if src["diversity"] >= 10:
                            st.success("✅ Wide coverage")
                        elif src["diversity"] >= 3:
                            st.warning("⚠️ Limited sources")
                        else:
                            st.error("🔴 Very narrow")
                    if src["sources"]:
                        sdf = pd.DataFrame([{"Source": k, "Articles": v} for k, v in src["sources"].items()])
                        c1, c2 = st.columns(2)
                        with c1:
                            fig = px.pie(sdf, names="Source", values="Articles", title="Source Distribution")
                            st.plotly_chart(fig, use_container_width=True)
                        with c2:
                            fig = px.bar(sdf, x="Articles", y="Source", orientation="h", title="Articles per Source",
                                         color="Articles", color_continuous_scale="Teal")
                            fig.update_layout(yaxis=dict(autorange="reversed"))
                            st.plotly_chart(fig, use_container_width=True)
                    with st.expander("📄 All articles"):
                        for a in articles:
                            st.markdown(f"**{a['title']}**")
                            st.caption(f"{a.get('source', '')} | {a.get('published_date', '')}")
                            if a.get('url'):
                                st.markdown(f"[Read article]({a['url']})")
                            st.markdown("---")

                elif key == "temporal":
                    st.subheader("📅 Temporal Distribution")
                    col_desc, col_verdict = st.columns([2, 1])
                    with col_desc:
                        st.markdown("""
                        **How it works**: Maps when articles were published. Reveals:
                        - **Spike** = event-driven coverage (funding round, launch)
                        - **Steady** = established media presence
                        - **Nothing** = invisible company
                        """)
                    with col_verdict:
                        st.metric("Days with coverage", dt["total_days"])
                        if dt["total_days"] >= days * 0.5:
                            st.success("✅ Consistent presence")
                        elif dt["total_days"] >= 3:
                            st.warning("⚠️ Sporadic")
                        else:
                            st.error("🔴 Near invisible")
                    if dt["dates"]:
                        ddf = pd.DataFrame([{"Date": k, "Articles": v} for k, v in dt["dates"].items()])
                        ddf["Date"] = pd.to_datetime(ddf["Date"])
                        fig = px.bar(ddf, x="Date", y="Articles", title="Article Timeline",
                                     color="Articles", color_continuous_scale="Blues")
                        st.plotly_chart(fig, use_container_width=True)
                    with st.expander("📄 All articles"):
                        for a in articles:
                            st.markdown(f"**{a['title']}**")
                            st.caption(f"{a.get('source', '')} | {a.get('published_date', '')}")
                            if a.get('url'):
                                st.markdown(f"[Read article]({a['url']})")
                            st.markdown("---")

                elif key == "topics":
                    st.subheader("🏷️ Topic Keyword Extraction")
                    col_desc, col_verdict = st.columns([2, 1])
                    with col_desc:
                        st.markdown("""
                        **How it works**: Uses TF-IDF to extract the most important terms from all articles.
                        Reveals what the company is being covered *for* — funding, product, controversy, etc.

                        **Needs 3+ articles** to work. With fewer, it produces noise.
                        """)
                    with col_verdict:
                        st.metric("Keywords found", len(topics))
                        if len(articles) < 3:
                            st.error("🔴 Too few articles")
                        elif topics:
                            st.success("✅ Topics extracted")
                    if topics:
                        tdf = pd.DataFrame(topics, columns=["Keyword", "Weight"])
                        fig = px.bar(tdf, x="Weight", y="Keyword", orientation="h",
                                     title=f"Top Keywords: {query}",
                                     color="Weight", color_continuous_scale="Sunset")
                        fig.update_layout(yaxis=dict(autorange="reversed"))
                        st.plotly_chart(fig, use_container_width=True)
                    elif len(articles) < 3:
                        st.warning("Need at least 3 articles for topic extraction")
                    with st.expander("📄 All articles"):
                        for a in articles:
                            st.markdown(f"**{a['title']}**")
                            st.caption(f"{a.get('source', '')} | {a.get('published_date', '')}")
                            if a.get('url'):
                                st.markdown(f"[Read article]({a['url']})")
                            st.markdown("---")

                elif key == "coverage":
                    st.subheader("📈 Composite Coverage Score")
                    st.markdown("Aggregates all signals into a single 0-100 score for quick comparison.")
                    c1, c2 = st.columns([1, 1])
                    with c1:
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number+delta", value=composite,
                            title={"text": f"Coverage Score: {query}"},
                            gauge={"axis": {"range": [0, 100]},
                                   "bar": {"color": "#4ECDC4"},
                                   "steps": [{"range": [0, 30], "color": "#FF6B6B"},
                                             {"range": [30, 60], "color": "#FFD93D"},
                                             {"range": [60, 100], "color": "#6BCB77"}]}))
                        fig.update_layout(height=300)
                        st.plotly_chart(fig, use_container_width=True)
                    with c2:
                        for lbl, sc in {"📰 Volume": v_score, "📡 Diversity": d_score,
                                        "📅 Temporal": t_score, "🎯 Relevance": r_score}.items():
                            st.progress(min(sc / 100, 1.0), text=f"{lbl}: {sc:.0f}/100")

                    st.divider()
                    st.subheader("🔬 Algorithm Comparison Summary")
                    summary_data = {
                        "Algorithm": [td[1] for td in tab_defs],
                        "Score": [f"{td[2]:.0f}/100" for td in tab_defs],
                        "Rank": [f"#{i+1}" for i in range(len(tab_defs))],
                        "Verdict": [],
                    }
                    for td_key, _, td_sc in tab_defs:
                        if td_sc >= 60:
                            summary_data["Verdict"].append("✅ Strong")
                        elif td_sc >= 30:
                            summary_data["Verdict"].append("⚠️ Moderate")
                        else:
                            summary_data["Verdict"].append("🔴 Weak")
                    st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)

                    with st.expander("📄 All articles"):
                        for a in articles:
                            st.markdown(f"**{a['title']}**")
                            st.caption(f"{a.get('source', '')} | {a.get('published_date', '')}")
                            if a.get('url'):
                                st.markdown(f"[Read article]({a['url']})")
                            st.markdown("---")

                    st.markdown(f"""
                    ### Key Takeaway for "{query}"
                    - **Best algorithm**: {best_algo[1]} (score: {best_algo[2]:.0f}/100)
                    - **TF-IDF** found {len(tfidf_res)} relevant articles out of {len(articles)} total
                    - **Keyword matching** found {len(kw_res)} — {'likely over-matching due to generic name' if pct > 80 else 'reasonable precision'}
                    - **Source diversity** is {'strong' if src['diversity'] >= 10 else 'limited' if src['diversity'] >= 3 else 'very low'} ({src['diversity']} sources)
                    - **Overall coverage score**: **{composite}/100** — {'models will produce reliable results' if composite >= 60 else 'results should be interpreted cautiously' if composite >= 30 else 'insufficient data for meaningful ML analysis'}
                    """)

else:
    # ── Landing state ──────────────────────────────────────────
    st.info("👆 Enter a company name above to get started!")
    st.markdown("""
    ### 💡 Try these examples:
    | Company | Expected Result |
    |---------|----------------|
    | **Zomato** | High coverage — unicorn, publicly listed |
    | **Flipkart** | Very high coverage — Walmart subsidiary |
    | **Zepto** | Good coverage — well-funded unicorn |
    | **PhonePe** | High coverage — major fintech |
    | **Lenskart** | Moderate coverage |
    | **Clairco** | Very low — niche cleantech startup |
    | **Navanc** | Near zero — early-stage fintech |

    The point? **Same ML algorithms, same code, drastically different results**
    based solely on how much media coverage exists.

    ---
    📊 **Want to explore our full dataset?** Head to the **Dataset Analysis** page in the sidebar
    to compare results across 50 startups + 5 MNCs.
    """)
