#Streamlit Web App for Research Lab Meeting Memory System

Tabs:
  1. Upload & Process   → Upload JSON / paste transcript, rebuild index
  2. Meeting Insights   → Per-meeting: keywords, entities, papers, deadlines
  3. Semantic Search    → Query-based chunk retrieval
  4. Q&A (RAG)          → Natural language Q&A over all meetings
  5. Cross-Meeting Trends → Clusters + recurring ideas

Run: streamlit run app.py
"""

import json
import sys
import time
import logging
from pathlib import Path

import streamlit as st

# ── Path setup ───────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Research Lab Memory System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 0.25rem;
    }
    .sub-header {
        font-size: 0.95rem;
        color: #555;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #f8f9fc;
        border-radius: 8px;
        padding: 1rem;
        border-left: 4px solid #4f46e5;
    }
    .keyword-tag {
        display: inline-block;
        background: #e0e7ff;
        color: #3730a3;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.8rem;
        margin: 2px;
    }
    .deadline-badge {
        background: #fef3c7;
        border-left: 3px solid #f59e0b;
        padding: 0.4rem 0.75rem;
        border-radius: 0 6px 6px 0;
        font-size: 0.85rem;
        margin: 4px 0;
    }
    .paper-badge {
        background: #ecfdf5;
        border-left: 3px solid #10b981;
        padding: 0.4rem 0.75rem;
        border-radius: 0 6px 6px 0;
        font-size: 0.85rem;
        margin: 4px 0;
    }
    .search-result {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .trend-card {
        background: #faf5ff;
        border: 1px solid #d8b4fe;
        border-radius: 8px;
        padding: 0.75rem;
        margin: 0.5rem 0;
    }
    .answer-box {
        background: #f0fdf4;
        border: 1px solid #86efac;
        border-radius: 8px;
        padding: 1.25rem;
        font-size: 1rem;
        line-height: 1.6;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# ── Session state init ────────────────────────────────────────────────────────
def init_session():
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = None
    if "pipeline_ready" not in st.session_state:
        st.session_state.pipeline_ready = False
    if "search_results" not in st.session_state:
        st.session_state.search_results = []
    if "qa_history" not in st.session_state:
        st.session_state.qa_history = []


init_session()


# ── Pipeline loader (cached) ─────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_pipeline(llm_mode: str = "local"):
    """Load the full pipeline (cached across reruns)."""
    from pipeline import ResearchMemoryPipeline
    pipe = ResearchMemoryPipeline(
        data_dir=str(ROOT / "data"),
        llm_mode=llm_mode,
    )
    pipe.load(str(ROOT / "data" / "meetings.json"))
    return pipe


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">🧠 Research Lab Meeting Memory System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Extract insights, search semantically, and ask questions across your research meetings.</div>', unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    llm_mode = st.selectbox(
        "LLM Mode",
        ["local", "openai"],
        help="'local' uses flan-t5 (no API key). 'openai' uses GPT-3.5 (needs OPENAI_API_KEY)."
    )
    top_k = st.slider("Search Top-K results", 3, 10, 5)
    st.markdown("---")
    st.markdown("### 📂 Data")
    st.markdown("Default: `data/meetings.json`")

    if st.button("🔄 Rebuild Index", use_container_width=True):
        st.cache_resource.clear()
        st.session_state.pipeline_ready = False
        st.rerun()

    st.markdown("---")
    st.markdown("### 📊 Status")
    if st.session_state.pipeline_ready:
        pipe = st.session_state.pipeline
        st.success(f"✅ {len(pipe.meetings)} meetings loaded")
        st.info(f"🔍 {pipe.embedder.index.ntotal} chunks indexed")
    else:
        st.warning("⏳ Loading pipeline...")

# ── Auto-load pipeline ────────────────────────────────────────────────────────
if not st.session_state.pipeline_ready:
    with st.spinner("Loading pipeline... (first run downloads embedding model ~90MB)"):
        try:
            pipe = load_pipeline(llm_mode)
            st.session_state.pipeline = pipe
            st.session_state.pipeline_ready = True
            st.rerun()
        except Exception as e:
            st.error(f"Pipeline load failed: {e}")
            st.stop()

pipe = st.session_state.pipeline

# ── Main tabs ─────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋 Meeting Insights",
    "🔍 Semantic Search",
    "💬 Q&A (RAG)",
    "📈 Trends & Clusters",
    "➕ Add Meeting",
])


# ════════════════════════════════════════════════════════════
# TAB 1: Meeting Insights
# ════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### 📋 Meeting Insights")

    meeting_options = {m["id"]: f"{m['date']} — {m['title']}" for m in pipe.meetings}
    selected_id = st.selectbox("Select a meeting", list(meeting_options.keys()),
                                format_func=lambda x: meeting_options[x])

    if selected_id:
        m = pipe.get_meeting_insights(selected_id)
        if m:
            st.markdown(f"#### {m['title']}")
            st.caption(f"📅 {m['date']} | {len(m.get('sentences', []))} sentences | {len(m.get('chunks', []))} chunks")

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Keywords", len(m.get("keywords", [])))
            col2.metric("Papers", len(m.get("papers_discussed", [])))
            col3.metric("Deadlines", len(m.get("deadlines", [])))
            col4.metric("Key Ideas", len(m.get("key_ideas", [])))

            # Summary
            st.markdown("**📝 Summary**")
            summary = m.get("summary", "No summary available.")
            st.info(summary)

            # Two columns for details
            left, right = st.columns(2)

            with left:
                # Keywords
                st.markdown("**🏷️ Keywords (TF-IDF)**")
                kw_html = " ".join([
                    f'<span class="keyword-tag">{kw} ({score})</span>'
                    for kw, score in m.get("keywords", [])[:10]
                ])
                st.markdown(kw_html, unsafe_allow_html=True)

                # Research topics
                st.markdown("**🔬 Research Topics**")
                topics = m.get("research_topics", [])
                if topics:
                    for t in topics:
                        st.markdown(f"- {t}")
                else:
                    st.caption("None detected")

                # Named entities
                st.markdown("**👤 Named Entities**")
                entities = m.get("entities", {})
                if any(entities.values()):
                    for etype, values in entities.items():
                        if values:
                            st.caption(f"**{etype}**: {', '.join(values)}")
                else:
                    st.caption("None detected")

            with right:
                # Papers
                st.markdown("**📄 Papers Discussed**")
                papers = m.get("papers_discussed", [])
                if papers:
                    for p in papers:
                        st.markdown(f'<div class="paper-badge">📎 {p}</div>', unsafe_allow_html=True)
                else:
                    st.caption("None detected")

                # Deadlines
                st.markdown("**⏰ Deadlines**")
                deadlines = m.get("deadlines", [])
                if deadlines:
                    for d in deadlines:
                        st.markdown(f'<div class="deadline-badge">📌 {d}</div>', unsafe_allow_html=True)
                else:
                    st.caption("None detected")

                # Key ideas
                st.markdown("**💡 Key Ideas**")
                ideas = m.get("key_ideas", [])
                if ideas:
                    for idea in ideas:
                        st.markdown(f"> {idea}")
                else:
                    st.caption("None detected")

            # Full transcript expander
            with st.expander("📜 Full Transcript"):
                st.text(m.get("clean_text", ""))


# ════════════════════════════════════════════════════════════
# TAB 2: Semantic Search
# ════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### 🔍 Semantic Search")
    st.caption("Find the most relevant meeting passages for any query using sentence embeddings.")

    query = st.text_input(
        "Search query",
        placeholder="e.g. multilingual BERT cross-lingual transfer",
        key="search_query"
    )

    col_a, col_b = st.columns([1, 5])
    search_btn = col_a.button("🔍 Search", use_container_width=True)

    if search_btn and query:
        with st.spinner("Searching..."):
            results = pipe.search(query, top_k=top_k)
            st.session_state.search_results = results

    if st.session_state.search_results:
        results = st.session_state.search_results
        st.markdown(f"**{len(results)} results** for *'{query or 'last query'}'*")

        for i, r in enumerate(results, 1):
            with st.container():
                score_pct = int(r["score"] * 100)
                score_color = "#16a34a" if score_pct > 70 else "#ca8a04" if score_pct > 40 else "#dc2626"
                st.markdown(
                    f'<div class="search-result">'
                    f'<b>#{i}</b> &nbsp; '
                    f'<span style="color:{score_color}">▶ {score_pct}% match</span> &nbsp; | &nbsp; '
                    f'📅 <b>{r["date"]}</b> — {r["meeting_title"]}<br><br>'
                    f'{r["chunk"]}'
                    f'</div>',
                    unsafe_allow_html=True
                )

    # Quick search suggestions
    st.markdown("**Try these queries:**")
    suggestions = [
        "RAG pipeline FAISS retrieval",
        "deadline submission conference",
        "cross-lingual multilingual model",
        "data augmentation back-translation",
        "evaluation metrics F1 score",
    ]
    cols = st.columns(len(suggestions))
    for col, sug in zip(cols, suggestions):
        if col.button(sug, key=f"sug_{sug[:10]}"):
            with st.spinner("Searching..."):
                st.session_state.search_results = pipe.search(sug, top_k=top_k)
            st.rerun()


# ════════════════════════════════════════════════════════════
# TAB 3: Q&A (RAG)
# ════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### 💬 Q&A over Meeting History")
    st.caption("Ask natural language questions. The system retrieves relevant context and generates an answer.")

    question = st.text_area(
        "Your question",
        placeholder="e.g. What papers were discussed related to attention mechanisms?",
        height=80,
        key="rag_question"
    )

    ask_col, clear_col = st.columns([1, 5])
    ask_btn = ask_col.button("💬 Ask", use_container_width=True)

    if ask_btn and question:
        with st.spinner("Retrieving context and generating answer..."):
            result = pipe.ask(question)

        # Add to history
        st.session_state.qa_history.insert(0, result)

    if clear_col.button("🗑️ Clear history"):
        st.session_state.qa_history = []
        st.rerun()

    # Display Q&A history
    for i, result in enumerate(st.session_state.qa_history):
        st.markdown(f"**Q: {result['question']}**")
        st.markdown(f'<div class="answer-box">{result["answer"]}</div>', unsafe_allow_html=True)

        with st.expander(f"📚 Sources ({len(result.get('sources', []))})"):
            for s in result.get("sources", []):
                st.markdown(
                    f"**{s['meeting_title']}** ({s['date']}) — score: {s['score']}\n\n"
                    f"*{s['chunk']}*"
                )
        st.markdown("---")

    # Sample questions
    if not st.session_state.qa_history:
        st.markdown("**Sample questions to try:**")
        sample_qs = [
            "What are all the deadlines mentioned across meetings?",
            "Which papers were cited in multiple meetings?",
            "What is the RAG pipeline architecture discussed?",
            "What recurring research topics came up?",
            "Who are the team members and what are their action items?",
        ]
        for q in sample_qs:
            st.markdown(f"- *{q}*")


# ════════════════════════════════════════════════════════════
# TAB 4: Trends & Clusters
# ════════════════════════════════════════════════════════════
with tab4:
    st.markdown("### 📈 Cross-Meeting Trends & Idea Clusters")

    # Summary metrics
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Clusters", len(pipe.clusters))
    m2.metric("Recurring Trends", len(pipe.trends))
    m3.metric("Repeated Ideas", len(pipe.repeated_ideas))

    st.markdown("---")

    # Clusters
    st.markdown("#### 🗂️ Semantic Clusters")
    st.caption("Chunks grouped by semantic similarity using KMeans on embeddings.")

    for cluster in pipe.clusters:
        badge = "🔁 Recurring" if len(cluster["meetings"]) >= 2 else "📌 Single-meeting"
        with st.expander(
            f"Cluster {cluster['cluster_id']}: **{cluster['label']}** "
            f"({cluster['size']} chunks, {len(cluster['meetings'])} meetings) {badge}"
        ):
            st.caption(f"**Meetings**: {', '.join(cluster['meetings'])}")
            for chunk in cluster["chunks"][:3]:
                st.markdown(f"> {chunk[:200]}...")
            if cluster["size"] > 3:
                st.caption(f"... and {cluster['size'] - 3} more chunks")

    st.markdown("---")

    # Trends
    if pipe.trends:
        st.markdown("#### 🔁 Recurring Themes (across 2+ meetings)")
        for t in pipe.trends:
            st.markdown(
                f'<div class="trend-card">'
                f'<b>{t["label"]}</b><br>'
                f'Appears in {t["trend_strength"]} meetings: {", ".join(t["meetings"])}'
                f'</div>',
                unsafe_allow_html=True
            )
    else:
        st.info("No trends detected yet. Add more meetings to detect patterns.")

    st.markdown("---")

    # Repeated ideas
    if pipe.repeated_ideas:
        st.markdown("#### 💡 Repeated Ideas (similar content across meetings)")
        for idea in pipe.repeated_ideas[:5]:
            with st.expander(
                f"Idea discussed in {idea['occurrences']} meetings "
                f"(avg similarity: {idea['avg_similarity']:.0%})"
            ):
                st.caption(f"Meetings: {', '.join(idea['meetings'])}")
                for chunk in idea["chunks"]:
                    st.markdown(f"> {chunk[:200]}...")
    else:
        st.info("No repeated ideas detected at current similarity threshold (0.80).")


# ════════════════════════════════════════════════════════════
# TAB 5: Add Meeting
# ════════════════════════════════════════════════════════════
with tab5:
    st.markdown("### ➕ Add New Meeting")
    st.caption("Paste a new transcript or upload a JSON file to expand the corpus.")

    with st.form("add_meeting_form"):
        new_id = st.text_input("Meeting ID", placeholder="meeting_006")
        new_date = st.date_input("Meeting Date")
        new_title = st.text_input("Meeting Title", placeholder="Weekly Research Sync")
        new_transcript = st.text_area(
            "Transcript / Notes",
            placeholder="Paste meeting transcript here...",
            height=200
        )
        submitted = st.form_submit_button("➕ Add to Corpus")

    if submitted:
        if not (new_id and new_title and new_transcript):
            st.error("Meeting ID, title, and transcript are required.")
        else:
            # Load existing data
            data_path = ROOT / "data" / "meetings.json"
            with open(data_path, "r") as f:
                existing = json.load(f)

            # Check for duplicate ID
            if any(m["id"] == new_id for m in existing):
                st.error(f"Meeting ID '{new_id}' already exists. Use a different ID.")
            else:
                new_meeting = {
                    "id": new_id,
                    "date": str(new_date),
                    "title": new_title,
                    "transcript": new_transcript,
                }
                existing.append(new_meeting)

                with open(data_path, "w") as f:
                    json.dump(existing, f, indent=2)

                st.success(f"Meeting '{new_title}' added. Rebuilding index...")
                st.cache_resource.clear()
                st.session_state.pipeline_ready = False
                time.sleep(1)
                st.rerun()

    st.markdown("---")
    st.markdown("**Or upload a JSON file** (same format as `data/meetings.json`):")
    uploaded = st.file_uploader("Upload meetings JSON", type=["json"])
    if uploaded:
        try:
            new_meetings = json.load(uploaded)
            if not isinstance(new_meetings, list):
                new_meetings = [new_meetings]

            data_path = ROOT / "data" / "meetings.json"
            with open(data_path, "r") as f:
                existing = json.load(f)

            existing_ids = {m["id"] for m in existing}
            added = [m for m in new_meetings if m["id"] not in existing_ids]
            existing.extend(added)

            with open(data_path, "w") as f:
                json.dump(existing, f, indent=2)

            st.success(f"Added {len(added)} meetings. Rebuilding index...")
            st.cache_resource.clear()
            st.session_state.pipeline_ready = False
            time.sleep(1)
            st.rerun()
        except json.JSONDecodeError:
            st.error("Invalid JSON file.")
