"""
Streamlit Web App: Research Lab Meeting Memory System
======================================================
Run with: streamlit run app/streamlit_app.py

Features:
- Upload meeting transcripts (paste text or upload file)
- View extracted insights (keywords, papers, deadlines, NER)
- Semantic search across all meetings
- RAG-powered Q&A
- Cross-meeting trend analysis
- Downloadable markdown report
"""

import streamlit as st
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Page config
st.set_page_config(
    page_title="Research Lab Memory System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-box {
        background: #1e1e2e;
        border: 1px solid #333;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
    .chunk-card {
        background: #16213e;
        border-left: 3px solid #4c9be8;
        padding: 10px 14px;
        margin: 8px 0;
        border-radius: 4px;
    }
    .answer-box {
        background: #0f3460;
        border: 1px solid #4c9be8;
        border-radius: 8px;
        padding: 16px;
        margin: 12px 0;
    }
    .source-tag {
        background: #333;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.8em;
        margin: 2px;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_system():
    """Load and cache the ResearchLabMemorySystem (expensive — runs once)."""
    from main import ResearchLabMemorySystem
    return ResearchLabMemorySystem(
        embedding_model="all-MiniLM-L6-v2",
        use_api=False
    )


@st.cache_data
def ingest_sample_data(_system):
    """Ingest sample meetings (cached so it only runs once)."""
    from data.sample_meetings import SAMPLE_MEETINGS
    return _system.ingest_meetings(SAMPLE_MEETINGS, chunk_level="sentence")


def main():
    # Sidebar
    with st.sidebar:
        st.title("🧠 Lab Memory")
        st.caption("Research Meeting Intelligence System")
        st.divider()

        page = st.radio(
            "Navigation",
            ["📊 Dashboard", "🔍 Search", "💬 Ask AI", "📈 Trends", "📝 Add Meeting"],
            label_visibility="collapsed"
        )

        st.divider()
        st.caption("Powered by:")
        st.caption("• all-MiniLM-L6-v2 embeddings")
        st.caption("• FAISS vector search")
        st.caption("• TF-IDF keyword extraction")
        st.caption("• RAG pipeline")

        if st.button("🔄 Load Sample Meetings", use_container_width=True):
            st.session_state.use_sample = True

    # Load system
    with st.spinner("Loading NLP models..."):
        system = load_system()

    # Auto-load sample data
    if "ingested" not in st.session_state:
        if st.session_state.get("use_sample") or True:  # Auto-load on first visit
            with st.spinner("Ingesting sample meetings..."):
                summary = ingest_sample_data(system)
                st.session_state.ingested = True
                st.session_state.summary = summary

    # ─── DASHBOARD ───────────────────────────────────────────
    if "Dashboard" in page:
        st.title("📊 Meeting Intelligence Dashboard")

        # Top metrics
        if st.session_state.get("ingested"):
            s = st.session_state.summary
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("Meetings", s['meetings_ingested'])
            with col2:
                st.metric("Chunks Indexed", s['total_chunks_indexed'])
            with col3:
                st.metric("Papers Found", s['total_papers_found'])
            with col4:
                st.metric("Deadlines", s['total_deadlines_found'])
            with col5:
                st.metric("Mode", s['generation_mode'])

            st.divider()

        # Per-meeting cards
        st.subheader("Meeting Insights")
        for pm in system.processed_meetings:
            insights = system.get_meeting_insights(pm.meeting_id)
            with st.expander(f"📋 {pm.meeting_id}: {pm.title} | {pm.date}"):
                col1, col2 = st.columns([1, 1])

                with col1:
                    st.write("**👥 Attendees:**", ", ".join(pm.attendees))
                    st.write(f"**📝 Words:** {pm.word_count} | **Sentences:** {pm.sentence_count}")

                    if insights:
                        st.write("**🔑 Keywords:**")
                        kw_tags = " ".join(
                            f"`{kw}`" for kw, _ in insights.keywords[:8]
                        )
                        st.markdown(kw_tags)

                with col2:
                    if insights and insights.papers_cited:
                        st.write("**📄 Papers Discussed:**")
                        for p in insights.papers_cited:
                            st.write(f"  • _{p['title']}_")

                    if pm.action_items:
                        st.write("**✅ Action Items:**")
                        for item in pm.action_items[:3]:
                            st.write(f"  • {item[:100]}")

                if insights and insights.summary_sentences:
                    st.write("**📌 Summary:**")
                    for sent in insights.summary_sentences[:2]:
                        st.info(sent[:200])

        # All papers table
        st.subheader("📚 All Papers Cited")
        papers = system.get_all_papers()
        if papers:
            import pandas as pd
            df = pd.DataFrame(papers)[['title', 'authors', 'venue', 'meeting_id', 'date']]
            st.dataframe(df, use_container_width=True)

        # All deadlines
        st.subheader("⏰ All Deadlines")
        deadlines = system.get_all_deadlines()
        if deadlines:
            df_dl = pd.DataFrame(deadlines)
            cols = [c for c in ['assignee', 'task', 'date', 'meeting_id'] if c in df_dl.columns]
            st.dataframe(df_dl[cols], use_container_width=True)

        # Download report
        st.subheader("📥 Export Report")
        report = system.generate_report()
        st.download_button(
            "Download Markdown Report",
            data=report,
            file_name="meeting_memory_report.md",
            mime="text/markdown"
        )

    # ─── SEMANTIC SEARCH ────────────────────────────────────────
    elif "Search" in page:
        st.title("🔍 Semantic Search")
        st.caption("Find relevant content across all meetings using natural language")

        query = st.text_input(
            "Search query",
            placeholder="e.g., transformer attention mechanism for multimodal learning"
        )

        col1, col2 = st.columns([3, 1])
        with col1:
            top_k = st.slider("Number of results", 3, 10, 5)
        with col2:
            meeting_filter = st.selectbox(
                "Filter by meeting",
                ["All"] + [pm.meeting_id for pm in system.processed_meetings]
            )

        if query:
            filter_id = None if meeting_filter == "All" else meeting_filter
            with st.spinner("Searching..."):
                results = system.search(query, top_k=top_k, filter_meeting_id=filter_id)

            st.write(f"**{len(results)} results** for: _{query}_")

            for i, r in enumerate(results, 1):
                with st.container():
                    col1, col2, col3 = st.columns([1, 3, 1])
                    with col1:
                        st.write(f"**#{i}**")
                        st.metric("Score", f"{r['score']:.3f}")
                    with col2:
                        st.write(f"**{r.get('meeting_id')}** | {r.get('date', '')} | {r.get('title', '')[:50]}")
                        st.write(r['text'][:300])
                    with col3:
                        st.caption(f"Type: {r.get('chunk_type', '')}")
                        if r.get('speaker'):
                            st.caption(f"Speaker: {r['speaker']}")
                    st.divider()

        # Quick examples
        st.subheader("Try these searches:")
        example_queries = [
            "RAG retrieval augmented generation",
            "cross-modal attention multimodal",
            "conference deadline submission",
            "FAISS vector database",
            "sentiment analysis accuracy benchmark"
        ]
        cols = st.columns(len(example_queries))
        for i, eq in enumerate(example_queries):
            with cols[i]:
                if st.button(eq, key=f"eq_{i}", use_container_width=True):
                    st.session_state.search_query = eq
                    st.rerun()

    # ─── RAG Q&A ────────────────────────────────────────────────
    elif "Ask AI" in page:
        st.title("💬 Ask AI About Your Meetings")
        st.caption("RAG-powered Q&A grounded in your meeting records")

        # Chat history
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        # Display chat
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
                if msg["role"] == "assistant" and "sources" in msg:
                    st.caption(f"Sources: {', '.join(msg['sources'])}")

        # Input
        question = st.chat_input("Ask anything about the meetings...")

        if question:
            st.session_state.chat_history.append({"role": "user", "content": question})

            with st.chat_message("user"):
                st.write(question)

            with st.chat_message("assistant"):
                with st.spinner("Retrieving relevant context and generating answer..."):
                    result = system.ask(question, top_k=5)

                st.write(result.answer)
                st.caption(f"Sources: {', '.join(result.sources)} | "
                          f"Confidence: {result.confidence:.0%} | "
                          f"Mode: {result.generation_mode}")

                # Show retrieved chunks (expandable)
                with st.expander("📎 Retrieved context chunks"):
                    for chunk in result.retrieved_chunks[:3]:
                        st.write(f"**[{chunk['meeting_id']} | Score: {chunk['score']:.3f}]**")
                        st.write(chunk['text'][:200])
                        st.divider()

            st.session_state.chat_history.append({
                "role": "assistant",
                "content": result.answer,
                "sources": result.sources
            })

        # Suggested questions
        if not st.session_state.chat_history:
            st.subheader("Suggested questions:")
            suggested = [
                "What papers were discussed about transformers?",
                "What are all the deadlines across meetings?",
                "Who is working on the RAG pipeline?",
                "What datasets are being used in the project?",
                "What is the target conference and its deadline?",
                "What are the main research directions of the lab?"
            ]
            for i in range(0, len(suggested), 2):
                col1, col2 = st.columns(2)
                with col1:
                    if i < len(suggested):
                        if st.button(suggested[i], key=f"sq_{i}"):
                            st.session_state.suggested_q = suggested[i]
                with col2:
                    if i + 1 < len(suggested):
                        if st.button(suggested[i+1], key=f"sq_{i+1}"):
                            st.session_state.suggested_q = suggested[i+1]

    # ─── TRENDS ────────────────────────────────────────────────
    elif "Trends" in page:
        st.title("📈 Cross-Meeting Trends")
        st.caption("Detect repeated themes, trending topics, and research evolution")

        with st.spinner("Analyzing trends across all meetings..."):
            try:
                trends = system.analyze_trends()
            except Exception as e:
                st.warning(f"Full clustering unavailable: {e}. Showing basic analysis.")
                trends = system._basic_trend_analysis()

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🔄 Repeated Themes")
            st.caption("Topics appearing in 2+ meetings")
            for theme in trends.get('repeated_themes', [])[:8]:
                theme_name = theme.get('theme') or theme.get('keyword', '')
                count = theme.get('frequency') or theme.get('count', 0)
                meetings = theme.get('meetings', [])
                st.write(f"**{theme_name}** — Count: {count}")
                if meetings:
                    st.caption(f"Meetings: {', '.join(meetings)}")

        with col2:
            st.subheader("📅 Topic Timeline")
            for entry in trends.get('timeline', []):
                kws = entry.get('top_keywords') or entry.get('new_topics', [])
                st.write(f"**[{entry.get('date', '')}] {entry.get('meeting_id')}**")
                st.write(entry.get('title', ''))
                if kws:
                    st.caption(f"Topics: {', '.join(kws[:5])}")
                st.divider()

        # Trending keywords table
        if trends.get('trending_keywords'):
            st.subheader("📊 Trending Keywords")
            import pandas as pd
            trending_data = []
            for kw in trends['trending_keywords'][:10]:
                trending_data.append({
                    "Keyword": kw['keyword'],
                    "Meetings": kw['total_meetings'],
                    "Trend": kw.get('trend', 'stable')
                })
            if trending_data:
                st.dataframe(pd.DataFrame(trending_data), use_container_width=True)

    # ─── ADD MEETING ────────────────────────────────────────────
    elif "Add Meeting" in page:
        st.title("📝 Add New Meeting")
        st.caption("Paste a transcript to add it to the memory system")

        with st.form("add_meeting_form"):
            col1, col2 = st.columns(2)
            with col1:
                meeting_id = st.text_input("Meeting ID", placeholder="M005")
                title = st.text_input("Title", placeholder="Weekly Lab Meeting")
            with col2:
                date = st.date_input("Date")
                attendees = st.text_input("Attendees (comma-separated)",
                                         placeholder="Dr. Sharma, Priya, Raju")

            transcript = st.text_area(
                "Meeting Transcript",
                placeholder="Dr. Sharma: Today we discuss...\nPriya: I've been working on...",
                height=300
            )

            submitted = st.form_submit_button("Add Meeting to Memory", type="primary")

            if submitted and transcript:
                new_meeting = {
                    "meeting_id": meeting_id or "NEW001",
                    "date": str(date),
                    "title": title or "Untitled Meeting",
                    "attendees": [a.strip() for a in attendees.split(",") if a.strip()],
                    "transcript": transcript
                }

                all_meetings = [
                    {"meeting_id": pm.meeting_id, "date": pm.date,
                     "title": pm.title, "attendees": pm.attendees,
                     "transcript": pm.raw_transcript}
                    for pm in system.processed_meetings
                ] + [new_meeting]

                with st.spinner("Re-indexing with new meeting..."):
                    summary = system.ingest_meetings(all_meetings, chunk_level="sentence")
                    st.session_state.summary = summary
                    st.cache_data.clear()

                st.success(f"✅ Meeting {meeting_id} added! "
                          f"Now {summary['meetings_ingested']} meetings indexed.")


if __name__ == "__main__":
    main()
