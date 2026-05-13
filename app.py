import streamlit as st
import pandas as pd
import numpy as np
import faiss
import re
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Research Lab Meeting Memory System",
    page_icon="🧠",
    layout="wide"
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>
    .main-title {
        font-size: 42px;
        font-weight: bold;
        color: #4F46E5;
        margin-bottom: 0.25rem;
    }

    .subtext {
        font-size: 16px;
        color: #4b5563;
        margin-bottom: 1rem;
    }

    .card {
        padding: 1rem;
        border-radius: 12px;
        background-color: #F9FAFB;
        margin-bottom: 1rem;
        border-left: 5px solid #4F46E5;
    }

    .keyword {
        display: inline-block;
        background-color: #E0E7FF;
        padding: 5px 10px;
        border-radius: 15px;
        margin: 4px 4px 4px 0;
        color: #3730A3;
        font-size: 14px;
    }

    .answer-box {
        background-color: #ECFDF5;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #10B981;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# HELPERS
# =========================================================

def get_default_meetings():
    return [
        {
            "id": "meeting_001",
            "date": "2026-05-01",
            "title": "RAG Pipeline Discussion",
            "transcript": """
Today we discussed FAISS vector databases and retrieval augmented generation.
Deadline for conference submission is June 10.
Research idea includes semantic search optimization.
Paper discussed: Attention Is All You Need.
""".strip(),
        },
        {
            "id": "meeting_002",
            "date": "2026-05-04",
            "title": "Multilingual NLP Meeting",
            "transcript": """
Team explored multilingual BERT and cross-lingual transfer learning.
We will perform experiments on Hindi and Marathi datasets.
Deadline for experiments is next Friday.
""".strip(),
        },
        {
            "id": "meeting_003",
            "date": "2026-05-08",
            "title": "Evaluation Metrics Meeting",
            "transcript": """
Discussion about F1 score, precision, recall and BLEU metrics.
We also discussed dashboard visualization for model evaluation.
""".strip(),
        },
        {
            "id": "meeting_004",
            "date": "2026-05-11",
            "title": "Research Planning Session",
            "transcript": """
Discussed future AI agents and autonomous systems.
Team members will work on LangChain pipelines and RAG applications.
Important paper: ReAct prompting paper.
Deadline for prototype is June 20.
""".strip(),
        },
    ]


def ensure_state():
    if "meetings" not in st.session_state:
        st.session_state.meetings = get_default_meetings()


@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


def build_index(meetings, embedder):
    texts = [m["transcript"] for m in meetings]
    embeddings = embedder.encode(texts, convert_to_numpy=True)
    embeddings = np.array(embeddings, dtype="float32")

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    return embeddings, index


def simple_summary(text, max_sentences=2, max_chars=350):
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    if not sentences:
        return "No summary available."
    summary = " ".join(sentences[:max_sentences]).strip()
    if len(summary) > max_chars:
        summary = summary[:max_chars].rsplit(" ", 1)[0] + "..."
    return summary


def extract_keywords(text, top_n=10):
    try:
        vectorizer = TfidfVectorizer(stop_words="english")
        X = vectorizer.fit_transform([text])
        scores = X.toarray()[0]
        terms = vectorizer.get_feature_names_out()
        ranked = sorted(zip(terms, scores), key=lambda x: x[1], reverse=True)
        return [term for term, score in ranked[:top_n] if score > 0]
    except Exception:
        words = re.findall(r"\b[a-zA-Z]{4,}\b", text.lower())
        seen = []
        for w in words:
            if w not in seen:
                seen.append(w)
        return seen[:top_n]


def find_deadlines(text):
    patterns = [
        r"\bJune \d{1,2}\b",
        r"\bJuly \d{1,2}\b",
        r"\bAugust \d{1,2}\b",
        r"\bSeptember \d{1,2}\b",
        r"\bOctober \d{1,2}\b",
        r"\bNovember \d{1,2}\b",
        r"\bDecember \d{1,2}\b",
        r"\bnext Friday\b",
        r"\bnext Monday\b",
        r"\bnext Tuesday\b",
        r"\bnext Wednesday\b",
        r"\bnext Thursday\b",
        r"\bnext Saturday\b",
        r"\bnext Sunday\b",
    ]

    found = []
    for pattern in patterns:
        found.extend(re.findall(pattern, text, flags=re.IGNORECASE))

    unique = []
    for item in found:
        cleaned = item.strip()
        if cleaned.lower() not in [x.lower() for x in unique]:
            unique.append(cleaned)
    return unique


def search_meetings(query, meetings, embedder, index, top_k=3):
    query_embedding = embedder.encode([query], convert_to_numpy=True)
    query_embedding = np.array(query_embedding, dtype="float32")
    distances, indices = index.search(query_embedding, min(top_k, len(meetings)))

    results = []
    for idx in indices[0]:
        if 0 <= idx < len(meetings):
            results.append(meetings[idx])
    return results


def rag_answer(question, meetings, embedder, index, top_k=2):
    retrieved = search_meetings(question, meetings, embedder, index, top_k=top_k)
    if not retrieved:
        return "I could not find any relevant meetings."

    context = "\n\n".join(
        [f"{m['title']} ({m['date']}): {m['transcript']}" for m in retrieved]
    )

    deadlines = find_deadlines(context)
    summary = simple_summary(context, max_sentences=3, max_chars=500)

    answer = f"Based on the most relevant meetings:\n\n{summary}"
    if deadlines:
        answer += "\n\nDeadlines mentioned: " + ", ".join(deadlines)
    return answer


# =========================================================
# INITIALIZE STATE
# =========================================================

ensure_state()
meetings = st.session_state.meetings

# =========================================================
# LOAD MODELS / INDEX
# =========================================================

embedder = load_embedding_model()
embeddings, index = build_index(meetings, embedder)

# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">🧠 Research Lab Meeting Memory System</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="subtext">AI-powered meeting intelligence system using NLP + Semantic Search + Retrieval</div>',
    unsafe_allow_html=True,
)

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("⚙️ Settings")

top_k = st.sidebar.slider(
    "Search Results",
    min_value=1,
    max_value=5,
    value=3,
)

st.sidebar.markdown("---")
st.sidebar.success(f"✅ Meetings Loaded: {len(meetings)}")
st.sidebar.info(f"🔍 Vector Chunks Indexed: {index.ntotal}")

# =========================================================
# TABS
# =========================================================

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "📋 Meeting Insights",
        "🔍 Semantic Search",
        "💬 RAG Q&A",
        "📈 Clustering & Trends",
    ]
)

# =========================================================
# TAB 1 - MEETING INSIGHTS
# =========================================================

with tab1:
    st.subheader("📋 Meeting Insights")

    titles = [m["title"] for m in meetings]
    selected_title = st.selectbox("Select Meeting", titles)

    meeting = next((m for m in meetings if m["title"] == selected_title), None)

    if meeting:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 📅 Date")
            st.info(meeting["date"])

            st.markdown("### 📝 Transcript")
            st.markdown(
                f"""
                <div class="card">
                {meeting["transcript"].replace("\n", "<br>")}
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col2:
            st.markdown("### 📌 AI Summary")
            st.success(simple_summary(meeting["transcript"]))

            st.markdown("### 🏷️ Keywords")
            keywords = extract_keywords(meeting["transcript"], top_n=10)
            if keywords:
                for word in keywords:
                    st.markdown(
                        f'<span class="keyword">{word}</span>',
                        unsafe_allow_html=True,
                    )
            else:
                st.info("No keywords found.")

            st.markdown("### ⏰ Deadlines")
            deadlines = find_deadlines(meeting["transcript"])
            if deadlines:
                for d in deadlines:
                    st.warning(f"📌 {d}")
            else:
                st.info("No deadlines found")

# =========================================================
# TAB 2 - SEMANTIC SEARCH
# =========================================================

with tab2:
    st.subheader("🔍 Semantic Search")

    query = st.text_input("Search meetings", placeholder="e.g. multilingual BERT")

    if st.button("Search"):
        if not query.strip():
            st.warning("Please enter a search query.")
        else:
            results = search_meetings(query, meetings, embedder, index, top_k=top_k)

            st.markdown("## 📚 Results")
            if not results:
                st.info("No results found.")
            for result in results:
                st.markdown(
                    f"""
                    <div class="card">
                    <h4>{result['title']}</h4>
                    <b>Date:</b> {result['date']} <br><br>
                    {result['transcript'].replace("\n", "<br>")}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

# =========================================================
# TAB 3 - RAG Q&A
# =========================================================

with tab3:
    st.subheader("💬 Ask Questions Across Meetings")

    question = st.text_area(
        "Enter your question",
        placeholder="What deadlines were discussed?",
    )

    if st.button("Generate Answer"):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            answer = rag_answer(question, meetings, embedder, index, top_k=2)

            st.markdown(
                f"""
                <div class="answer-box">
                {answer.replace("\n", "<br>")}
                </div>
                """,
                unsafe_allow_html=True,
            )

# =========================================================
# TAB 4 - CLUSTERING
# =========================================================

with tab4:
    st.subheader("📈 Semantic Clustering")

    if len(meetings) < 2:
        st.info("Add at least two meetings to see clustering.")
    else:
        max_clusters = min(4, len(meetings))
        num_clusters = st.slider(
            "Number of Clusters",
            min_value=2,
            max_value=max_clusters,
            value=min(2, max_clusters),
        )

        kmeans = KMeans(
            n_clusters=num_clusters,
            random_state=42,
            n_init=10,
        )
        cluster_labels = kmeans.fit_predict(embeddings)

        cluster_df = pd.DataFrame(
            {
                "Meeting": [m["title"] for m in meetings],
                "Cluster": cluster_labels,
            }
        )

        st.dataframe(cluster_df, use_container_width=True)

        st.markdown("### 🔁 Cluster Insights")
        for cluster in sorted(cluster_df["Cluster"].unique()):
            st.markdown(f"## Cluster {cluster}")
            cluster_meetings = cluster_df[cluster_df["Cluster"] == cluster]
            for m in cluster_meetings["Meeting"]:
                st.success(m)

# =========================================================
# ADD NEW MEETING
# =========================================================

st.markdown("---")
st.subheader("➕ Add New Meeting")

with st.form("meeting_form"):
    title = st.text_input("Meeting Title")
    date = st.date_input("Meeting Date")
    transcript = st.text_area("Transcript")
    submitted = st.form_submit_button("Add Meeting")

    if submitted:
        if not title.strip() or not transcript.strip():
            st.warning("Please enter both a meeting title and transcript.")
        else:
            new_meeting = {
                "id": f"meeting_{len(meetings) + 1}",
                "date": str(date),
                "title": title.strip(),
                "transcript": transcript.strip(),
            }

            st.session_state.meetings.append(new_meeting)
            st.success("✅ Meeting Added Successfully!")
            st.rerun()

# =========================================================
# FOOTER
# =========================================================

st.markdown("---")
st.caption(
    "Built using Streamlit + Sentence Transformers + FAISS + Scikit-learn")


