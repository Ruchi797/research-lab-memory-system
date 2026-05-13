import streamlit as st
import pandas as pd
import numpy as np
import faiss
import re
from datetime import datetime
from sentence_transformers import SentenceTransformer
from transformers import pipeline
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

st.markdown("""
<style>
.main-title {
    font-size: 42px;
    font-weight: bold;
    color: #4F46E5;
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
    margin: 4px;
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
""", unsafe_allow_html=True)

# =========================================================
# HEADER
# =========================================================

st.markdown('<div class="main-title">🧠 Research Lab Meeting Memory System</div>', unsafe_allow_html=True)

st.write("AI-powered meeting intelligence system using NLP + Semantic Search + RAG")

# =========================================================
# SAMPLE DATA
# =========================================================

@st.cache_data
def load_data():

    meetings = [
        {
            "id": "meeting_001",
            "date": "2026-05-01",
            "title": "RAG Pipeline Discussion",
            "transcript": """
            Today we discussed FAISS vector databases and retrieval augmented generation.
            Deadline for conference submission is June 10.
            Research idea includes semantic search optimization.
            Paper discussed: Attention Is All You Need.
            """
        },

        {
            "id": "meeting_002",
            "date": "2026-05-04",
            "title": "Multilingual NLP Meeting",
            "transcript": """
            Team explored multilingual BERT and cross-lingual transfer learning.
            We will perform experiments on Hindi and Marathi datasets.
            Deadline for experiments is next Friday.
            """
        },

        {
            "id": "meeting_003",
            "date": "2026-05-08",
            "title": "Evaluation Metrics Meeting",
            "transcript": """
            Discussion about F1 score, precision, recall and BLEU metrics.
            We also discussed dashboard visualization for model evaluation.
            """
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
            """
        }
    ]

    return meetings

meetings = load_data()

# =========================================================
# LOAD MODELS
# =========================================================

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
# def load_summarizer():
 #  return pipeline(
  #      "summarization",
   #     model="facebook/bart-large-cnn"
    #)

embedder = load_embedding_model()
# summarizer = load_summarizer()

# =========================================================
# VECTOR DATABASE
# =========================================================

texts = [m["transcript"] for m in meetings]

embeddings = embedder.encode(texts)

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(np.array(embeddings).astype("float32"))

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("⚙️ Settings")

top_k = st.sidebar.slider(
    "Search Results",
    min_value=1,
    max_value=5,
    value=3
)

st.sidebar.markdown("---")

st.sidebar.success(f"✅ Meetings Loaded: {len(meetings)}")

st.sidebar.info(f"🔍 Vector Chunks Indexed: {index.ntotal}")

# =========================================================
# TABS
# =========================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "📋 Meeting Insights",
    "🔍 Semantic Search",
    "💬 RAG Q&A",
    "📈 Clustering & Trends"
])

# =========================================================
# TAB 1
# =========================================================

with tab1:

    st.subheader("📋 Meeting Insights")

    titles = [m["title"] for m in meetings]

    selected_title = st.selectbox(
        "Select Meeting",
        titles
    )

    meeting = next(
        m for m in meetings
        if m["title"] == selected_title
    )

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("### 📅 Date")
        st.info(meeting["date"])

        st.markdown("### 📝 Transcript")

        st.markdown(
            f"""
            <div class="card">
            {meeting["transcript"]}
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:

        st.markdown("### 📌 AI Summary")

        try:
            summary = summarizer(
                meeting["transcript"],
                max_length=60,
                min_length=20,
                do_sample=False
            )

            st.success(summary[0]["summary_text"])

        except:
            st.warning("Summary model loading...")

        st.markdown("### 🏷️ Keywords")

        vectorizer = TfidfVectorizer(
            stop_words="english"
        )

        X = vectorizer.fit_transform(
            [meeting["transcript"]]
        )

        keywords = zip(
            vectorizer.get_feature_names_out(),
            X.toarray()[0]
        )

        sorted_keywords = sorted(
            keywords,
            key=lambda x: x[1],
            reverse=True
        )[:10]

        for word, score in sorted_keywords:

            st.markdown(
                f'<span class="keyword">{word}</span>',
                unsafe_allow_html=True
            )

        st.markdown("### ⏰ Deadlines")

        deadline_pattern = r"(June \d+|next Friday|June \d+)"

        deadlines = re.findall(
            deadline_pattern,
            meeting["transcript"]
        )

        if deadlines:
            for d in deadlines:
                st.warning(f"📌 {d}")
        else:
            st.info("No deadlines found")

# =========================================================
# TAB 2
# =========================================================

with tab2:

    st.subheader("🔍 Semantic Search")

    query = st.text_input(
        "Search meetings",
        placeholder="e.g. multilingual BERT"
    )

    if st.button("Search"):

        query_embedding = embedder.encode([query])

        distances, indices = index.search(
            np.array(query_embedding).astype("float32"),
            top_k
        )

        st.markdown("## 📚 Results")

        for idx in indices[0]:

            result = meetings[idx]

            st.markdown(
                f"""
                <div class="card">
                <h4>{result['title']}</h4>
                <b>Date:</b> {result['date']} <br><br>
                {result['transcript']}
                </div>
                """,
                unsafe_allow_html=True
            )

# =========================================================
# TAB 3
# =========================================================

with tab3:

    st.subheader("💬 Ask Questions Across Meetings")

    question = st.text_area(
        "Enter your question",
        placeholder="What deadlines were discussed?"
    )

    if st.button("Generate Answer"):

        question_embedding = embedder.encode([question])

        distances, indices = index.search(
            np.array(question_embedding).astype("float32"),
            2
        )

        retrieved_docs = []

        for idx in indices[0]:
            retrieved_docs.append(
                meetings[idx]["transcript"]
            )

        context = "\n".join(retrieved_docs)

        answer = f"""
        Based on retrieved meetings:

        {context}
        """

        st.markdown(
            f"""
            <div class="answer-box">
            {answer}
            </div>
            """,
            unsafe_allow_html=True
        )

# =========================================================
# TAB 4
# =========================================================

with tab4:

    st.subheader("📈 Semantic Clustering")

    num_clusters = st.slider(
        "Number of Clusters",
        2,
        4,
        2
    )

    kmeans = KMeans(
        n_clusters=num_clusters,
        random_state=42
    )

    cluster_labels = kmeans.fit_predict(embeddings)

    cluster_df = pd.DataFrame({
        "Meeting": [m["title"] for m in meetings],
        "Cluster": cluster_labels
    })

    st.dataframe(cluster_df)

    st.markdown("### 🔁 Cluster Insights")

    for cluster in sorted(cluster_df["Cluster"].unique()):

        st.markdown(f"## Cluster {cluster}")

        cluster_meetings = cluster_df[
            cluster_df["Cluster"] == cluster
        ]

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

        new_meeting = {
            "id": f"meeting_{len(meetings)+1}",
            "date": str(date),
            "title": title,
            "transcript": transcript
        }

        meetings.append(new_meeting)

        st.success("✅ Meeting Added Successfully!")

# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.caption("Built using Streamlit + Sentence Transformers + FAISS + HuggingFace + Scikit-learn")
