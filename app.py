""
Main System Orchestrator
=========================
Ties all modules together into a single ResearchLabMemorySystem class.
This is the single entry point for all operations.

Usage:
    system = ResearchLabMemorySystem()
    system.ingest_meetings(meetings_list)
    
    # Search
    results = system.search("RAG pipeline vector database")
    
    # Q&A
    answer = system.ask("What papers were discussed about transformers?")
    
    # Analysis
    report = system.analyze_trends()
    
    # Get insights for a meeting
    insights = system.get_meeting_insights("M001")
"""

import os
import json
import logging
from typing import List, Dict, Optional
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("ResearchLabMemory")

from modules.preprocessing import TextPreprocessor, ProcessedMeeting
from modules.extraction import InformationExtractor, MeetingInsights
from modules.embeddings import MeetingIndexer
from modules.clustering import CrossMeetingAnalyzer
from modules.rag import RAGPipeline, RAGResult


class ResearchLabMemorySystem:
    """
    End-to-end Research Lab Meeting Memory System.
    
    Architecture overview:
    ┌─────────────────────────────────────────────┐
    │           ResearchLabMemorySystem            │
    │                                             │
    │  Raw Transcripts                            │
    │       ↓                                     │
    │  TextPreprocessor  ──→  ProcessedMeeting    │
    │       ↓                                     │
    │  InformationExtractor → MeetingInsights     │
    │       ↓              (keywords, NER,        │
    │       ↓               papers, deadlines)    │
    │  MeetingIndexer    ──→  FAISS Index         │
    │       ↓                                     │
    │  RAGPipeline       ──→  Q&A Answers         │
    │  CrossMeetingAnalyzer → Trend Reports       │
    └─────────────────────────────────────────────┘
    """

    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2",
                 use_api: bool = None,
                 persist_dir: Optional[str] = None):
        """
        Args:
            embedding_model: SentenceTransformer model name
            use_api: Use Anthropic API for generation (None=auto-detect from env)
            persist_dir: Directory to save/load index (None=in-memory only)
        """
        self.persist_dir = persist_dir
        self.embedding_model = embedding_model

        # Initialize all modules
        self.preprocessor = TextPreprocessor()
        self.extractor = InformationExtractor()
        self.indexer = MeetingIndexer(embedding_model=embedding_model)
        self.analyzer = CrossMeetingAnalyzer()

        # State
        self.processed_meetings: List[ProcessedMeeting] = []
        self.insights: Dict[str, MeetingInsights] = {}  # meeting_id → insights
        self.is_indexed = False

        # RAG pipeline (initialized after indexing)
        self._rag: Optional[RAGPipeline] = None
        self._use_api = use_api

        logger.info("ResearchLabMemorySystem initialized")

    def ingest_meetings(self, meetings: List[Dict],
                        chunk_level: str = "all") -> Dict:
        """
        Full ingestion pipeline: raw meeting dicts → indexed, searchable system.
        
        Args:
            meetings: List of meeting dicts with keys:
                      meeting_id, date, title, attendees, transcript
            chunk_level: "sentence" | "turn" | "meeting" | "all"
        Returns:
            Summary dict with ingestion stats
        """
        logger.info(f"Ingesting {len(meetings)} meetings...")
        start = datetime.now()

        # Step 1: Preprocess
        logger.info("Step 1/4: Preprocessing transcripts...")
        self.processed_meetings = self.preprocessor.process_batch(meetings)

        # Step 2: Extract insights
        logger.info("Step 2/4: Extracting insights (keywords, NER, papers, deadlines)...")
        corpus_texts = [pm.clean_text for pm in self.processed_meetings]
        self.extractor.fit_keywords(corpus_texts)

        for pm in self.processed_meetings:
            insights = self.extractor.extract_all(pm)
            self.insights[pm.meeting_id] = insights

        # Step 3: Build FAISS index
        logger.info("Step 3/4: Building FAISS index...")
        self.indexer.index_meetings(self.processed_meetings, chunk_level=chunk_level)
        self.is_indexed = True

        # Step 4: Initialize RAG
        logger.info("Step 4/4: Initializing RAG pipeline...")
        self._rag = RAGPipeline(self.indexer, use_api=self._use_api)

        # Persist if directory specified
        if self.persist_dir:
            self.save(self.persist_dir)

        elapsed = (datetime.now() - start).total_seconds()
        summary = {
            "meetings_ingested": len(self.processed_meetings),
            "total_chunks_indexed": self.indexer.vector_store.total_vectors,
            "total_papers_found": sum(i.total_papers for i in self.insights.values()),
            "total_deadlines_found": sum(i.total_deadlines for i in self.insights.values()),
            "generation_mode": self._rag.generation_mode,
            "elapsed_seconds": round(elapsed, 2)
        }
        logger.info(f"Ingestion complete: {summary}")
        return summary

    def search(self, query: str, top_k: int = 5,
               filter_meeting_id: Optional[str] = None) -> List[Dict]:
        """
        Semantic search over all indexed meeting content.
        
        Args:
            query: Natural language search query
            top_k: Number of results
            filter_meeting_id: Optional scope to specific meeting
        Returns:
            Ranked list of matching chunks
        """
        self._check_indexed()
        results = self.indexer.semantic_search(
            query=query, top_k=top_k, filter_meeting_id=filter_meeting_id
        )
        logger.info(f"Search '{query[:50]}': {len(results)} results")
        return results

    def ask(self, question: str, top_k: int = 5,
            filter_meeting_id: Optional[str] = None) -> RAGResult:
        """
        Answer a question using RAG over meeting content.
        
        Args:
            question: Natural language question
            top_k: Number of chunks to retrieve for context
            filter_meeting_id: Scope to specific meeting (e.g., for "what was decided in M003?")
        Returns:
            RAGResult with answer, sources, and retrieved context
        """
        self._check_indexed()
        return self._rag.query(question, top_k=top_k, filter_meeting_id=filter_meeting_id)

    def get_meeting_insights(self, meeting_id: str) -> Optional[MeetingInsights]:
        """Get all extracted insights for a specific meeting."""
        return self.insights.get(meeting_id)

    def get_all_insights(self) -> List[MeetingInsights]:
        """Get insights for all meetings."""
        return list(self.insights.values())

    def get_all_papers(self) -> List[Dict]:
        """Aggregate all papers cited across all meetings."""
        all_papers = []
        for mid, insights in self.insights.items():
            for paper in insights.papers_cited:
                paper_with_meeting = {**paper, "meeting_id": mid, "date": insights.date}
                all_papers.append(paper_with_meeting)
        return all_papers

    def get_all_deadlines(self) -> List[Dict]:
        """Aggregate all deadlines from all meetings."""
        all_deadlines = []
        for mid, insights in self.insights.items():
            for dl in insights.deadlines:
                dl_with_meeting = {**dl, "meeting_id": mid, "meeting_title": insights.title}
                all_deadlines.append(dl_with_meeting)
        # Sort by meeting date
        return sorted(all_deadlines, key=lambda x: x.get('date', ''))

    def get_all_action_items(self) -> List[Dict]:
        """Aggregate all action items from preprocessed meetings."""
        all_items = []
        for pm in self.processed_meetings:
            for item in pm.action_items:
                all_items.append({
                    "meeting_id": pm.meeting_id,
                    "date": pm.date,
                    "action": item
                })
        return all_items

    def analyze_trends(self) -> Dict:
        """
        Cross-meeting trend analysis.
        Returns repeated themes, trending keywords, timeline.
        """
        self._check_indexed()

        store = self.indexer.vector_store
        if store.total_vectors == 0:
            return {"error": "No vectors in index"}

        # Get embeddings from FAISS
        try:
            import faiss
            embeddings = faiss.rev_swig_ptr(
                store.index.get_xb(),
                store.total_vectors * store.embedding_dim
            ).reshape(store.total_vectors, store.embedding_dim).copy()
        except Exception as e:
            logger.warning(f"Could not extract FAISS embeddings for clustering: {e}")
            # Fallback: return basic trend analysis without clustering
            return self._basic_trend_analysis()

        report = self.analyzer.analyze(
            self.processed_meetings,
            list(self.insights.values()),
            embeddings,
            store.metadata_store
        )

        return {
            "repeated_themes": report.repeated_themes[:10],
            "trending_keywords": report.trending_keywords[:10],
            "unique_topics_per_meeting": report.unique_topics_per_meeting,
            "timeline": report.timeline,
            "total_clusters": len(self.analyzer.clusterer.clusters)
        }

    def _basic_trend_analysis(self) -> Dict:
        """Fallback trend analysis without clustering."""
        from collections import Counter

        all_keywords = []
        for insights in self.insights.values():
            all_keywords.extend([kw for kw, _ in insights.keywords[:10]])

        keyword_freq = Counter(all_keywords)
        repeated = [{"keyword": kw, "count": count}
                   for kw, count in keyword_freq.most_common(10) if count >= 2]

        return {
            "repeated_themes": repeated,
            "trending_keywords": [],
            "timeline": [{"meeting_id": mid, "date": i.date, "title": i.title,
                         "top_keywords": [kw for kw, _ in i.keywords[:5]]}
                        for mid, i in self.insights.items()]
        }

    def generate_report(self) -> str:
        """
        Generate a human-readable markdown report of all meeting insights.
        """
        lines = ["# Research Lab Meeting Memory Report\n"]
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        lines.append(f"Total Meetings: {len(self.processed_meetings)}\n")

        # Papers section
        all_papers = self.get_all_papers()
        if all_papers:
            lines.append("\n## Papers Discussed\n")
            for p in all_papers:
                lines.append(f"- \"{p['title']}\" by {p['authors']} "
                           f"[Meeting {p['meeting_id']}, {p['date']}]\n")

        # Deadlines section
        deadlines = self.get_all_deadlines()
        if deadlines:
            lines.append("\n## Deadlines & Action Items\n")
            for d in deadlines:
                lines.append(f"- **{d.get('assignee', 'Unknown')}**: {d.get('task', '')} "
                           f"— Due: {d.get('date', '')} (Meeting {d.get('meeting_id', '')})\n")

        # Per-meeting summaries
        lines.append("\n## Meeting Summaries\n")
        for pm in self.processed_meetings:
            insights = self.insights.get(pm.meeting_id)
            lines.append(f"\n### {pm.meeting_id}: {pm.title} ({pm.date})\n")
            lines.append(f"**Attendees:** {', '.join(pm.attendees)}\n\n")
            if insights:
                kws = [kw for kw, _ in insights.keywords[:8]]
                lines.append(f"**Keywords:** {', '.join(kws)}\n\n")
                if insights.summary_sentences:
                    lines.append("**Summary:**\n")
                    for sent in insights.summary_sentences[:2]:
                        lines.append(f"> {sent}\n\n")

        return ''.join(lines)

    def save(self, directory: str) -> None:
        """Save FAISS index to disk for persistence."""
        self.indexer.save(directory)
        logger.info(f"System saved to {directory}")

    def load(self, directory: str) -> None:
        """Load previously saved FAISS index."""
        self.indexer.load_index(directory)
        self.is_indexed = True
        self._rag = RAGPipeline(self.indexer, use_api=self._use_api)
        logger.info(f"System loaded from {directory}")

    def _check_indexed(self):
        if not self.is_indexed:
            raise RuntimeError("System not indexed. Call ingest_meetings() first.")


# ─────────────────────────────────────────────────────────────────────
# DEMO RUN
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, '/home/claude/research_lab_memory')
    from data.sample_meetings import SAMPLE_MEETINGS

    print("=" * 70)
    print("  RESEARCH LAB MEETING MEMORY SYSTEM — FULL DEMO")
    print("=" * 70)

    # Initialize system
    system = ResearchLabMemorySystem(
        embedding_model="all-MiniLM-L6-v2",
        use_api=False  # Set to None to auto-detect ANTHROPIC_API_KEY
    )

    # Ingest all sample meetings
    print("\n[1] INGESTING MEETINGS...")
    summary = system.ingest_meetings(SAMPLE_MEETINGS, chunk_level="sentence")
    print(f"    ✓ {summary['meetings_ingested']} meetings | "
          f"{summary['total_chunks_indexed']} chunks | "
          f"{summary['total_papers_found']} papers | "
          f"{summary['total_deadlines_found']} deadlines | "
          f"{summary['elapsed_seconds']}s")

    # Show insights per meeting
    print("\n[2] MEETING INSIGHTS SNAPSHOT...")
    for mid, insights in system.insights.items():
        kws = [kw for kw, _ in insights.keywords[:5]]
        print(f"    {mid}: {insights.title}")
        print(f"         Keywords: {kws}")
        print(f"         Papers: {insights.total_papers} | Deadlines: {insights.total_deadlines}")

    # All papers
    print("\n[3] ALL PAPERS CITED ACROSS MEETINGS...")
    for p in system.get_all_papers():
        print(f"    [{p['meeting_id']}] \"{p['title']}\" — {p['authors']}")

    # All deadlines
    print("\n[4] ALL DEADLINES...")
    for d in system.get_all_deadlines():
        print(f"    [{d['meeting_id']}] {d.get('assignee', '?')}: "
              f"{d.get('task', '')[:50]} | Due: {d.get('date', '')}")

    # Semantic search
    print("\n[5] SEMANTIC SEARCH DEMO...")
    queries = [
        "RAG retrieval augmented generation",
        "multimodal fusion audio text sentiment",
        "conference submission deadline"
    ]
    for q in queries:
        results = system.search(q, top_k=2)
        print(f"\n    Query: '{q}'")
        for r in results:
            print(f"    → [{r['meeting_id']} | {r['score']:.3f}] {r['text'][:90]}...")

    # RAG Q&A
    print("\n[6] RAG Q&A DEMO...")
    questions = [
        "What papers were discussed about RAG and retrieval?",
        "What are all the deadlines mentioned in the meetings?",
        "Who is working on the cross-modal attention model?",
        "What is the target conference and when is the submission deadline?",
        "What are the main research directions of the lab?",
    ]
    for q in questions:
        result = system.ask(q, top_k=4)
        print(f"\n    Q: {q}")
        print(f"    A: {result.answer[:250]}...")
        print(f"       [Sources: {result.sources} | Confidence: {result.confidence:.2f}]")

    # Trend report
    print("\n[7] CROSS-MEETING TREND ANALYSIS...")
    trends = system.analyze_trends()
    print(f"    Repeated themes (appear in 2+ meetings):")
    for theme in trends.get('repeated_themes', [])[:5]:
        if isinstance(theme, dict):
            print(f"    • {theme.get('keyword') or theme.get('theme')}: "
                  f"{theme.get('count') or theme.get('frequency')}")

    print("\n[8] TIMELINE...")
    for entry in trends.get('timeline', []):
        print(f"    [{entry.get('date', '')}] {entry.get('meeting_id')}: "
              f"Keywords: {entry.get('top_keywords') or entry.get('new_topics', [])[:3]}")

    # Generate report
    print("\n[9] GENERATING MARKDOWN REPORT...")
    report = system.generate_report()
    report_path = "/home/claude/research_lab_memory/meeting_report.md"
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"    Report saved to: {report_path}")

    print("\n" + "=" * 70)
    print("  DEMO COMPLETE")
    print("=" * 70)
