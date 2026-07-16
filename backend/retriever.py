import sys
from pathlib import Path

from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.chat_engine import CondensePlusContextChatEngine
from llama_index.core.indices.query.query_transform import HyDEQueryTransform
from llama_index.core.query_engine import TransformQueryEngine
from llama_index.retrievers.bm25 import BM25Retriever

# Add parent directory to path to ensure backend imports work
sys.path.append(str(Path(__file__).resolve().parent.parent))

from backend.config import (
    VECTOR_DB_DIR,
    BM25_DB_DIR,
    VECTOR_TOP_K,
    BM25_TOP_K,
    FUSION_NUM_QUERIES,
    RERANK_TOP_N,
    RERANKER_PROVIDER,
    LOCAL_RERANKER_MODEL,
    COHERE_API_KEY,
    ENABLE_HYDE,
)
from backend.indexer import configure_llama_index


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------
def get_indexed_repositories() -> list[str]:
    """Returns a list of all indexed repository names (vector index present)."""
    if not VECTOR_DB_DIR.exists():
        return []
    repos = []
    for path in VECTOR_DB_DIR.iterdir():
        if path.is_dir() and (path / "docstore.json").exists():
            repos.append(path.name)
    return repos


# ---------------------------------------------------------------------------
# Index loading
# ---------------------------------------------------------------------------
def load_repository_index(repo_name: str):
    """Loads the persisted vector index for a given repository name."""
    persist_dir = VECTOR_DB_DIR / repo_name
    if not persist_dir.exists() or not (persist_dir / "docstore.json").exists():
        raise FileNotFoundError(f"No index found for repository '{repo_name}' at {persist_dir}")

    configure_llama_index()
    storage_context = StorageContext.from_defaults(persist_dir=str(persist_dir))
    index = load_index_from_storage(storage_context)
    return index


def _load_bm25_retriever(repo_name: str, index):
    """
    Loads a persisted BM25 retriever if available; otherwise builds one
    on-the-fly from the vector index's docstore (e.g. for repos indexed
    before BM25 support existed).
    """
    bm25_dir = BM25_DB_DIR / repo_name
    if bm25_dir.exists() and any(bm25_dir.iterdir()):
        try:
            return BM25Retriever.from_persist_dir(str(bm25_dir))
        except Exception as e:
            print(f"  [retriever] Failed to load persisted BM25 index ({e}); rebuilding in-memory.")

    all_nodes = list(index.storage_context.docstore.docs.values())
    return BM25Retriever.from_defaults(nodes=all_nodes, similarity_top_k=BM25_TOP_K)


# ---------------------------------------------------------------------------
# Reranker
# ---------------------------------------------------------------------------
def _get_reranker():
    """
    Returns a node postprocessor that reranks fused candidates with a
    cross-encoder before they're sent to the LLM. This is the single
    highest-leverage addition for retrieval precision: raw vector/BM25
    similarity is noisy, a cross-encoder scores query+chunk pairs jointly.
    """
    if RERANKER_PROVIDER == "none":
        return None

    if RERANKER_PROVIDER == "cohere" and COHERE_API_KEY:
        from llama_index.postprocessor.cohere_rerank import CohereRerank
        return CohereRerank(api_key=COHERE_API_KEY, top_n=RERANK_TOP_N, model="rerank-english-v3.0")

    # Default: local, free, self-hosted cross-encoder. No external API cost.
    return SentenceTransformerRerank(model=LOCAL_RERANKER_MODEL, top_n=RERANK_TOP_N)


# ---------------------------------------------------------------------------
# Hybrid fusion retriever
# ---------------------------------------------------------------------------
def get_hybrid_retriever(repo_name: str):
    """
    Builds a retriever that fuses dense vector search with BM25 sparse
    search using Reciprocal Rank Fusion, and fans the original query out
    into several LLM-rewritten variants first (improves recall for vague
    or under-specified questions).
    """
    index = load_repository_index(repo_name)

    vector_retriever = index.as_retriever(similarity_top_k=VECTOR_TOP_K)
    bm25_retriever = _load_bm25_retriever(repo_name, index)

    fusion_retriever = QueryFusionRetriever(
        [vector_retriever, bm25_retriever],
        similarity_top_k=max(VECTOR_TOP_K, BM25_TOP_K),
        num_queries=FUSION_NUM_QUERIES,  # 1 = no rewriting, >1 = query expansion
        mode="reciprocal_rerank",
        use_async=False,
    )
    return index, fusion_retriever


# ---------------------------------------------------------------------------
# Query engine (single-shot, non-chat) — useful for CLI / eval
# ---------------------------------------------------------------------------
def get_query_engine(repo_name: str):
    """Returns a single-shot query engine with hybrid retrieval + reranking,
    optionally wrapped in a HyDE transform for better code-search recall."""
    index, fusion_retriever = get_hybrid_retriever(repo_name)
    reranker = _get_reranker()
    postprocessors = [reranker] if reranker else []

    query_engine = index.as_query_engine(
        retriever=fusion_retriever,
        node_postprocessors=postprocessors,
    )

    if ENABLE_HYDE:
        hyde = HyDEQueryTransform(include_original=True)
        query_engine = TransformQueryEngine(query_engine, hyde)

    return query_engine


# ---------------------------------------------------------------------------
# Chat engine (multi-turn) — used by the Streamlit app
# ---------------------------------------------------------------------------
def get_chat_engine(repo_name: str):
    """
    Returns a chat engine for the given repository with:
      - Hybrid (dense + BM25) retrieval fused via reciprocal rank
      - Cross-encoder reranking of fused candidates
      - Condense+context mode: prior turns are condensed into a standalone
        query before retrieval, so follow-up questions like "what about
        the retry logic in there?" resolve correctly instead of being
        embedded in isolation.
    """
    index, fusion_retriever = get_hybrid_retriever(repo_name)
    reranker = _get_reranker()
    postprocessors = [reranker] if reranker else []

    system_prompt = (
        "You are CodeMind, an advanced AI codebase assistant. Your goal is to help the user "
        "understand the codebase by answering questions using the provided file chunks.\n"
        "Guidelines:\n"
        "1. Rely strictly on the provided context nodes to answer the questions.\n"
        "2. When explaining code, provide clean syntax-highlighted code snippets.\n"
        "3. Always mention the file path(s) of the code you are referring to.\n"
        "4. If the answer cannot be found in the provided context, state that you cannot find "
        "it, but suggest where it might be in general terms if possible.\n"
        "5. If a question spans multiple files, explicitly connect how they relate."
    )

    chat_engine = CondensePlusContextChatEngine.from_defaults(
        retriever=fusion_retriever,
        node_postprocessors=postprocessors,
        system_prompt=system_prompt,
    )
    return chat_engine


# ---------------------------------------------------------------------------
# CLI (for quick retrieval-quality checks without the UI)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CodeMind Repository Retriever CLI")
    parser.add_argument("--repo", type=str, required=True, help="Name of the indexed repository")
    parser.add_argument("--query", type=str, required=True, help="Question to ask about the codebase")

    args = parser.parse_args()

    try:
        print(f"Loading index for {args.repo}...")
        query_engine = get_query_engine(args.repo)
        print(f"Querying: '{args.query}'...")
        response = query_engine.query(args.query)

        print("\n=== Response ===")
        print(response)
        print("\n=== Sources ===")
        for idx, source in enumerate(response.source_nodes):
            print(f"[{idx+1}] File: {source.node.metadata.get('file_path')} (Score: {source.score:.3f})")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
