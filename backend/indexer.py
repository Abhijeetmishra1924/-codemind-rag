import argparse
import hashlib
import json
import sys
from pathlib import Path

import git
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
    Settings,
)
from llama_index.retrievers.bm25 import BM25Retriever

# Add parent directory to path to ensure backend imports work
sys.path.append(str(Path(__file__).resolve().parent.parent))

from backend.config import (
    RAW_REPOS_DIR,
    VECTOR_DB_DIR,
    BM25_DB_DIR,
    MANIFEST_DIR,
    GEMINI_API_KEY,
    OPENAI_API_KEY,
    REQUIRED_EXTS,
    BM25_TOP_K,
)
from backend.chunking import chunk_documents

# https://github.com/Abhijeetmishra1924/academic_summrizer
# ---------------------------------------------------------------------------
# LLM / Embedding configurationSettings.embed_model = GeminiEmbedding(api_key=gemini_key, model_name="models/gemini-embedding-2")
# Settings.llm = Gemini(api_key=gemini_key, model_name="models/gemini-2.5-flash-lite")
# ---------------------------------------------------------------------------
def configure_llama_index():
    """Configures LlamaIndex's global Settings for embeddings + LLM."""
    gemini_key = GEMINI_API_KEY if GEMINI_API_KEY and "your_gemini_api" not in GEMINI_API_KEY.lower() else None
    openai_key = OPENAI_API_KEY if OPENAI_API_KEY and "your_openai_api" not in OPENAI_API_KEY.lower() else None

    if gemini_key:
        from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
        from llama_index.llms.google_genai import GoogleGenAI
        print("Configuring LlamaIndex with Gemini API...")
        Settings.embed_model = GoogleGenAIEmbedding(api_key=gemini_key, model_name="models/gemini-embedding-2")
        Settings.llm = GoogleGenAI(api_key=gemini_key, model_name="models/gemini-2.5-flash-lite")
    elif openai_key:
        from llama_index.embeddings.openai import OpenAIEmbedding
        from llama_index.llms.openai import OpenAI
        print("Configuring LlamaIndex with OpenAI API...")
        Settings.embed_model = OpenAIEmbedding(api_key=openai_key)
        Settings.llm = OpenAI(api_key=openai_key)
    else:
        print("Warning: No valid API Keys found. Falling back to HuggingFace Embeddings and Mock LLM.")
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        from llama_index.core.llms import MockLLM
        Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
        Settings.llm = MockLLM()


# ---------------------------------------------------------------------------
# Repository cloning
# ---------------------------------------------------------------------------
def clone_repository(repo_url: str) -> Path:
    """Clones a GitHub repository to the raw_repos directory."""
    repo_name = repo_url.rstrip("/").split("/")[-1]
    if repo_name.endswith(".git"):
        repo_name = repo_name[:-4]

    target_path = RAW_REPOS_DIR / repo_name

    if target_path.exists() and (target_path / ".git").exists():
        print(f"Repository already exists at {target_path}. Attempting to pull latest changes...")
        try:
            repo = git.Repo(target_path)
            repo.remotes.origin.pull()
            print("Successfully pulled latest changes.")
        except Exception as e:
            print(f"Could not pull latest changes: {e}. Reusing existing codebase.")
    else:
        print(f"Cloning {repo_url} to {target_path}...")
        git.Repo.clone_from(repo_url, target_path)
        print("Cloning completed successfully.")

    return target_path


# ---------------------------------------------------------------------------
# Incremental indexing support
# ---------------------------------------------------------------------------
def _manifest_path(repo_name: str) -> Path:
    return MANIFEST_DIR / f"{repo_name}.json"


def _load_manifest(repo_name: str) -> dict:
    path = _manifest_path(repo_name)
    if path.exists():
        return json.loads(path.read_text())
    return {}


def _save_manifest(repo_name: str, manifest: dict) -> None:
    _manifest_path(repo_name).write_text(json.dumps(manifest, indent=2))


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _diff_repository(repo_path: Path, repo_name: str):
    """
    Compares current file hashes against the stored manifest.
    Returns (changed_or_new_files, deleted_file_paths, new_manifest).
    """
    old_manifest = _load_manifest(repo_name)
    new_manifest = {}
    changed_files = []

    for ext_group in REQUIRED_EXTS:
        pass  # placeholder to keep intent explicit; actual walk below

    for file_path in repo_path.rglob("*"):
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() not in REQUIRED_EXTS:
            continue
        if any(part in {".git", "node_modules", ".venv", "__pycache__"} for part in file_path.parts):
            continue

        rel_path = str(file_path.relative_to(repo_path))
        try:
            h = _file_hash(file_path)
        except Exception:
            continue
        new_manifest[rel_path] = h

        if old_manifest.get(rel_path) != h:
            changed_files.append(file_path)

    deleted_files = [rel for rel in old_manifest if rel not in new_manifest]

    return changed_files, deleted_files, new_manifest


# ---------------------------------------------------------------------------
# BM25 persistence helpers
# ---------------------------------------------------------------------------
def _bm25_persist_dir(repo_name: str) -> Path:
    d = BM25_DB_DIR / repo_name
    d.mkdir(parents=True, exist_ok=True)
    return d


def _build_and_persist_bm25(nodes, repo_name: str):
    """Builds a fresh BM25 index from the full node set and persists it."""
    if not nodes:
        return
    bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=BM25_TOP_K)
    bm25_retriever.persist(str(_bm25_persist_dir(repo_name)))


# ---------------------------------------------------------------------------
# Core indexing
# ---------------------------------------------------------------------------
def index_repository(repo_path: Path) -> Path:
    """
    Parses changed/new files, generates AST-aware chunks, updates the
    vector index incrementally (removing stale nodes for changed/deleted
    files first, to avoid duplicates), and rebuilds the BM25 sparse index
    from the resulting full node set.
    """
    repo_name = repo_path.name
    persist_dir = VECTOR_DB_DIR / repo_name

    print(f"Starting ingestion for repository: {repo_name}...")
    configure_llama_index()

    changed_files, deleted_files, new_manifest = _diff_repository(repo_path, repo_name)

    index_exists = persist_dir.exists() and (persist_dir / "docstore.json").exists()

    if index_exists:
        storage_context = StorageContext.from_defaults(persist_dir=str(persist_dir))
        index = load_index_from_storage(storage_context)
    else:
        storage_context = StorageContext.from_defaults()
        index = None

    if not changed_files and not deleted_files and index_exists:
        print("No file changes detected since last index. Skipping re-embedding.")
        _rebuild_bm25_from_docstore(index, repo_name)
        return persist_dir

    print(f"{len(changed_files)} new/changed file(s), {len(deleted_files)} deleted file(s) detected.")

    # Remove stale nodes for changed or deleted files so re-indexing never
    # produces duplicates (this was a bug in the original implementation,
    # which blindly called insert_nodes on every run).
    if index is not None:
        stale_paths = {str(f.relative_to(repo_path)) for f in changed_files} | set(deleted_files)
        if stale_paths:
            _remove_nodes_for_paths(index, stale_paths)

    # Read only the changed/new files instead of the whole repo.
    if changed_files:
        reader = SimpleDirectoryReader(input_files=[str(f) for f in changed_files])
        documents = reader.load_data()
        print(f"Loaded {len(documents)} document(s) for changed files.")

        print("Chunking documents (AST-aware for code, token-based for text)...")
        nodes = chunk_documents(documents)
        print(f"Generated {len(nodes)} node(s).")
    else:
        nodes = []

    if index is None:
        print("Building new vector index...")
        index = VectorStoreIndex(nodes, storage_context=storage_context)
    elif nodes:
        index.insert_nodes(nodes)

    print(f"Persisting vector index to {persist_dir}...")
    index.storage_context.persist(persist_dir=str(persist_dir))

    _save_manifest(repo_name, new_manifest)

    # Rebuild BM25 from the full, now-consistent docstore.
    _rebuild_bm25_from_docstore(index, repo_name)

    print("Repo successfully indexed.")
    return persist_dir


def _remove_nodes_for_paths(index: VectorStoreIndex, rel_paths: set) -> None:
    """Deletes all nodes whose file_path metadata matches the given set."""
    docstore = index.storage_context.docstore
    to_delete = []
    for node_id, node in docstore.docs.items():
        node_rel = node.metadata.get("file_path", "")
        # file_path is stored as an absolute-ish path from the reader; match on suffix
        if any(node_rel.endswith(rp) for rp in rel_paths):
            to_delete.append(node_id)
    for node_id in to_delete:
        try:
            index.delete_nodes([node_id])
        except Exception:
            docstore.delete_document(node_id, raise_error=False)
    if to_delete:
        print(f"  Removed {len(to_delete)} stale node(s) from vector index.")


def _rebuild_bm25_from_docstore(index: VectorStoreIndex, repo_name: str) -> None:
    all_nodes = list(index.storage_context.docstore.docs.values())
    print(f"Rebuilding BM25 sparse index from {len(all_nodes)} node(s)...")
    _build_and_persist_bm25(all_nodes, repo_name)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CodeMind Repository Indexer")
    parser.add_argument("--url", type=str, help="GitHub Repository URL to clone and index")
    parser.add_argument("--path", type=str, help="Path to a local repository directory to index directly")

    args = parser.parse_args()

    if not args.url and not args.path:
        print("Error: You must provide either --url or --path.")
        sys.exit(1)

    try:
        if args.url:
            repo_path = clone_repository(args.url)
        else:
            repo_path = Path(args.path)
            if not repo_path.exists():
                print(f"Error: Local path {repo_path} does not exist.")
                sys.exit(1)

        persist_path = index_repository(repo_path)
        print(f"Success: Vector database ready at {persist_path}")
    except Exception as e:
        print(f"An error occurred during indexing: {e}")
        sys.exit(1)
