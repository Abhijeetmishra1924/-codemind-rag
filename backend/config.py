import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# ---------------------------------------------------------------------------
# Base project paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_REPOS_DIR = DATA_DIR / "raw_repos"
VECTOR_DB_DIR = DATA_DIR / "vector_db"
BM25_DB_DIR = DATA_DIR / "bm25_db"          # NEW: sparse index persistence
CACHE_DIR = DATA_DIR / "cache"              # NEW: query/response cache
MANIFEST_DIR = DATA_DIR / "manifests"       # NEW: per-repo file-hash manifests for incremental indexing

# Ensure data directories exist
for _dir in (RAW_REPOS_DIR, VECTOR_DB_DIR, BM25_DB_DIR, CACHE_DIR, MANIFEST_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# API Keys and Models configurations
# ---------------------------------------------------------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")  # optional: for hosted reranker

# Default Embedding Model Selection
# We default to Gemini if API key is present, otherwise OpenAI or local backup.
DEFAULT_EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "gemini").lower()
DEFAULT_LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower()

# ---------------------------------------------------------------------------
# Chunking configuration (code-aware)
# ---------------------------------------------------------------------------
# Extension -> tree-sitter language name, used by CodeSplitter.
# Anything not in this map falls back to a token-based splitter.
CODE_LANGUAGE_MAP = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".java": "java",
    ".go": "go",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".c": "cpp",
    ".h": "cpp",
    ".hpp": "cpp",
    ".rs": "rust",
    ".rb": "ruby",
}
NON_CODE_EXTS = {".md", ".json", ".html", ".css", ".yaml", ".yml", ".txt"}

CODE_CHUNK_LINES = int(os.getenv("CODE_CHUNK_LINES", 40))
CODE_CHUNK_LINES_OVERLAP = int(os.getenv("CODE_CHUNK_LINES_OVERLAP", 10))
CODE_MAX_CHARS = int(os.getenv("CODE_MAX_CHARS", 1500))

TEXT_CHUNK_SIZE = int(os.getenv("TEXT_CHUNK_SIZE", 512))
TEXT_CHUNK_OVERLAP = int(os.getenv("TEXT_CHUNK_OVERLAP", 64))

REQUIRED_EXTS = list(CODE_LANGUAGE_MAP.keys()) + list(NON_CODE_EXTS)

# ---------------------------------------------------------------------------
# Retrieval configuration
# ---------------------------------------------------------------------------
VECTOR_TOP_K = int(os.getenv("VECTOR_TOP_K", 10))       # candidates from dense search
BM25_TOP_K = int(os.getenv("BM25_TOP_K", 10))            # candidates from sparse search
FUSION_NUM_QUERIES = int(os.getenv("FUSION_NUM_QUERIES", 3))  # query rewriting fan-out
RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", 5))         # final chunks sent to LLM

RERANKER_PROVIDER = os.getenv("RERANKER_PROVIDER", "local").lower()  # "local" | "cohere" | "none"
LOCAL_RERANKER_MODEL = os.getenv("LOCAL_RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

ENABLE_HYDE = os.getenv("ENABLE_HYDE", "true").lower() == "true"
