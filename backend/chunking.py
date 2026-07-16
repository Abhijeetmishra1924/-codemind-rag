"""
Code-aware chunking.

Why this exists:
The original pipeline used a single TokenTextSplitter for every file type,
which cuts chunks mid-function and destroys code structure. This module
splits code files along AST boundaries (functions/classes) using
tree-sitter, and falls back to token-based splitting for file types that
aren't source code (README, JSON, HTML, etc.) or aren't supported by the
installed grammars.

Each resulting node is tagged with rich metadata (file_path, language,
line_range) so retrieval results can cite exact locations and later be
filtered by language or path.
"""

from pathlib import Path
from typing import List

from llama_index.core import Document
from llama_index.core.node_parser import CodeSplitter, TokenTextSplitter
from llama_index.core.schema import BaseNode

from backend.config import (
    CODE_LANGUAGE_MAP,
    CODE_CHUNK_LINES,
    CODE_CHUNK_LINES_OVERLAP,
    CODE_MAX_CHARS,
    TEXT_CHUNK_SIZE,
    TEXT_CHUNK_OVERLAP,
)

# Cache splitters per-language so we don't reconstruct tree-sitter parsers
# for every single file.
_code_splitter_cache = {}


def _get_code_splitter(language: str) -> CodeSplitter:
    if language not in _code_splitter_cache:
        _code_splitter_cache[language] = CodeSplitter(
            language=language,
            chunk_lines=CODE_CHUNK_LINES,
            chunk_lines_overlap=CODE_CHUNK_LINES_OVERLAP,
            max_chars=CODE_MAX_CHARS,
        )
    return _code_splitter_cache[language]


_text_splitter = TokenTextSplitter(
    chunk_size=TEXT_CHUNK_SIZE,
    chunk_overlap=TEXT_CHUNK_OVERLAP,
)


def _enrich_metadata(nodes: List[BaseNode], doc: Document, language: str) -> None:
    """Attach retrieval/citation-friendly metadata to every node in place."""
    file_path = doc.metadata.get("file_path", "unknown")
    file_name = Path(file_path).name
    for node in nodes:
        node.metadata["file_path"] = file_path
        node.metadata["file_name"] = file_name
        node.metadata["language"] = language
        # Keep metadata out of the embedding text itself (it would pollute
        # the semantic signal) but keep it visible to the LLM at synth time.
        node.excluded_embed_metadata_keys = list(
            set(node.excluded_embed_metadata_keys or []) | {"file_path", "file_name", "language"}
        )


def chunk_documents(documents: List[Document]) -> List[BaseNode]:
    """
    Splits a list of loaded documents into retrieval nodes, routing each
    document to an AST-aware splitter when its language is supported, and
    to a token-based splitter otherwise.
    """
    all_nodes: List[BaseNode] = []
    skipped_language_failures = 0

    for doc in documents:
        file_path = doc.metadata.get("file_path", "")
        ext = Path(file_path).suffix.lower()
        language = CODE_LANGUAGE_MAP.get(ext)

        if language:
            try:
                splitter = _get_code_splitter(language)
                nodes = splitter.get_nodes_from_documents([doc])
            except Exception as e:
                # tree-sitter grammar missing, syntax error in file, etc.
                # Never let one bad file kill the whole indexing run.
                print(f"  [chunking] AST split failed for {file_path} ({e}); falling back to token split.")
                skipped_language_failures += 1
                nodes = _text_splitter.get_nodes_from_documents([doc])
                language = f"{language}(fallback)"
        else:
            nodes = _text_splitter.get_nodes_from_documents([doc])
            language = ext.lstrip(".") or "text"

        _enrich_metadata(nodes, doc, language)
        all_nodes.extend(nodes)

    if skipped_language_failures:
        print(f"  [chunking] {skipped_language_failures} file(s) fell back to token-based splitting.")

    return all_nodes
