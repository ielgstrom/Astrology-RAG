"""Source loading: turn files, directories, and URLs into LangChain Documents.

Every loader returns a list of `Document`, with `metadata["source"]` set to
something short enough to cite in an answer.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable, Sequence

from langchain_core.documents import Document

# Directories that are never worth reading when walking a tree.
SKIP_DIRS = {
    ".git", ".hg", ".svn", "node_modules", ".venv", "venv", "env",
    "__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    "dist", "build", ".next", ".idea", ".vscode", ".DS_Store",
}

# Files above this size are skipped when walking a directory. A file named
# explicitly on the command line is always read, however large.
MAX_WALK_FILE_BYTES = 2_000_000


class UnsupportedSourceError(ValueError):
    """Raised when a source exists but cannot be read as text."""


def load_sources(refs: Sequence[str], *, quiet: bool = False) -> list[Document]:
    """Load every reference into a flat list of Documents."""
    docs: list[Document] = []
    for ref in refs:
        docs.extend(load_source(ref, quiet=quiet))
    return docs


def load_source(ref: str, *, quiet: bool = False) -> list[Document]:
    """Load one reference: a URL, a directory, or a single file."""
    if ref.startswith(("http://", "https://")):
        return _load_url(ref)

    path = Path(ref).expanduser()
    
    return _load_file(path)


def _load_file(path: Path) -> list[Document]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _load_pdf(path)
    return UnsupportedSourceError(f"{path} is not a supported source type; only .pdf is supported")


def _load_pdf(path: Path) -> list[Document]:
    try:
        from langchain_community.document_loaders import PyPDFLoader
    except ImportError as exc:  # pragma: no cover - depends on install
        raise UnsupportedSourceError(
            "Reading PDFs needs langchain-community. Run: pip install langchain-community pypdf"
        ) from exc

    try:
        pages = PyPDFLoader(str(path)).load()
    except ImportError as exc:
        raise UnsupportedSourceError("Reading PDFs needs pypdf. Run: pip install pypdf") from exc

    # One Document per file keeps citations simple; page numbers stay inline.
    body = "\n\n".join(
        f"[page {page.metadata.get('page', i) + 1}]\n{page.page_content}"
        for i, page in enumerate(pages)
    )
    return [Document(page_content=body, metadata={"source": str(path)})]


def _load_url(url: str) -> list[Document]:
    try:
        from langchain_community.document_loaders import WebBaseLoader
    except ImportError as exc:  # pragma: no cover - depends on install
        raise UnsupportedSourceError(
            "Reading URLs needs langchain-community. Run: pip install langchain-community beautifulsoup4"
        ) from exc

    docs = WebBaseLoader(url).load()
    for doc in docs:
        doc.metadata["source"] = url
        # WebBaseLoader keeps the page's whitespace; collapse the worst of it.
        doc.page_content = "\n".join(
            line.strip() for line in doc.page_content.splitlines() if line.strip()
        )
    return docs


def describe(docs: Iterable[Document]) -> str:
    """One-line summary of what was loaded, for CLI feedback."""
    docs = list(docs)
    chars = sum(len(d.page_content) for d in docs)
    return f"{len(docs)} document(s), ~{chars // 4:,} tokens"
