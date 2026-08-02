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
    if path.is_dir():
        return _load_directory(path, quiet=quiet)
    if path.is_file():
        return _load_file(path)

    raise FileNotFoundError(f"No such file, directory, or URL: {ref}")


def _load_file(path: Path) -> list[Document]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _load_pdf(path)
    if suffix == ".docx":
        return _load_docx(path)
    return [Document(page_content=_read_text(path), metadata={"source": str(path)})]


def _read_text(path: Path) -> str:
    """Read a file as UTF-8, rejecting anything that is clearly binary."""
    raw = path.read_bytes()
    if b"\x00" in raw[:8192]:
        raise UnsupportedSourceError(f"{path} looks like a binary file")
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise UnsupportedSourceError(f"{path} is not valid UTF-8 text") from exc


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


def _load_docx(path: Path) -> list[Document]:
    try:
        from langchain_community.document_loaders import Docx2txtLoader
    except ImportError as exc:  # pragma: no cover - depends on install
        raise UnsupportedSourceError(
            "Reading .docx needs langchain-community. Run: pip install langchain-community docx2txt"
        ) from exc

    try:
        docs = Docx2txtLoader(str(path)).load()
    except ImportError as exc:
        raise UnsupportedSourceError("Reading .docx needs docx2txt. Run: pip install docx2txt") from exc

    for doc in docs:
        doc.metadata["source"] = str(path)
    return docs


def _load_directory(root: Path, *, quiet: bool = False) -> list[Document]:
    docs: list[Document] = []
    skipped: list[str] = []

    for path in sorted(root.rglob("*")):
        if not path.is_file() or _is_hidden_or_skipped(path, root):
            continue
        if path.stat().st_size > MAX_WALK_FILE_BYTES:
            skipped.append(f"{path} (too large)")
            continue
        try:
            docs.extend(_load_file(path))
        except (UnsupportedSourceError, OSError) as exc:
            skipped.append(f"{path} ({exc})")

    if skipped and not quiet:
        print(f"Skipped {len(skipped)} file(s) under {root}:", file=sys.stderr)
        for line in skipped[:10]:
            print(f"  - {line}", file=sys.stderr)
        if len(skipped) > 10:
            print(f"  ... and {len(skipped) - 10} more", file=sys.stderr)

    return docs


def _is_hidden_or_skipped(path: Path, root: Path) -> bool:
    parts = path.relative_to(root).parts
    return any(part in SKIP_DIRS or part.startswith(".") for part in parts)


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
