"""
Document loaders — ingest text, PDF, web pages, Markdown, CSV, and directories.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Union

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """Core document unit throughout the RAG pipeline."""

    page_content: str
    metadata: Dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.page_content)


class TextFileLoader:
    """Load plain text (or Markdown) files."""

    def load(self, path: Union[str, Path]) -> List[Document]:
        path = Path(path)
        text = path.read_text(encoding="utf-8", errors="replace")
        return [
            Document(page_content=text, metadata={"source": str(path), "type": "text"})
        ]


class PDFLoader:
    """
    Load PDF files using PyMuPDF (fitz).
    Install: pip install pymupdf
    """

    def load(self, path: Union[str, Path]) -> List[Document]:
        try:
            import fitz  # noqa
        except ImportError:
            raise ImportError("PyMuPDF required: pip install pymupdf")

        path = Path(path)
        docs = []
        with fitz.open(str(path)) as pdf:
            for page_num, page in enumerate(pdf):
                text = page.get_text()
                if text.strip():
                    docs.append(
                        Document(
                            page_content=text,
                            metadata={
                                "source": str(path),
                                "page": page_num + 1,
                                "type": "pdf",
                            },
                        )
                    )
        logger.info("PDF '%s': loaded %d pages", path.name, len(docs))
        return docs


class WebPageLoader:
    """
    Load a web page via requests + BeautifulSoup.
    Install: pip install requests beautifulsoup4
    """

    def load(self, url: str) -> List[Document]:
        try:
            import requests
            from bs4 import BeautifulSoup
        except ImportError:
            raise ImportError(
                "requests + beautifulsoup4 required: pip install requests beautifulsoup4"
            )

        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        return [Document(page_content=text, metadata={"source": url, "type": "web"})]


class CSVLoader:
    """Load CSV rows as individual documents."""

    def __init__(self, content_columns: Optional[List[str]] = None) -> None:
        self.content_columns = content_columns

    def load(self, path: Union[str, Path]) -> List[Document]:
        import csv

        path = Path(path)
        docs = []
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row_num, row in enumerate(reader):
                if self.content_columns:
                    text = " | ".join(
                        str(row[c]) for c in self.content_columns if c in row
                    )
                else:
                    text = " | ".join(str(v) for v in row.values())
                docs.append(
                    Document(
                        page_content=text,
                        metadata={"source": str(path), "row": row_num, **row},
                    )
                )
        return docs


_LOADER_MAP = {
    ".txt": TextFileLoader,
    ".md": TextFileLoader,
    ".pdf": PDFLoader,
    ".csv": CSVLoader,
}


class DirectoryLoader:
    """
    Recursively load all supported files from a directory.
    """

    def __init__(self, glob: str = "**/*", recursive: bool = True) -> None:
        self.glob = glob
        self.recursive = recursive

    def load(self, directory: Union[str, Path]) -> List[Document]:
        directory = Path(directory)
        docs: List[Document] = []
        pattern = "**/*" if self.recursive else "*"
        for file_path in directory.glob(pattern):
            if not file_path.is_file():
                continue
            loader_cls = _LOADER_MAP.get(file_path.suffix.lower())
            if loader_cls:
                try:
                    docs.extend(loader_cls().load(file_path))
                except Exception as e:
                    logger.warning("Failed to load %s: %s", file_path, e)
        logger.info(
            "DirectoryLoader: loaded %d documents from %s", len(docs), directory
        )
        return docs
