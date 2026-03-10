import logging
from pathlib import Path

logger = logging.getLogger("alvinai")


def parse_document(file_path: str) -> dict:
    """Parse a document file and return its text content + metadata.

    Supports: PDF, DOCX, TXT, HTML.
    Returns: {"text": str, "title": str, "doc_type": str, "pages": int}
    """
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return _parse_pdf(path)
    elif suffix == ".docx":
        return _parse_docx(path)
    elif suffix in (".txt", ".md"):
        return _parse_text(path)
    elif suffix in (".html", ".htm"):
        return _parse_html(path)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")


def _parse_pdf(path: Path) -> dict:
    import pdfplumber

    text_parts = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
        page_count = len(pdf.pages)

    return {
        "text": "\n\n".join(text_parts),
        "title": path.stem,
        "doc_type": "pdf",
        "pages": page_count,
    }


def _parse_docx(path: Path) -> dict:
    import docx

    doc = docx.Document(str(path))
    text_parts = [p.text for p in doc.paragraphs if p.text.strip()]

    return {
        "text": "\n\n".join(text_parts),
        "title": path.stem,
        "doc_type": "docx",
        "pages": 0,
    }


def _parse_text(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    return {
        "text": text,
        "title": path.stem,
        "doc_type": "txt",
        "pages": 0,
    }


def _parse_html(path: Path) -> dict:
    from bs4 import BeautifulSoup

    html = path.read_text(encoding="utf-8")
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    text = soup.get_text(separator="\n", strip=True)
    title = soup.title.string if soup.title else path.stem

    return {
        "text": text,
        "title": title,
        "doc_type": "html",
        "pages": 0,
    }
