"""
Shared utilities for v2 dataset generation scripts.

Corpus loading, section-based chunking, template-based Q&A generation,
and constants used by prepare_sft_dataset.py, prepare_raft_dataset.py,
and prepare_dpo_dataset.py.

No external API calls required — all generation is template-based.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Allow imports from backend/
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "backend"))

from app.ai.prompt import SYSTEM_PROMPTS  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NAMESPACES = [
    "customer_support",
    "engineering",
    "dealer_sales",
    "compliance",
    "employee_hr",
    "vendor",
]

NAMESPACE_PERSONAS: dict[str, str] = {
    "customer_support": "customer service representative",
    "engineering": "automotive engineer",
    "dealer_sales": "dealer or internal sales team member",
    "compliance": "compliance officer or legal team member",
    "employee_hr": "company employee",
    "vendor": "procurement or supply chain specialist",
}

RAFT_SYSTEM_PROMPT = (
    "You are an AI Assistant for an automotive engineering company. "
    "You assist employees, business partners, vendors, customers, and internal "
    "engineering teams. When answering, you must base your response ONLY on the "
    "documents provided in the context. If the answer is not present in the "
    "provided documents, you must say so clearly — do not fabricate information. "
    "When you use information from a document, cite it using "
    "##begin_quote## ... ##end_quote## markers."
)

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class Section:
    """A semantically meaningful chunk of a corpus document."""

    section_id: str
    heading: str
    content: str
    char_count: int = 0

    def __post_init__(self) -> None:
        self.char_count = len(self.content)


@dataclass
class CorpusDoc:
    """A single document from the knowledge corpus."""

    doc_id: str
    title: str
    namespace: str
    category: str
    filepath: str
    raw_text: str
    sections: list[Section] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Corpus loading
# ---------------------------------------------------------------------------

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def _parse_frontmatter(text: str) -> tuple[dict, str]:
    """Extract YAML frontmatter and return (metadata_dict, body)."""
    m = _FRONTMATTER_RE.match(text)
    if not m:
        return {}, text
    try:
        meta = yaml.safe_load(m.group(1)) or {}
    except yaml.YAMLError:
        meta = {}
    body = text[m.end() :]
    return meta, body


def _namespace_from_path(filepath: Path, corpus_dir: Path) -> str:
    """Derive namespace from directory structure: corpus_auto/{namespace}/..."""
    rel = filepath.relative_to(corpus_dir)
    return rel.parts[0] if rel.parts else "unknown"


def _category_from_path(filepath: Path, corpus_dir: Path) -> str:
    """Derive category from subdirectory: corpus_auto/{namespace}/{category}/..."""
    rel = filepath.relative_to(corpus_dir)
    return rel.parts[1] if len(rel.parts) > 2 else "general"


def load_corpus(corpus_dir: str | Path) -> list[CorpusDoc]:
    """Load all .md files from the corpus directory into CorpusDoc objects."""
    corpus_dir = Path(corpus_dir)
    docs: list[CorpusDoc] = []

    for md_path in sorted(corpus_dir.rglob("*.md")):
        if md_path.name == "README.md":
            continue

        text = md_path.read_text(encoding="utf-8")
        meta, body = _parse_frontmatter(text)

        namespace = meta.get("namespace") or _namespace_from_path(md_path, corpus_dir)
        category = meta.get("category") or _category_from_path(md_path, corpus_dir)
        doc_id = meta.get("document_id") or meta.get("tsb_number") or md_path.stem
        title = meta.get("title") or md_path.stem.replace("_", " ")

        doc = CorpusDoc(
            doc_id=doc_id,
            title=title,
            namespace=namespace,
            category=category,
            filepath=str(md_path),
            raw_text=body,
        )
        doc.sections = chunk_document(doc)
        docs.append(doc)

    return docs


# ---------------------------------------------------------------------------
# Section-based chunking
# ---------------------------------------------------------------------------

_H2_RE = re.compile(r"^## ", re.MULTILINE)
_H3_RE = re.compile(r"^### ", re.MULTILINE)

MAX_SECTION_CHARS = 2000


def chunk_document(doc: CorpusDoc) -> list[Section]:
    """Split a document body into sections using markdown headers."""
    body = doc.raw_text.strip()
    if not body:
        return []

    h2_parts = _split_on_pattern(body, _H2_RE)
    sections: list[Section] = []

    for idx, (heading, content) in enumerate(h2_parts):
        if len(content) <= MAX_SECTION_CHARS:
            sections.append(
                Section(
                    section_id=f"{doc.doc_id}-S{len(sections) + 1}",
                    heading=heading,
                    content=content.strip(),
                )
            )
        else:
            h3_parts = _split_on_pattern(content, _H3_RE)
            for sub_heading, sub_content in h3_parts:
                full_heading = f"{heading} > {sub_heading}" if sub_heading else heading
                if len(sub_content) <= MAX_SECTION_CHARS:
                    sections.append(
                        Section(
                            section_id=f"{doc.doc_id}-S{len(sections) + 1}",
                            heading=full_heading,
                            content=sub_content.strip(),
                        )
                    )
                else:
                    for para in _split_paragraphs(sub_content, MAX_SECTION_CHARS):
                        sections.append(
                            Section(
                                section_id=f"{doc.doc_id}-S{len(sections) + 1}",
                                heading=full_heading,
                                content=para.strip(),
                            )
                        )

    sections = [s for s in sections if s.char_count >= 50]
    return sections


def _split_on_pattern(text: str, pattern: re.Pattern) -> list[tuple[str, str]]:
    """Split text on a regex pattern, returning (heading, content) pairs."""
    positions = [m.start() for m in pattern.finditer(text)]
    if not positions:
        return [("", text)]

    parts: list[tuple[str, str]] = []
    if positions[0] > 0:
        preamble = text[: positions[0]].strip()
        if preamble:
            parts.append(("", preamble))

    for i, pos in enumerate(positions):
        end = positions[i + 1] if i + 1 < len(positions) else len(text)
        chunk = text[pos:end]
        first_nl = chunk.find("\n")
        if first_nl == -1:
            heading = chunk.lstrip("#").strip()
            content = ""
        else:
            heading = chunk[:first_nl].lstrip("#").strip()
            content = chunk[first_nl + 1 :]
        parts.append((heading, content))

    return parts


def _split_paragraphs(text: str, max_chars: int) -> list[str]:
    """Split text at double newlines, merging to stay under max_chars."""
    paragraphs = re.split(r"\n\n+", text)
    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        if current and len(current) + len(para) + 2 > max_chars:
            chunks.append(current)
            current = para
        else:
            current = f"{current}\n\n{para}" if current else para
    if current:
        chunks.append(current)
    return chunks


# ---------------------------------------------------------------------------
# Text extraction helpers (for template-based generation)
# ---------------------------------------------------------------------------

# Matches markdown table rows: | col1 | col2 | col3 |
_TABLE_ROW_RE = re.compile(r"^\|(.+)\|$", re.MULTILINE)
# Matches key: value or key — value patterns
_KV_RE = re.compile(r"^\*?\*?([A-Za-z][A-Za-z0-9 _/()-]{2,40})\*?\*?\s*[:—–]\s*(.+)$", re.MULTILINE)
# Matches numbered or bulleted list items
_LIST_ITEM_RE = re.compile(r"^(?:\d+\.|[-*])\s+(.+)$", re.MULTILINE)
# Matches part numbers like BG-PAD-CER-SUV-F, SPEC-BRK-100, etc.
_PART_NUM_RE = re.compile(r"\b[A-Z]{2,5}[-][A-Z0-9][-A-Z0-9]{3,20}\b")
# Matches numeric values with units
_VALUE_RE = re.compile(r"\b(\d+(?:\.\d+)?)\s*(mm|kg|kN|N·?m|Nm|V|A|°C|C|dB|Hz|kHz|GHz|MPa|bar|psi|ms|µs|hours?|days?|weeks?|years?|miles?|km|%)\b")
# Matches regulation/standard references
_REG_REF_RE = re.compile(r"\b(?:FMVSS|UN R|ISO|SAE|ECE|UNECE|CFR|EPA)\s*[\d/.]+(?:\s*§\s*[\d.]+)?")
# Matches DTC codes like C1A00, U0100, P0500
_DTC_RE = re.compile(r"\b[BCPU][0-9A-F]{4}\b")


def extract_key_values(text: str) -> list[tuple[str, str]]:
    """Extract key-value pairs from section content."""
    return [(m.group(1).strip(), m.group(2).strip()) for m in _KV_RE.finditer(text)]


def extract_table_rows(text: str) -> list[list[str]]:
    """Extract table rows as lists of cell values."""
    rows = []
    for m in _TABLE_ROW_RE.finditer(text):
        cells = [c.strip() for c in m.group(1).split("|")]
        # Skip separator rows (all dashes)
        if all(re.match(r"^[-:]+$", c) for c in cells if c):
            continue
        if any(c for c in cells):
            rows.append(cells)
    return rows


def extract_list_items(text: str) -> list[str]:
    """Extract bulleted/numbered list items."""
    return [m.group(1).strip() for m in _LIST_ITEM_RE.finditer(text)]


def extract_values_with_units(text: str) -> list[tuple[str, str]]:
    """Extract numeric values with their units."""
    return [(m.group(1), m.group(2)) for m in _VALUE_RE.finditer(text)]


def extract_part_numbers(text: str) -> list[str]:
    """Extract part numbers from text."""
    return _PART_NUM_RE.findall(text)


def extract_dtc_codes(text: str) -> list[str]:
    """Extract DTC codes from text."""
    return _DTC_RE.findall(text)


def extract_reg_references(text: str) -> list[str]:
    """Extract regulation/standard references."""
    return _REG_REF_RE.findall(text)


def get_first_sentence(text: str) -> str:
    """Get the first meaningful sentence from text."""
    # Skip markdown headers and empty lines
    for line in text.split("\n"):
        line = line.strip()
        if line and not line.startswith("#") and not line.startswith("|") and not line.startswith("---"):
            # Find first sentence end
            for end_char in [".", "!", "?"]:
                pos = line.find(end_char)
                if pos > 20:
                    return line[: pos + 1]
            if len(line) > 20:
                return line[:200]
    return text[:200]


def truncate(text: str, max_chars: int) -> str:
    """Truncate text to max_chars at a sentence boundary."""
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    last_period = truncated.rfind(".")
    if last_period > max_chars * 0.5:
        return truncated[: last_period + 1]
    return truncated + "..."


# ---------------------------------------------------------------------------
# Question templates by namespace
# ---------------------------------------------------------------------------

QUESTION_TEMPLATES: dict[str, list[str]] = {
    "customer_support": [
        "What does the {topic} section say about {detail}?",
        "Can you explain what {detail} means for my vehicle?",
        "I have a warning light for {detail} — what should I do?",
        "What is the procedure for {detail}?",
        "Is {detail} covered under warranty?",
        "How do I resolve {detail}?",
        "What does {detail} indicate on my dashboard?",
        "My vehicle is showing {detail}. Is it safe to continue driving?",
        "What are the steps for {detail}?",
        "Could you help me understand {detail} from the service manual?",
        "When should I bring my vehicle in for {detail}?",
        "What tools or parts are needed for {detail}?",
        "A customer is asking about {detail}. What should I tell them?",
        "What is the warranty coverage period for {detail}?",
        "How urgent is it when a customer reports {detail}?",
        "Is it safe for a customer to keep driving with {detail}?",
        "What is the recommended action for {detail}?",
        "Can a customer fix {detail} themselves or do they need a dealer?",
        "What causes {detail} in most cases?",
        "How long does it typically take to repair {detail}?",
        "Is there a recall related to {detail}?",
        "What should a technician check first for {detail}?",
        "Are there any known issues with {detail}?",
        "What is the cost estimate for repairing {detail}?",
        "How do I explain {detail} to a non-technical customer?",
        "What is the severity level of {detail}?",
        "Does {detail} affect vehicle safety?",
        "What replacement parts are needed for {detail}?",
        "How often does {detail} occur across our fleet?",
        "What should I document when a customer reports {detail}?",
    ],
    "engineering": [
        "What is the specification for {detail}?",
        "What does TSB {detail} recommend as the corrective action?",
        "What is the DTC code for {detail}?",
        "What are the test parameters for {detail}?",
        "What is the root cause analysis for {detail}?",
        "What tolerance is specified for {detail}?",
        "Which part number applies to {detail}?",
        "What is the affected population for {detail}?",
        "What are the validation requirements for {detail}?",
        "Explain the technical details of {detail}.",
        "What diagnostic procedure should be followed for {detail}?",
        "What are the environmental conditions for testing {detail}?",
        "What is the failure mode for {detail}?",
        "What are the material requirements for {detail}?",
        "What is the operating temperature range for {detail}?",
        "What ECU firmware version is required for {detail}?",
        "What CAN bus signals are relevant to {detail}?",
        "What is the measurement method for {detail}?",
        "What is the service life expectancy of {detail}?",
        "What are the torque specifications for {detail}?",
        "How does {detail} interact with other vehicle systems?",
        "What is the calibration procedure for {detail}?",
        "What FMEA risk level is assigned to {detail}?",
        "What supplier provides {detail}?",
        "What is the design intent behind {detail}?",
        "What are the durability test requirements for {detail}?",
        "Is {detail} affected by any active TSBs?",
        "What connector type is used for {detail}?",
        "What is the sampling rate for {detail} sensor data?",
        "What are the acceptance criteria for {detail}?",
    ],
    "dealer_sales": [
        "What are the key features of {detail}?",
        "How does {detail} compare to competitors?",
        "What is the pricing for {detail}?",
        "What warranty coverage does {detail} come with?",
        "What are the selling points for {detail}?",
        "Can you give me the product specifications for {detail}?",
        "What is the lead time for ordering {detail}?",
        "What makes {detail} stand out from alternatives?",
        "What customer benefits does {detail} provide?",
        "What is the recommended retail price for {detail}?",
        "How should I position {detail} against competing products?",
        "What performance data can I share with customers about {detail}?",
        "What is the margin structure for {detail}?",
        "What is the minimum order quantity for {detail}?",
        "What are the available variants of {detail}?",
        "What test results support the claims about {detail}?",
        "What is the installation time for {detail}?",
        "Does {detail} require any special tools for fitting?",
        "What vehicle applications is {detail} compatible with?",
        "What is the expected service life of {detail}?",
        "What certifications does {detail} hold?",
        "How does {detail} perform in extreme temperatures?",
        "What is the return rate on {detail}?",
        "What training materials are available for {detail}?",
        "What promotional materials exist for {detail}?",
        "What customer feedback have we received about {detail}?",
        "What is the competitive advantage of {detail} on price?",
        "What bulk discount is available for {detail}?",
        "When is the next production run for {detail}?",
        "What is the shelf life or storage requirement for {detail}?",
    ],
    "compliance": [
        "What does {detail} require for compliance?",
        "What are the testing requirements under {detail}?",
        "What is the scope of {detail}?",
        "Which vehicles are subject to {detail}?",
        "What are the performance thresholds specified in {detail}?",
        "What documentation is needed to demonstrate compliance with {detail}?",
        "When did {detail} come into effect?",
        "What are the key definitions in {detail}?",
        "What are the penalties for non-compliance with {detail}?",
        "How does {detail} apply to electric vehicles?",
        "What are the type approval requirements under {detail}?",
        "What test procedures does {detail} specify?",
        "What is the transition period for {detail}?",
        "Which contracting parties have ratified {detail}?",
        "What exemptions exist under {detail}?",
        "How does {detail} differ from the previous version?",
        "What markings or labels are required by {detail}?",
        "What is the conformity of production requirement under {detail}?",
        "Does {detail} apply to aftermarket components?",
        "What are the reporting requirements under {detail}?",
        "What sample size is required for testing under {detail}?",
        "What is the recall trigger threshold in {detail}?",
        "How does {detail} interact with other regulations?",
        "What instrumentation is specified for testing under {detail}?",
        "What ambient conditions are required for {detail} tests?",
        "What is the pass/fail criterion in {detail}?",
        "Does {detail} require third-party certification?",
        "What is the homologation process under {detail}?",
        "What records must be retained for {detail} compliance?",
        "How frequently must compliance with {detail} be re-verified?",
    ],
    "employee_hr": [
        "What is the company policy on {detail}?",
        "How many days of {detail} am I entitled to?",
        "What is the procedure for requesting {detail}?",
        "Who do I contact about {detail}?",
        "What are the eligibility requirements for {detail}?",
        "Can you explain the {detail} policy?",
        "What happens if I {detail}?",
        "Is {detail} available to part-time employees?",
        "What documentation do I need for {detail}?",
        "What is the approval process for {detail}?",
        "How is {detail} calculated?",
        "What are the deadlines for {detail}?",
        "Does {detail} apply to contractors as well?",
        "What is the notice period for {detail}?",
        "Can I combine {detail} with other leave types?",
        "What happens to {detail} if I change roles internally?",
        "Is there a cap on {detail}?",
        "What is the carry-over policy for {detail}?",
        "Who approves requests for {detail}?",
        "What is the appeals process if {detail} is denied?",
        "Are there regional differences in {detail}?",
        "How does {detail} work during probation?",
        "What is the company's stance on {detail}?",
        "How does {detail} affect my benefits?",
        "What training is required for {detail}?",
        "When was the {detail} policy last updated?",
        "Can {detail} be taken retroactively?",
        "What confidentiality applies to {detail}?",
        "How does {detail} interact with statutory requirements?",
        "What support resources are available for {detail}?",
    ],
    "vendor": [
        "What are the delivery terms for {detail}?",
        "What quality requirements apply to {detail}?",
        "What is the payment schedule for {detail}?",
        "What are the SLA metrics for {detail}?",
        "What are the qualification criteria for {detail}?",
        "What inspection procedures apply to {detail}?",
        "What are the contract terms for {detail}?",
        "What happens if {detail} is not met?",
        "What documentation must suppliers provide for {detail}?",
        "What are the acceptance criteria for {detail}?",
        "How is supplier performance measured for {detail}?",
        "What are the escalation procedures for {detail}?",
        "What is the penalty clause for {detail}?",
        "What certifications are required for {detail}?",
        "What is the lead time requirement for {detail}?",
        "How is pricing structured for {detail}?",
        "What is the defect rate threshold for {detail}?",
        "What change notification is required for {detail}?",
        "What are the packaging requirements for {detail}?",
        "What traceability is required for {detail}?",
        "What insurance coverage is needed for {detail}?",
        "What audit rights apply to {detail}?",
        "What is the dispute resolution process for {detail}?",
        "What confidentiality obligations apply to {detail}?",
        "What are the force majeure provisions for {detail}?",
        "How is inventory managed for {detail}?",
        "What are the safety stock requirements for {detail}?",
        "What is the warranty period the supplier provides for {detail}?",
        "What capacity commitment is required for {detail}?",
        "What sustainability requirements apply to {detail}?",
    ],
}

# Conversational prefixes to vary question style
QUESTION_PREFIXES = [
    "",
    "",
    "",
    "Quick question: ",
    "I need to know: ",
    "Can you help me with this? ",
    "Hi, ",
    "Could you clarify: ",
    "I have a question — ",
    "Technical question: ",
    "Our company needs to know: ",
    "I'm looking for information on ",
    "Please advise: ",
    "Urgent: ",
    "For a customer inquiry: ",
    "Following up on ",
    "Regarding ",
    "Can you look up ",
    "What do we know about ",
    "I need clarification on ",
]

# Conversational suffixes
QUESTION_SUFFIXES = [
    "",
    "",
    "",
    "",
    "",
    " Thanks in advance.",
    " Any guidance would be helpful.",
    " Please let me know how to proceed.",
    " Thanks.",
    " Could you clarify this for me?",
    " Please advise.",
    " This is time-sensitive.",
    " I need this for a customer.",
    " Can you check?",
]


# ---------------------------------------------------------------------------
# Answer building helpers
# ---------------------------------------------------------------------------

def build_answer_from_content(
    content: str,
    doc_title: str,
    section_id: str,
    namespace: str,
) -> str:
    """Build a concise, grounded answer from section content.

    Extracts key facts and composes a 2-5 sentence answer that
    references the source document.
    """
    # Extract structured data
    kvs = extract_key_values(content)
    tables = extract_table_rows(content)
    items = extract_list_items(content)
    values = extract_values_with_units(content)
    first_sent = get_first_sentence(content)

    parts = []

    # Lead with the first sentence as context
    if first_sent:
        parts.append(first_sent)

    # Add key-value details (up to 3)
    if kvs:
        for k, v in kvs[:3]:
            parts.append(f"{k}: {v}.")

    # Add table highlights (first data row)
    if tables and len(tables) > 1:
        header = tables[0]
        data = tables[1]
        pairs = [f"{h}: {d}" for h, d in zip(header, data) if h.strip() and d.strip()]
        if pairs:
            parts.append("Key data — " + ", ".join(pairs[:3]) + ".")

    # Add list items (up to 3)
    if items and not kvs and not tables:
        parts.append("Key points: " + "; ".join(items[:3]) + ".")

    # Add spec values
    if values and not parts[1:]:
        val_strs = [f"{v} {u}" for v, u in values[:3]]
        parts.append("Relevant values: " + ", ".join(val_strs) + ".")

    # Add source reference based on namespace tone
    if namespace == "compliance":
        parts.append(f"Refer to {section_id} for the full regulatory text.")
    elif namespace == "engineering":
        parts.append(f"See {doc_title}, section {section_id} for complete details.")
    elif namespace == "customer_support":
        parts.append(f"For further assistance, contact your authorised dealer or refer to {section_id}.")
    elif namespace == "dealer_sales":
        parts.append(f"Full product details are available in {doc_title}.")
    elif namespace == "vendor":
        parts.append(f"Refer to {section_id} in the contract documentation.")
    else:
        parts.append(f"See {section_id} for the full policy details.")

    # Compose — limit to 2-5 sentences
    answer = " ".join(parts[:5])
    return answer


def build_raft_answer(
    content: str,
    doc_title: str,
    section_id: str,
) -> str:
    """Build an answer with ##begin_quote## citation markers for RAFT."""
    first_sent = get_first_sentence(content)

    # Find a quotable passage (first substantial sentence or key-value)
    quote = ""
    for line in content.split("\n"):
        line = line.strip()
        if len(line) > 30 and not line.startswith("#") and not line.startswith("|") and not line.startswith("---") and not line.startswith("*"):
            quote = line[:200]
            break

    if not quote:
        quote = first_sent

    parts = []
    if first_sent and first_sent != quote:
        parts.append(first_sent)

    parts.append(f"##begin_quote## {quote} ##end_quote##")

    # Add supporting detail
    kvs = extract_key_values(content)
    if kvs:
        parts.append(f"Specifically, {kvs[0][0]}: {kvs[0][1]}.")

    parts.append(f"See {section_id} for complete details.")

    return " ".join(parts[:4])


# ---------------------------------------------------------------------------
# RAFT abstention templates
# ---------------------------------------------------------------------------

ABSTENTION_TEMPLATES = [
    "The information requested is not available in the provided documents. Please consult the relevant department or primary source for this information.",
    "Based on the documents provided, I cannot find information to answer this question. I recommend checking with the appropriate team or referring to the original documentation.",
    "The provided documents do not contain the information needed to answer this question. Please contact the relevant department for assistance.",
    "I cannot find an answer to this question in the provided context documents. Please refer to the primary source or contact the appropriate specialist.",
    "This specific information is not covered in the documents provided. I recommend consulting the relevant subject matter expert or referring to the primary documentation.",
    "After reviewing the provided context, I cannot locate the information needed. Please check the original source documentation or contact the appropriate team.",
]


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def corpus_summary(docs: list[CorpusDoc]) -> str:
    """Print a summary table of the loaded corpus."""
    from collections import Counter

    ns_counts: Counter[str] = Counter()
    ns_sections: Counter[str] = Counter()

    for doc in docs:
        ns_counts[doc.namespace] += 1
        ns_sections[doc.namespace] += len(doc.sections)

    lines = [
        "Namespace            Docs  Sections",
        "─" * 40,
    ]
    total_docs = 0
    total_sections = 0
    for ns in NAMESPACES:
        d = ns_counts.get(ns, 0)
        s = ns_sections.get(ns, 0)
        total_docs += d
        total_sections += s
        lines.append(f"{ns:<20} {d:>4}  {s:>8}")

    lines.append("─" * 40)
    lines.append(f"{'Total':<20} {total_docs:>4}  {total_sections:>8}")
    return "\n".join(lines)


def write_jsonl(records: list[dict], path: str | Path) -> None:
    """Write records to a JSONL file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"  Wrote {len(records)} records to {path}")
