import re

# --- Greeting / Small-talk Detection ---
_GREETING_PATTERNS = re.compile(
    r"^\s*(hi|hello|hey|good\s*(morning|afternoon|evening)|howdy|sup|yo|"
    r"what'?s?\s*up|how\s*are\s*you|thanks|thank\s*you|ok|okay|bye|goodbye|"
    r"see\s*you|cheers|hola|namaste)\s*[!?.]*\s*$",
    re.IGNORECASE,
)


def is_greeting(query: str) -> bool:
    """Return True if the query is a simple greeting or small-talk."""
    return bool(_GREETING_PATTERNS.match(query.strip()))


# --- Query Type Classification ---
# Routes queries to: factual_lookup | document_search | general

FACTUAL_PATTERNS = [
    r"what is the (torque|spec|specification|price|cost|part number|msrp)",
    r"how much does .+ cost",
    r"what('s| is) the .+ for .+\?",
    r"give me the .+ (spec|number|value|price)",
    r"(torque|pressure|voltage|amperage|capacity|weight) .+ (spec|value|rating)",
]

DOCUMENT_SEARCH_PATTERNS = [
    r"which (tsb|bulletin|document|manual|regulation|policy|procedure)",
    r"find .+ (about|related|regarding|concerning)",
    r"what does .+ (say|state|mention|require|specify) about",
    r"show me .+ (policy|regulation|document|manual|guide)",
    r"(fmvss|ece|regulation|compliance|standard) .+ (require|apply|cover)",
    r"according to",
    r"search for",
    r"look up",
]

FACTUAL_COMPILED = [re.compile(p, re.IGNORECASE) for p in FACTUAL_PATTERNS]
DOCUMENT_COMPILED = [re.compile(p, re.IGNORECASE) for p in DOCUMENT_SEARCH_PATTERNS]


def classify_query_type(query: str) -> str:
    """Classify query into: factual_lookup | document_search | general.

    - factual_lookup: specific value retrieval (torque specs, prices, part numbers)
    - document_search: full RAG pipeline (retrieve + rerank + generate)
    - general: LLM-only, no retrieval needed
    """
    for pattern in FACTUAL_COMPILED:
        if pattern.search(query):
            return "factual_lookup"

    for pattern in DOCUMENT_COMPILED:
        if pattern.search(query):
            return "document_search"

    # Default: if query has a question mark or looks like a question, use document_search
    # This ensures RAG is used for most real queries
    if "?" in query or len(query.split()) > 5:
        return "document_search"

    return "general"


# --- Document Reference Extraction ---
# Matches patterns like HR-POL-003, TSB-2024-001, FMVSS-301, etc.
_DOC_REF_PATTERN = re.compile(
    r"\b([A-Z]{2,}[-_][A-Z]{2,}[-_]\d{2,})\b", re.IGNORECASE
)


def extract_document_ref(query: str) -> str | None:
    """Extract a document ID reference from the query (e.g. 'HR-POL-003').

    Returns the uppercased document reference if found, else None.
    """
    match = _DOC_REF_PATTERN.search(query)
    if match:
        return match.group(1).upper()
    return None


# --- Namespace Classification ---

NAMESPACE_KEYWORDS = {
    "compliance": [
        "compliance", "regulation", "safety standard", "emission", "legal",
        "recall notice", "fmvss", "ece", "homologation", "audit",
    ],
    "engineering": [
        "diagnostic", "torque", "engine", "transmission", "wiring", "schematic",
        "repair procedure", "tsb", "technical bulletin", "spec",
    ],
    "dealer_sales": [
        "pricing", "inventory", "financing", "msrp", "lease", "trade-in",
        "dealership", "brochure", "feature comparison",
    ],
    "employee_hr": [
        "hr", "human resources", "leave", "benefits", "performance review",
        "pto", "payroll", "remote work", "code of conduct", "compensation",
    ],
    "vendor": [
        "supplier", "procurement", "vendor", "parts order", "purchase order",
    ],
    "customer_support": [
        "warranty", "service", "appointment", "maintenance", "oil change",
        "recall", "fault code",
    ],
}


def classify_namespace(query: str) -> str:
    """Simple keyword-based namespace classifier."""
    query_lower = query.lower()
    scores = {}
    for ns, keywords in NAMESPACE_KEYWORDS.items():
        scores[ns] = sum(1 for kw in keywords if kw in query_lower)
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "customer_support"
