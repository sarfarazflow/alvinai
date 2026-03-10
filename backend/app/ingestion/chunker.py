import logging
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger("alvinai")

# Namespace-specific chunking strategies (from architecture doc Section 3)
CHUNK_CONFIGS = {
    "customer_support": {"chunk_size": 1500, "chunk_overlap": 200},
    "engineering": {"chunk_size": 800, "chunk_overlap": 150},
    "dealer_sales": {"chunk_size": 512, "chunk_overlap": 50},
    "compliance": {"chunk_size": 600, "chunk_overlap": 100},
    "employee_hr": {"chunk_size": 800, "chunk_overlap": 150},
    "vendor": {"chunk_size": 800, "chunk_overlap": 150},
}


def chunk_text(text: str, namespace: str) -> list[dict]:
    """Split text into chunks using namespace-specific strategy.

    Returns list of {"content": str, "chunk_index": int, "token_count": int}.
    """
    config = CHUNK_CONFIGS.get(namespace, CHUNK_CONFIGS["customer_support"])

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config["chunk_size"],
        chunk_overlap=config["chunk_overlap"],
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_text(text)
    logger.info("Chunked text into %d chunks (namespace=%s)", len(chunks), namespace)

    return [
        {
            "content": chunk,
            "chunk_index": i,
            "token_count": len(chunk.split()),
        }
        for i, chunk in enumerate(chunks)
    ]
