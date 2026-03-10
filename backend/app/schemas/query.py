from pydantic import BaseModel


class QueryRequest(BaseModel):
    query: str
    namespace: str = "customer_support"
    conversation_id: str | None = None


class QueryResponse(BaseModel):
    answer: str
    namespace: str
    conversation_id: str
    sources: list[dict] = []
    latency_ms: float = 0.0
