NAMESPACE_KEYWORDS = {
    "compliance": ["compliance", "regulation", "safety standard", "emission", "legal", "recall notice"],
    "engineering": ["diagnostic", "torque", "engine", "transmission", "wiring", "schematic", "repair procedure"],
    "dealer_sales": ["pricing", "inventory", "financing", "msrp", "lease", "trade-in", "dealership"],
    "employee_hr": ["hr", "human resources", "leave", "benefits", "performance review", "pto", "payroll"],
    "vendor": ["supplier", "procurement", "vendor", "parts order", "purchase order"],
    "customer_support": ["warranty", "service", "appointment", "maintenance", "oil change"],
}


def classify_namespace(query: str) -> str:
    """Simple keyword-based namespace classifier."""
    query_lower = query.lower()
    scores = {}
    for ns, keywords in NAMESPACE_KEYWORDS.items():
        scores[ns] = sum(1 for kw in keywords if kw in query_lower)
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "customer_support"
