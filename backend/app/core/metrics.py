import time
import logging

logger = logging.getLogger("alvinai")


class QueryMetrics:
    def __init__(self):
        self.start_time = time.time()

    def elapsed_ms(self) -> float:
        return (time.time() - self.start_time) * 1000

    def log(self, query: str, namespace: str, num_chunks: int = 0):
        logger.info(
            "query=%s namespace=%s chunks=%d latency_ms=%.1f",
            query[:80], namespace, num_chunks, self.elapsed_ms(),
        )
