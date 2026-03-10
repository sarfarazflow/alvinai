import asyncio
import httpx
import logging
from app.core.config import get_settings

logger = logging.getLogger("alvinai")
settings = get_settings()


def _format_mistral_prompt(messages: list[dict]) -> str:
    """Format messages into Mistral instruct template."""
    prompt = "<s>"
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "user":
            prompt += f"[INST] {content} [/INST]"
        elif role == "assistant":
            prompt += f" {content}</s>"
        elif role == "system":
            prompt += f"[INST] {content}\n\n"
    return prompt


def _extract_response(data: dict) -> str:
    """Extract text from RunPod/vLLM response formats."""
    output = data.get("output", data)
    if isinstance(output, list):
        output = output[0] if output else {}

    # OpenAI chat format
    choices = output.get("choices", [])
    if choices:
        choice = choices[0]
        msg = choice.get("message", {})
        if msg and msg.get("content"):
            return msg["content"].strip()
        if choice.get("text"):
            return choice["text"].strip()
        tokens = choice.get("tokens", [])
        if tokens:
            return "".join(tokens).strip()

    if isinstance(output, str):
        return output.strip()

    return str(output)


async def generate(
    query: str,
    system_prompt: str = "",
    context: str = "",
    max_tokens: int = 512,
    temperature: float = 0.3,
) -> str:
    """Send query to LLM (RunPod serverless or direct vLLM)."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    user_content = query
    if context:
        user_content = f"Context:\n{context}\n\nQuestion: {query}"
    messages.append({"role": "user", "content": user_content})

    try:
        if settings.is_runpod_serverless:
            return await _call_runpod(messages, max_tokens, temperature)
        else:
            return await _call_vllm_openai(messages, max_tokens, temperature)
    except httpx.ReadTimeout:
        logger.warning("LLM request timed out, retrying with /run endpoint...")
        if settings.is_runpod_serverless:
            return await _call_runpod_async(messages, max_tokens, temperature)
        raise
    except Exception as e:
        logger.error("LLM generation failed: %s", e)
        return f"I'm sorry, I'm having trouble processing your request right now. The inference service may be warming up — please try again in a moment. (Error: {type(e).__name__})"


async def _call_runpod(
    messages: list[dict], max_tokens: int, temperature: float
) -> str:
    """Call RunPod serverless — try OpenAI endpoint first, fall back to /run + poll."""
    headers = {"Authorization": f"Bearer {settings.RUNPOD_API_KEY}"}

    # Try the /run async endpoint (handles cold starts gracefully)
    prompt = _format_mistral_prompt(messages)
    run_url = f"{settings.VLLM_BASE_URL}/run"
    run_payload = {
        "input": {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
    }

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(run_url, json=run_payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()

        # Synchronous response (worker was warm)
        if data.get("status") == "COMPLETED":
            return _extract_response(data)

        # Async response — poll for result
        if data.get("id"):
            return await _poll_runpod(data["id"], headers)

        return _extract_response(data)


async def _call_runpod_async(
    messages: list[dict], max_tokens: int, temperature: float
) -> str:
    """Fallback: always use /run + poll for cold start scenarios."""
    return await _call_runpod(messages, max_tokens, temperature)


async def _poll_runpod(job_id: str, headers: dict, max_attempts: int = 60) -> str:
    """Poll RunPod for async job result. Waits up to ~2 minutes for cold starts."""
    status_url = f"{settings.VLLM_BASE_URL}/status/{job_id}"
    logger.info("Polling RunPod job %s...", job_id)

    async with httpx.AsyncClient(timeout=30) as client:
        for attempt in range(max_attempts):
            await asyncio.sleep(2)
            try:
                resp = await client.get(status_url, headers=headers)
                data = resp.json()
            except Exception as e:
                logger.warning("Poll attempt %d failed: %s", attempt, e)
                continue

            status = data.get("status", "")
            if status == "COMPLETED":
                logger.info("RunPod job %s completed after %d polls", job_id, attempt + 1)
                return _extract_response(data)
            if status == "FAILED":
                error = data.get("error", "Unknown error")
                logger.error("RunPod job %s failed: %s", job_id, error)
                return f"I'm sorry, the inference service encountered an error. Please try again. (RunPod: {error})"
            # IN_QUEUE or IN_PROGRESS — keep polling
            if attempt % 5 == 0:
                logger.info("Job %s status: %s (poll %d)", job_id, status, attempt + 1)

    return "I'm sorry, the request timed out. The serverless worker may be starting up — please try again in about 30 seconds."


async def _call_vllm_openai(
    messages: list[dict], max_tokens: int, temperature: float
) -> str:
    """Call direct vLLM OpenAI-compatible endpoint."""
    url = f"{settings.VLLM_BASE_URL}/chat/completions"
    payload = {
        "model": settings.VLLM_MODEL_NAME,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    async with httpx.AsyncClient(timeout=settings.VLLM_TIMEOUT_SECONDS) as client:
        resp = await client.post(url, json=payload)
        resp.raise_for_status()
        return _extract_response(resp.json())
