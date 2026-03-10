#!/usr/bin/env python3
"""
Validate AlvinAI model on a RunPod Serverless vLLM endpoint.

Usage:
    python scripts/eval_model.py --endpoint-url <RUNPOD_URL> --api-key <RUNPOD_API_KEY>

    # Or set env vars:
    export RUNPOD_ENDPOINT_URL=https://api.runpod.ai/v2/<endpoint_id>
    export RUNPOD_API_KEY=<your_key>
    python scripts/eval_model.py
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

try:
    import requests
except ImportError:
    print("Installing requests...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "requests", "-q"])
    import requests


SYSTEM_PROMPT = (
    "You are an AI Assistant for an automotive engineering company. You assist "
    "employees, business partners, vendors, customers, and internal engineering "
    "teams with accurate, professional, and helpful responses. You have knowledge "
    "of HR policies, procurement processes, product and warranty information, and "
    "engineering documentation including Technical Service Bulletins, specifications, "
    "and compliance standards. Always be concise, factual, and guide users to the "
    "right next step where applicable."
)

# Test prompts across all 4 namespace categories
VALIDATION_PROMPTS = [
    {
        "category": "customer_support",
        "query": "What warranty coverage does my new vehicle have?",
        "expect_keywords": ["warranty", "year", "coverage", "dealer"],
    },
    {
        "category": "customer_support",
        "query": "How do I book a service appointment for my car?",
        "expect_keywords": ["service", "appointment", "dealer", "book"],
    },
    {
        "category": "engineering",
        "query": "What is the torque specification for cylinder head bolts on the 2.0L engine?",
        "expect_keywords": ["torque", "Nm", "bolt", "specification"],
    },
    {
        "category": "dealer_sales",
        "query": "What financing options are available for fleet purchases?",
        "expect_keywords": ["financ", "fleet", "option"],
    },
    {
        "category": "compliance",
        "query": "What are the FMVSS crash test requirements for frontal impact?",
        "expect_keywords": ["FMVSS", "crash", "impact", "standard"],
    },
    {
        "category": "employee_hr",
        "query": "How many days of annual leave am I entitled to?",
        "expect_keywords": ["leave", "day", "annual", "entitle"],
    },
    {
        "category": "employee_hr",
        "query": "What is the remote work policy?",
        "expect_keywords": ["remote", "work", "policy", "home"],
    },
    {
        "category": "vendor",
        "query": "What are the payment terms for approved suppliers?",
        "expect_keywords": ["payment", "term", "supplier", "invoice"],
    },
]


def format_mistral_prompt(messages: list[dict]) -> str:
    """Format messages into Mistral instruct format."""
    # Mistral uses [INST] ... [/INST] format
    prompt = ""
    system_msg = ""
    for msg in messages:
        if msg["role"] == "system":
            system_msg = msg["content"]
        elif msg["role"] == "user":
            user_text = msg["content"]
            if system_msg:
                user_text = f"{system_msg}\n\n{user_text}"
                system_msg = ""
            prompt += f"[INST] {user_text} [/INST]"
        elif msg["role"] == "assistant":
            prompt += f" {msg['content']}</s>"
    return prompt


def call_runpod_openai(endpoint_url: str, api_key: str, messages: list[dict], max_tokens: int = 512, temperature: float = 0.3) -> dict:
    """Call RunPod serverless vLLM endpoint.

    Supports both:
    - RunPod vLLM worker format (raw prompt via /runsync)
    - OpenAI-compatible chat route (openai_route via /runsync)
    """
    url = f"{endpoint_url.rstrip('/')}/runsync"

    # Try OpenAI chat route first — cleaner if the worker supports it
    payload = {
        "input": {
            "openai_route": "/v1/chat/completions",
            "openai_input": {
                "model": "sarfarazflow/alvinai-v1",
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
        }
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    resp = requests.post(url, json=payload, headers=headers, timeout=120)
    resp.raise_for_status()
    result = resp.json()

    # If the worker doesn't support openai_route, fall back to raw prompt
    if result.get("status") == "FAILED" or (result.get("output") and "error" in str(result["output"]).lower()):
        prompt = format_mistral_prompt(messages)
        payload = {
            "input": {
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
        }
        resp = requests.post(url, json=payload, headers=headers, timeout=120)
        resp.raise_for_status()
        result = resp.json()

    return result


def call_openai_compatible(base_url: str, api_key: str, messages: list[dict], max_tokens: int = 512, temperature: float = 0.3) -> dict:
    """Call a standard OpenAI-compatible endpoint (direct vLLM)."""
    url = f"{base_url.rstrip('/')}/v1/chat/completions"

    payload = {
        "model": "sarfarazflow/alvinai-v1",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    resp = requests.post(url, json=payload, headers=headers, timeout=120)
    resp.raise_for_status()
    return resp.json()


def extract_response_text(result: dict, is_runpod: bool) -> str:
    """Extract the assistant's response text from the API result."""
    if is_runpod:
        # RunPod wraps output in a list or dict
        output = result.get("output", result)

        # Handle list format: [{"choices": [{"tokens": [...]}]}]
        if isinstance(output, list) and output:
            output = output[0]

        if isinstance(output, dict):
            choices = output.get("choices", [])
        else:
            return str(output)
    else:
        choices = result.get("choices", [])

    if not choices:
        return ""

    choice = choices[0]

    # Format 1: RunPod vLLM worker — {"tokens": ["text"]}
    if "tokens" in choice:
        tokens = choice["tokens"]
        if isinstance(tokens, list):
            return "".join(tokens).strip()
        return str(tokens).strip()

    # Format 2: Standard OpenAI chat — {"message": {"content": "text"}}
    msg = choice.get("message", {})
    if isinstance(msg, dict) and "content" in msg:
        return msg["content"].strip()

    # Format 3: Completions — {"text": "..."}
    if "text" in choice:
        return choice["text"].strip()

    return str(choice)


def check_keywords(text: str, keywords: list[str]) -> tuple[bool, list[str]]:
    """Check if response contains expected keywords (case-insensitive)."""
    text_lower = text.lower()
    found = [kw for kw in keywords if kw.lower() in text_lower]
    missing = [kw for kw in keywords if kw.lower() not in text_lower]
    return len(missing) == 0, missing


def run_validation(endpoint_url: str, api_key: str, is_runpod: bool = True):
    """Run all validation prompts and report results."""
    print("=" * 70)
    print("AlvinAI Model Validation")
    print(f"Endpoint: {endpoint_url}")
    print(f"Mode: {'RunPod Serverless' if is_runpod else 'Direct OpenAI-compatible'}")
    print("=" * 70)

    # 1. Health check — test with a simple prompt first
    print("\n[1/3] Health Check — sending test prompt...")
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "Hello, who are you?"},
    ]

    try:
        start = time.time()
        if is_runpod:
            result = call_runpod_openai(endpoint_url, api_key, messages, max_tokens=128)
        else:
            result = call_openai_compatible(endpoint_url, api_key, messages, max_tokens=128)
        elapsed = time.time() - start

        text = extract_response_text(result, is_runpod)
        if not text:
            print(f"  FAIL: Empty response. Raw: {json.dumps(result, indent=2)[:500]}")
            return False
        print(f"  PASS: Got response in {elapsed:.1f}s")
        print(f"  Response: {text[:200]}...")
    except Exception as e:
        print(f"  FAIL: {e}")
        return False

    # 2. Run category validation prompts
    print(f"\n[2/3] Category Validation — {len(VALIDATION_PROMPTS)} prompts across namespaces...")
    results = []

    for i, test in enumerate(VALIDATION_PROMPTS, 1):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": test["query"]},
        ]

        try:
            start = time.time()
            if is_runpod:
                result = call_runpod_openai(endpoint_url, api_key, messages)
            else:
                result = call_openai_compatible(endpoint_url, api_key, messages)
            elapsed = time.time() - start

            text = extract_response_text(result, is_runpod)
            all_found, missing = check_keywords(text, test["expect_keywords"])

            status = "PASS" if all_found else "PARTIAL"
            results.append({
                "category": test["category"],
                "query": test["query"],
                "passed": all_found,
                "missing_keywords": missing,
                "response_length": len(text),
                "latency_s": elapsed,
                "response": text,
            })

            kw_status = f"keywords: {len(test['expect_keywords']) - len(missing)}/{len(test['expect_keywords'])}"
            print(f"  [{i}/{len(VALIDATION_PROMPTS)}] {status} | {test['category']:20s} | {elapsed:.1f}s | {kw_status}")
            if missing:
                print(f"         Missing: {missing}")

        except Exception as e:
            print(f"  [{i}/{len(VALIDATION_PROMPTS)}] FAIL | {test['category']:20s} | Error: {e}")
            results.append({
                "category": test["category"],
                "query": test["query"],
                "passed": False,
                "error": str(e),
            })

    # 3. Summary
    print(f"\n[3/3] Summary")
    print("-" * 50)
    passed = sum(1 for r in results if r.get("passed"))
    total = len(results)
    avg_latency = sum(r.get("latency_s", 0) for r in results) / max(total, 1)
    avg_length = sum(r.get("response_length", 0) for r in results) / max(total, 1)

    print(f"  Prompts passed:    {passed}/{total}")
    print(f"  Avg latency:       {avg_latency:.1f}s")
    print(f"  Avg response len:  {avg_length:.0f} chars")

    # Quality checks
    issues = []
    for r in results:
        resp = r.get("response", "")
        if len(resp) < 30:
            issues.append(f"  Very short response for '{r['category']}': {len(resp)} chars")
        if resp and resp == resp.upper():
            issues.append(f"  All-caps response in '{r['category']}'")

    if issues:
        print("\n  Quality warnings:")
        for issue in issues:
            print(f"    {issue}")

    # Write detailed results
    output_path = Path(__file__).parent.parent / "validation_results.json"
    with open(output_path, "w") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "endpoint": endpoint_url,
            "model": "sarfarazflow/alvinai-v1",
            "summary": {
                "passed": passed,
                "total": total,
                "avg_latency_s": round(avg_latency, 2),
                "avg_response_length": round(avg_length),
            },
            "results": results,
        }, f, indent=2)
    print(f"\n  Detailed results saved to: {output_path}")

    # Print sample responses
    print("\n" + "=" * 70)
    print("SAMPLE RESPONSES")
    print("=" * 70)
    for r in results[:4]:
        print(f"\n--- {r['category']} ---")
        print(f"Q: {r['query']}")
        print(f"A: {r.get('response', 'N/A')[:400]}")

    return passed == total


def main():
    parser = argparse.ArgumentParser(description="Validate AlvinAI model on RunPod")
    parser.add_argument("--endpoint-url", default=os.environ.get("RUNPOD_ENDPOINT_URL", ""),
                        help="RunPod endpoint URL or direct vLLM URL")
    parser.add_argument("--api-key", default=os.environ.get("RUNPOD_API_KEY", ""),
                        help="API key for the endpoint")
    parser.add_argument("--direct", action="store_true",
                        help="Use direct OpenAI-compatible mode instead of RunPod serverless wrapper")
    args = parser.parse_args()

    if not args.endpoint_url:
        print("Error: --endpoint-url or RUNPOD_ENDPOINT_URL env var required")
        print("  RunPod:  https://api.runpod.ai/v2/<endpoint_id>")
        print("  Direct:  http://<host>:8080")
        sys.exit(1)

    if not args.api_key:
        print("Error: --api-key or RUNPOD_API_KEY env var required")
        sys.exit(1)

    success = run_validation(args.endpoint_url, args.api_key, is_runpod=not args.direct)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
