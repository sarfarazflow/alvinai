#!/usr/bin/env python3
"""Generate RAGAS eval datasets from corpus documents using template-based extraction.

Reads corpus .txt files, extracts factual passages, and generates
question/answer/context triples per namespace.

Usage:
    python scripts/generate_eval_from_corpus.py \
        --corpus-dir /path/to/corpus-utility/output/corpus \
        --output-dir data/v1/eval \
        --per-namespace 50
"""

import argparse
import json
import os
import re
import hashlib
from pathlib import Path

# Map corpus domains to AlvinAI namespaces
DOMAIN_TO_NAMESPACE = {
    # customer_support
    "after_sales": "customer_support",
    # engineering
    "engineering_adas": "engineering",
    "engineering_design": "engineering",
    "engineering_ev": "engineering",
    "engineering_manufacturing": "engineering",
    "engineering_vehicle": "engineering",
    "problem_solving": "engineering",
    "lean_tps": "engineering",
    # dealer_sales
    "market_industry": "dealer_sales",
    # compliance
    "compliance_legal": "compliance",
    "standards_edi": "compliance",
    "standards_electrical": "compliance",
    "standards_emissions": "compliance",
    "standards_engineering": "compliance",
    "standards_ev_battery": "compliance",
    "standards_quality": "compliance",
    "standards_safety": "compliance",
    "standards_sustainability": "compliance",
    "standards_testing": "compliance",
    # employee_hr
    "hse": "employee_hr",
    "management_frameworks": "employee_hr",
    # vendor
    "operations_supply_chain": "vendor",
    "operations_production": "vendor",
    "operations_quality": "vendor",
    # skip (not directly mappable or less relevant)
    "finance_tax": "vendor",
    "it_systems": "engineering",
}

# Question templates by type
TEMPLATES = {
    "definition": [
        "What is {topic}?",
        "Define {topic}.",
        "Explain the concept of {topic}.",
    ],
    "factual": [
        "According to the document on {topic}, what is {detail}?",
        "What does the {topic} document state about {detail}?",
    ],
    "purpose": [
        "What is the purpose of {topic}?",
        "Why is {topic} important in the automotive industry?",
    ],
    "component": [
        "What are the key components of {topic}?",
        "What elements make up {topic}?",
    ],
}


def extract_title(text: str, filename: str) -> str:
    """Extract title from first heading or filename."""
    lines = text.strip().split("\n")
    for line in lines[:5]:
        line = line.strip()
        if line.startswith("# "):
            return line.lstrip("# ").strip()
    return filename.replace("_", " ").replace(".txt", "")


def extract_sections(text: str) -> list[dict]:
    """Extract sections with headings and content."""
    sections = []
    current_heading = ""
    current_content = []

    for line in text.split("\n"):
        if line.startswith("## "):
            if current_heading and current_content:
                content = "\n".join(current_content).strip()
                if len(content) > 100:
                    sections.append({"heading": current_heading, "content": content})
            current_heading = line.lstrip("# ").strip()
            current_content = []
        else:
            current_content.append(line)

    if current_heading and current_content:
        content = "\n".join(current_content).strip()
        if len(content) > 100:
            sections.append({"heading": current_heading, "content": content})

    return sections


def extract_key_sentences(text: str, max_sentences: int = 3) -> list[str]:
    """Extract key factual sentences from text."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    key = []
    for s in sentences:
        s = s.strip()
        if len(s) < 30 or len(s) > 500:
            continue
        # Skip reference/link sentences
        if s.startswith("[") or "http" in s or s.startswith("See also"):
            continue
        # Prefer sentences with factual indicators
        if any(w in s.lower() for w in [
            "is defined", "refers to", "consists of", "requires", "must",
            "standard", "regulation", "process", "method", "system",
            "designed to", "used for", "responsible for", "ensures",
            "according to", "specified", "mandates", "compliance",
        ]):
            key.append(s)
        elif len(key) < max_sentences:
            key.append(s)
        if len(key) >= max_sentences:
            break
    return key


def generate_questions_from_doc(
    title: str, sections: list[dict], namespace: str, domain: str, doc_idx: int
) -> list[dict]:
    """Generate eval questions from a single document."""
    questions = []
    q_idx = 0

    # Definition question from first section
    if sections:
        first = sections[0]
        key_sentences = extract_key_sentences(first["content"], 2)
        if key_sentences:
            q_idx += 1
            questions.append({
                "id": f"{namespace.upper()}-EVAL-{doc_idx:03d}-{q_idx:02d}",
                "question": f"What is {title}?",
                "ground_truth": " ".join(key_sentences),
                "reference_context": first["content"][:800],
                "namespace": namespace,
                "source_doc": title,
                "source_section": first["heading"] or "Introduction",
                "question_type": "definition",
                "topic": domain,
            })

    # Section-specific questions
    for section in sections[:6]:
        heading = section["heading"]
        content = section["content"]
        key_sentences = extract_key_sentences(content, 2)

        if not key_sentences:
            continue

        # Factual question about the section
        q_idx += 1
        questions.append({
            "id": f"{namespace.upper()}-EVAL-{doc_idx:03d}-{q_idx:02d}",
            "question": f"What does the document on {title} state about {heading.lower()}?",
            "ground_truth": " ".join(key_sentences),
            "reference_context": content[:800],
            "namespace": namespace,
            "source_doc": title,
            "source_section": heading,
            "question_type": "factual",
            "topic": domain,
        })

        # Purpose/component question if heading suggests it
        if any(w in heading.lower() for w in ["purpose", "objective", "goal", "overview", "introduction"]):
            q_idx += 1
            questions.append({
                "id": f"{namespace.upper()}-EVAL-{doc_idx:03d}-{q_idx:02d}",
                "question": f"What is the purpose of {title}?",
                "ground_truth": " ".join(key_sentences),
                "reference_context": content[:800],
                "namespace": namespace,
                "source_doc": title,
                "source_section": heading,
                "question_type": "purpose",
                "topic": domain,
            })

        if any(w in heading.lower() for w in ["component", "element", "structure", "type", "classification"]):
            q_idx += 1
            questions.append({
                "id": f"{namespace.upper()}-EVAL-{doc_idx:03d}-{q_idx:02d}",
                "question": f"What are the key components or types of {title}?",
                "ground_truth": " ".join(key_sentences),
                "reference_context": content[:800],
                "namespace": namespace,
                "source_doc": title,
                "source_section": heading,
                "question_type": "component",
                "topic": domain,
            })

    return questions


def process_corpus(corpus_dir: str, output_dir: str, per_namespace: int):
    """Process all corpus docs and generate eval datasets."""
    corpus_path = Path(corpus_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Collect all questions by namespace
    all_questions: dict[str, list] = {}
    doc_idx = 0

    for domain_dir in sorted(corpus_path.iterdir()):
        if not domain_dir.is_dir():
            continue

        domain = domain_dir.name
        namespace = DOMAIN_TO_NAMESPACE.get(domain)
        if not namespace:
            print(f"  Skipping unmapped domain: {domain}")
            continue

        txt_files = sorted(domain_dir.glob("*.txt"))
        if not txt_files:
            continue

        print(f"  {domain} → {namespace}: {len(txt_files)} docs")

        for txt_file in txt_files:
            doc_idx += 1
            text = txt_file.read_text(encoding="utf-8", errors="replace")

            if len(text) < 200:
                continue

            title = extract_title(text, txt_file.name)
            sections = extract_sections(text)

            if not sections:
                # No headings — treat whole doc as one section
                sections = [{"heading": "Overview", "content": text[:2000]}]

            questions = generate_questions_from_doc(
                title, sections, namespace, domain, doc_idx
            )

            if namespace not in all_questions:
                all_questions[namespace] = []
            all_questions[namespace].extend(questions)

    # Write per-namespace eval files, capped at per_namespace
    total = 0
    for namespace, questions in sorted(all_questions.items()):
        # Deduplicate by question text
        seen = set()
        unique = []
        for q in questions:
            key = q["question"]
            if key not in seen:
                seen.add(key)
                unique.append(q)

        # Cap at per_namespace
        selected = unique[:per_namespace]

        # Re-number IDs
        for i, q in enumerate(selected, 1):
            q["id"] = f"{namespace.upper()}-EVAL-{i:03d}"

        outfile = output_path / f"{namespace}_eval.jsonl"
        with open(outfile, "w") as f:
            for q in selected:
                f.write(json.dumps(q, ensure_ascii=False) + "\n")

        print(f"  ✓ {namespace}: {len(selected)} questions → {outfile}")
        total += len(selected)

    print(f"\nTotal: {total} eval questions across {len(all_questions)} namespaces")


def main():
    parser = argparse.ArgumentParser(description="Generate RAGAS eval from corpus")
    parser.add_argument("--corpus-dir", required=True, help="Path to corpus directory")
    parser.add_argument("--output-dir", default="data/v1/eval", help="Output directory")
    parser.add_argument("--per-namespace", type=int, default=50, help="Max questions per namespace")
    args = parser.parse_args()

    print(f"Generating eval datasets from {args.corpus_dir}")
    print(f"Output: {args.output_dir}, max {args.per_namespace} per namespace\n")

    process_corpus(args.corpus_dir, args.output_dir, args.per_namespace)


if __name__ == "__main__":
    main()
