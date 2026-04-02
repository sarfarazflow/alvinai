#!/usr/bin/env python3
"""Generate v2 SFT dataset from the knowledge corpus (template-based).

No API calls required. Reads corpus documents, extracts facts from sections,
and composes Q&A pairs using question templates and answer builders.

Usage:
    python scripts/v2/prepare_sft_dataset.py
    python scripts/v2/prepare_sft_dataset.py --dry-run
    python scripts/v2/prepare_sft_dataset.py --namespace engineering
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from corpus_utils import (
    NAMESPACES,
    NAMESPACE_PERSONAS,
    QUESTION_PREFIXES,
    QUESTION_SUFFIXES,
    QUESTION_TEMPLATES,
    CorpusDoc,
    Section,
    build_answer_from_content,
    corpus_summary,
    extract_dtc_codes,
    extract_key_values,
    extract_list_items,
    extract_part_numbers,
    extract_reg_references,
    extract_table_rows,
    extract_values_with_units,
    load_corpus,
    write_jsonl,
    SYSTEM_PROMPTS,
)


# ---------------------------------------------------------------------------
# Detail extraction — what to fill into question templates
# ---------------------------------------------------------------------------


def extract_details(section: Section, doc: CorpusDoc) -> list[str]:
    """Extract concrete details from a section to use in question templates."""
    details: list[str] = []

    # Use heading as a detail
    if section.heading and len(section.heading) > 3:
        details.append(section.heading)

    # Key-value pairs
    for k, v in extract_key_values(section.content):
        details.append(f"{k}")
        if len(v) < 80:
            details.append(f"{k} ({v})")

    # Table headers as topics
    tables = extract_table_rows(section.content)
    if tables and len(tables) > 1:
        for row in tables[1:3]:  # first data rows
            for cell in row:
                cell = cell.strip()
                if cell and len(cell) > 3 and len(cell) < 80:
                    details.append(cell)

    # List items as topics
    for item in extract_list_items(section.content)[:5]:
        if len(item) < 80:
            details.append(item)

    # DTC codes
    for dtc in extract_dtc_codes(section.content)[:3]:
        details.append(f"DTC {dtc}")

    # Part numbers
    for pn in extract_part_numbers(section.content)[:3]:
        details.append(f"part number {pn}")

    # Regulation references
    for reg in extract_reg_references(section.content)[:3]:
        details.append(reg)

    # Values with units
    for val, unit in extract_values_with_units(section.content)[:3]:
        details.append(f"the {val} {unit} specification")

    # Doc title as fallback
    if not details:
        details.append(doc.title)

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for d in details:
        d_lower = d.lower()
        if d_lower not in seen:
            seen.add(d_lower)
            unique.append(d)

    return unique


# ---------------------------------------------------------------------------
# SFT pair generation
# ---------------------------------------------------------------------------


def generate_sft_pairs(
    doc: CorpusDoc,
    section: Section,
    num_pairs: int,
    namespace: str,
    rng: random.Random,
) -> list[dict]:
    """Generate Q&A pairs for a single section using templates."""
    details = extract_details(section, doc)
    templates = QUESTION_TEMPLATES.get(namespace, QUESTION_TEMPLATES["customer_support"])
    system_prompt = SYSTEM_PROMPTS.get(namespace, SYSTEM_PROMPTS["customer_support"])

    records = []
    used_questions: set[str] = set()

    for i in range(num_pairs):
        # Pick a template and detail
        template = templates[i % len(templates)]
        detail = details[i % len(details)] if details else doc.title

        # Build question with variation
        try:
            question = template.format(topic=doc.title, detail=detail)
        except (KeyError, IndexError):
            question = template.replace("{detail}", detail).replace("{topic}", doc.title)

        # Add conversational prefix/suffix for variety
        prefix = rng.choice(QUESTION_PREFIXES)
        suffix = rng.choice(QUESTION_SUFFIXES)
        question = f"{prefix}{question}{suffix}".strip()

        # Skip duplicates
        if question.lower() in used_questions:
            continue
        used_questions.add(question.lower())

        # Build answer
        answer = build_answer_from_content(
            section.content, doc.title, section.section_id, namespace
        )

        records.append({
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ],
            "metadata": {
                "category": namespace,
                "source_doc": doc.doc_id,
                "source_section": section.section_id,
            },
        })

    return records


def generate_all_sft(
    docs: list[CorpusDoc],
    target_per_namespace: int,
    namespaces: list[str] | None = None,
    seed: int = 42,
) -> list[dict]:
    """Generate the full SFT dataset across all namespaces."""
    rng = random.Random(seed)
    ns_filter = set(namespaces) if namespaces else set(NAMESPACES)
    all_records: list[dict] = []

    # Group sections by namespace
    ns_sections: dict[str, list[tuple[CorpusDoc, Section]]] = defaultdict(list)
    for doc in docs:
        if doc.namespace not in ns_filter:
            continue
        for section in doc.sections:
            ns_sections[doc.namespace].append((doc, section))

    for ns in sorted(ns_filter):
        sections = ns_sections.get(ns, [])
        if not sections:
            print(f"  WARNING: No sections for namespace '{ns}'")
            continue

        # Distribute pairs across sections
        pairs_per_section = max(1, target_per_namespace // len(sections))
        remainder = target_per_namespace - (pairs_per_section * len(sections))

        ns_records: list[dict] = []
        for i, (doc, section) in enumerate(sections):
            n = pairs_per_section + (1 if i < remainder else 0)
            n = min(n, 30)  # cap per section
            pairs = generate_sft_pairs(doc, section, n, ns, rng)
            ns_records.extend(pairs)

        # Trim to target if overshot
        if len(ns_records) > target_per_namespace:
            rng.shuffle(ns_records)
            ns_records = ns_records[:target_per_namespace]

        all_records.extend(ns_records)
        print(f"  {ns:<20} {len(ns_records)} pairs")

    return all_records


# ---------------------------------------------------------------------------
# Train/val split
# ---------------------------------------------------------------------------


def split_train_val(
    records: list[dict],
    val_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """Split records into train/val, stratified by namespace at document level."""
    rng = random.Random(seed)

    doc_groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in records:
        key = (r["metadata"]["category"], r["metadata"]["source_doc"])
        doc_groups[key].append(r)

    ns_docs: dict[str, list[str]] = defaultdict(list)
    for (ns, doc_id) in doc_groups:
        if doc_id not in ns_docs[ns]:
            ns_docs[ns].append(doc_id)

    train, val = [], []
    for ns, doc_ids in ns_docs.items():
        rng.shuffle(doc_ids)
        n_val = max(1, round(len(doc_ids) * val_ratio))
        val_docs = set(doc_ids[:n_val])

        for doc_id in doc_ids:
            split = "val" if doc_id in val_docs else "train"
            for r in doc_groups[(ns, doc_id)]:
                r["metadata"]["split"] = split
                if split == "val":
                    val.append(r)
                else:
                    train.append(r)

    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


# ---------------------------------------------------------------------------
# Summary & CLI
# ---------------------------------------------------------------------------


def print_summary(train: list[dict], val: list[dict]) -> None:
    print("\n=== SFT v2 Generation Summary ===")
    print(f"{'Namespace':<20} {'Train':>6} {'Val':>6} {'Total':>6}")
    print("─" * 42)

    train_counts = Counter(r["metadata"]["category"] for r in train)
    val_counts = Counter(r["metadata"]["category"] for r in val)

    for ns in NAMESPACES:
        t = train_counts.get(ns, 0)
        v = val_counts.get(ns, 0)
        print(f"{ns:<20} {t:>6} {v:>6} {t + v:>6}")

    print("─" * 42)
    print(f"{'Total':<20} {len(train):>6} {len(val):>6} {len(train) + len(val):>6}")


def main():
    parser = argparse.ArgumentParser(description="Generate v2 SFT dataset (template-based)")
    parser.add_argument("--corpus-dir", default="knowledge/corpus_auto")
    parser.add_argument("--output-dir", default="data/v2/sft")
    parser.add_argument("--train-count", type=int, default=3000)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--namespace", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    print("Loading corpus...")
    docs = load_corpus(args.corpus_dir)
    print(corpus_summary(docs))

    namespaces = [args.namespace] if args.namespace else None
    n_ns = 1 if args.namespace else len(NAMESPACES)
    target_per_ns = math.ceil(args.train_count / n_ns)

    if args.dry_run:
        print(f"\n=== DRY RUN ===\nTarget: {args.train_count} train, {target_per_ns}/namespace\n")
        for ns in (namespaces or NAMESPACES):
            ns_docs = [d for d in docs if d.namespace == ns]
            ns_secs = sum(len(d.sections) for d in ns_docs)
            per_sec = round(target_per_ns / ns_secs, 1) if ns_secs else 0
            print(f"  {ns:<20} {ns_secs:>4} sections × ~{per_sec} pairs = {target_per_ns}")

        # Show sample output
        print("\n--- Sample Q&A pair ---")
        rng = random.Random(args.seed)
        for doc in docs[:1]:
            if doc.sections:
                pairs = generate_sft_pairs(doc, doc.sections[0], 2, doc.namespace, rng)
                for p in pairs:
                    print(f"\n  Q: {p['messages'][1]['content']}")
                    print(f"  A: {p['messages'][2]['content'][:200]}")
        return

    print(f"\nGenerating SFT pairs (target: {args.train_count} train)...")
    all_records = generate_all_sft(docs, target_per_ns, namespaces, seed=args.seed)

    train, val = split_train_val(all_records, val_ratio=args.val_ratio, seed=args.seed)

    print(f"\nWriting to {args.output_dir}/...")
    write_jsonl(train, Path(args.output_dir) / "train.jsonl")
    write_jsonl(val, Path(args.output_dir) / "val.jsonl")

    print_summary(train, val)


if __name__ == "__main__":
    main()
