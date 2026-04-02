#!/usr/bin/env python3
"""Generate v2 RAFT dataset from the knowledge corpus (template-based).

No API calls required. Assembles context windows with oracle + distractor
documents and generates grounded answers with citation markers.

Usage:
    python scripts/v2/prepare_raft_dataset.py
    python scripts/v2/prepare_raft_dataset.py --dry-run
    python scripts/v2/prepare_raft_dataset.py --namespace engineering
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
    ABSTENTION_TEMPLATES,
    NAMESPACES,
    NAMESPACE_PERSONAS,
    QUESTION_PREFIXES,
    QUESTION_SUFFIXES,
    QUESTION_TEMPLATES,
    RAFT_SYSTEM_PROMPT,
    CorpusDoc,
    Section,
    build_raft_answer,
    corpus_summary,
    extract_dtc_codes,
    extract_key_values,
    extract_list_items,
    extract_part_numbers,
    extract_reg_references,
    extract_table_rows,
    extract_values_with_units,
    load_corpus,
    truncate,
    write_jsonl,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUM_DISTRACTORS = 3
ORACLE_RATIO = 0.70
MAX_ORACLE_CHARS = 1500
MAX_DISTRACTOR_CHARS = 600


# ---------------------------------------------------------------------------
# Detail extraction (reused from SFT)
# ---------------------------------------------------------------------------


def extract_details(section: Section, doc: CorpusDoc) -> list[str]:
    """Extract concrete details from a section for question templates."""
    details: list[str] = []

    if section.heading and len(section.heading) > 3:
        details.append(section.heading)

    for k, v in extract_key_values(section.content):
        details.append(k)

    tables = extract_table_rows(section.content)
    if tables and len(tables) > 1:
        for row in tables[1:3]:
            for cell in row:
                cell = cell.strip()
                if cell and 3 < len(cell) < 80:
                    details.append(cell)

    for item in extract_list_items(section.content)[:5]:
        if len(item) < 80:
            details.append(item)

    for dtc in extract_dtc_codes(section.content)[:3]:
        details.append(f"DTC {dtc}")

    for pn in extract_part_numbers(section.content)[:3]:
        details.append(f"part number {pn}")

    for reg in extract_reg_references(section.content)[:3]:
        details.append(reg)

    if not details:
        details.append(doc.title)

    seen = set()
    unique = []
    for d in details:
        d_lower = d.lower()
        if d_lower not in seen:
            seen.add(d_lower)
            unique.append(d)
    return unique


# ---------------------------------------------------------------------------
# Distractor selection
# ---------------------------------------------------------------------------


def build_distractor_index(
    docs: list[CorpusDoc],
) -> dict[str, dict[str, list[tuple[CorpusDoc, Section]]]]:
    """Build index: {namespace: {category: [(doc, section), ...]}}"""
    index: dict[str, dict[str, list[tuple[CorpusDoc, Section]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for doc in docs:
        for section in doc.sections:
            index[doc.namespace][doc.category].append((doc, section))
    return index


def select_distractors(
    oracle_doc: CorpusDoc,
    oracle_section: Section,
    distractor_index: dict[str, dict[str, list[tuple[CorpusDoc, Section]]]],
    num_distractors: int,
    rng: random.Random,
) -> list[tuple[CorpusDoc, Section]]:
    """Select non-trivial distractors using tiered strategy."""
    ns = oracle_doc.namespace
    cat = oracle_doc.category
    ns_index = distractor_index.get(ns, {})

    tier1 = [
        (d, s) for cat_secs in ns_index.values() for d, s in cat_secs
        if d.doc_id == oracle_doc.doc_id and s.section_id != oracle_section.section_id
    ]
    tier2 = [
        (d, s) for d, s in ns_index.get(cat, [])
        if d.doc_id != oracle_doc.doc_id
    ]
    tier3 = [
        (d, s) for other_cat, secs in ns_index.items()
        if other_cat != cat for d, s in secs
    ]

    selected: list[tuple[CorpusDoc, Section]] = []
    used: set[str] = {oracle_section.section_id}

    t1_count = max(1, round(num_distractors * 0.50)) if tier1 else 0
    t2_count = max(1, round(num_distractors * 0.35)) if tier2 else 0
    t3_count = num_distractors - t1_count - t2_count

    for pool, count in [(tier1, t1_count), (tier2, t2_count), (tier3, t3_count)]:
        available = [(d, s) for d, s in pool if s.section_id not in used]
        picked = rng.sample(available, min(count, len(available))) if available else []
        for d, s in picked:
            selected.append((d, s))
            used.add(s.section_id)

    while len(selected) < num_distractors:
        all_remaining = [
            (d, s) for pool in [tier1, tier2, tier3]
            for d, s in pool if s.section_id not in used
        ]
        if not all_remaining:
            break
        pick = rng.choice(all_remaining)
        selected.append(pick)
        used.add(pick[1].section_id)

    return selected[:num_distractors]


# ---------------------------------------------------------------------------
# Context formatting
# ---------------------------------------------------------------------------


def format_context_documents(
    documents: list[tuple[CorpusDoc, Section, bool]],
    rng: random.Random,
) -> tuple[str, int | None]:
    """Format documents into RAFT context block. Returns (context_str, oracle_position)."""
    shuffled = list(documents)
    rng.shuffle(shuffled)

    oracle_pos = None
    blocks = []
    for i, (doc, section, is_oracle) in enumerate(shuffled, 1):
        max_chars = MAX_ORACLE_CHARS if is_oracle else MAX_DISTRACTOR_CHARS
        content = truncate(section.content, max_chars)
        block = f"[Document {i}]\nTitle: {doc.title}: {section.heading}\nID: {section.section_id}\n{content}"
        blocks.append(block)
        if is_oracle:
            oracle_pos = i

    return "\n\n".join(blocks), oracle_pos


# ---------------------------------------------------------------------------
# Question generation
# ---------------------------------------------------------------------------


def generate_question(
    section: Section,
    doc: CorpusDoc,
    namespace: str,
    rng: random.Random,
    used: set[str] | None = None,
) -> str:
    """Generate a question from a section using templates."""
    details = extract_details(section, doc)
    templates = QUESTION_TEMPLATES.get(namespace, QUESTION_TEMPLATES["customer_support"])

    if used is None:
        used = set()

    for _ in range(20):  # try up to 20 times for uniqueness
        template = rng.choice(templates)
        detail = rng.choice(details) if details else doc.title
        try:
            question = template.format(topic=doc.title, detail=detail)
        except (KeyError, IndexError):
            question = template.replace("{detail}", detail).replace("{topic}", doc.title)

        prefix = rng.choice(QUESTION_PREFIXES)
        suffix = rng.choice(QUESTION_SUFFIXES)
        question = f"{prefix}{question}{suffix}".strip()

        if question.lower() not in used:
            used.add(question.lower())
            return question

    return f"What information does {doc.title} provide about {section.heading}?"


# ---------------------------------------------------------------------------
# Core generation
# ---------------------------------------------------------------------------


def generate_raft_dataset(
    docs: list[CorpusDoc],
    target_per_namespace: int,
    namespaces: list[str] | None = None,
    seed: int = 42,
) -> list[dict]:
    """Generate the full RAFT dataset."""
    rng = random.Random(seed)
    ns_filter = set(namespaces) if namespaces else set(NAMESPACES)
    distractor_index = build_distractor_index(docs)
    all_records: list[dict] = []

    ns_sections: dict[str, list[tuple[CorpusDoc, Section]]] = defaultdict(list)
    for doc in docs:
        if doc.namespace not in ns_filter:
            continue
        for section in doc.sections:
            ns_sections[doc.namespace].append((doc, section))

    for ns in sorted(ns_filter):
        sections = ns_sections.get(ns, [])
        if not sections:
            continue

        # Calculate max oracle examples we can produce (sections × cap)
        max_oracle = len(sections) * 30
        oracle_target = round(target_per_namespace * ORACLE_RATIO)
        oracle_count = min(oracle_target, max_oracle)

        # Oracle-free is always 30% of actual oracle count to maintain ratio
        oracle_free_count = round(oracle_count * (1 - ORACLE_RATIO) / ORACLE_RATIO)

        used_questions: set[str] = set()
        ns_records: list[dict] = []

        # Oracle examples
        pairs_per_section = max(1, oracle_count // len(sections))
        remainder = oracle_count - (pairs_per_section * len(sections))

        for i, (doc, section) in enumerate(sections):
            n = pairs_per_section + (1 if i < remainder else 0)
            n = min(n, 30)

            for _ in range(n):
                question = generate_question(section, doc, ns, rng, used_questions)

                distractors = select_distractors(
                    doc, section, distractor_index, NUM_DISTRACTORS, rng
                )
                context_docs = [(doc, section, True)] + [
                    (d, s, False) for d, s in distractors
                ]
                context_str, _ = format_context_documents(context_docs, rng)

                answer = build_raft_answer(
                    section.content, doc.title, section.section_id
                )

                ns_records.append({
                    "messages": [
                        {"role": "system", "content": RAFT_SYSTEM_PROMPT},
                        {"role": "user", "content": f"Context documents:\n\n{context_str}\n\nQuestion: {question}"},
                        {"role": "assistant", "content": answer},
                    ],
                    "metadata": {
                        "type": "oracle",
                        "oracle_doc": section.section_id,
                        "category": ns,
                    },
                })

        # Oracle-free examples
        for _ in range(oracle_free_count):
            src_doc, src_section = rng.choice(sections)
            question = generate_question(src_section, src_doc, ns, rng, used_questions)

            distractor_pool = [
                (d, s) for d, s in sections
                if s.section_id != src_section.section_id
            ]
            n_dist = min(NUM_DISTRACTORS + 1, len(distractor_pool))
            if n_dist > 0:
                distractors = rng.sample(distractor_pool, n_dist)
            else:
                distractors = distractor_pool

            context_docs = [(d, s, False) for d, s in distractors]
            context_str, _ = format_context_documents(context_docs, rng)

            ns_records.append({
                "messages": [
                    {"role": "system", "content": RAFT_SYSTEM_PROMPT},
                    {"role": "user", "content": f"Context documents:\n\n{context_str}\n\nQuestion: {question}"},
                    {"role": "assistant", "content": rng.choice(ABSTENTION_TEMPLATES)},
                ],
                "metadata": {
                    "type": "oracle_free",
                    "oracle_doc": None,
                    "category": ns,
                },
            })

        # Trim to target
        if len(ns_records) > target_per_namespace:
            rng.shuffle(ns_records)
            ns_records = ns_records[:target_per_namespace]

        all_records.extend(ns_records)
        oracle = sum(1 for r in ns_records if r["metadata"]["type"] == "oracle")
        free = sum(1 for r in ns_records if r["metadata"]["type"] == "oracle_free")
        print(f"  {ns:<20} {len(ns_records)} total (oracle={oracle}, free={free})")

    return all_records


# ---------------------------------------------------------------------------
# Train/val split
# ---------------------------------------------------------------------------


def split_train_val(
    records: list[dict],
    val_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)

    ns_records: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        ns_records[r["metadata"]["category"]].append(r)

    train, val = [], []
    for ns, recs in ns_records.items():
        rng.shuffle(recs)
        n_val = max(1, round(len(recs) * val_ratio))
        for r in recs[:n_val]:
            r["metadata"]["split"] = "val"
            val.append(r)
        for r in recs[n_val:]:
            r["metadata"]["split"] = "train"
            train.append(r)

    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


# ---------------------------------------------------------------------------
# Summary & CLI
# ---------------------------------------------------------------------------


def print_summary(train: list[dict], val: list[dict]) -> None:
    all_records = train + val
    print("\n=== RAFT v2 Generation Summary ===")
    print(f"{'Namespace':<20} {'Train':>6} {'Val':>6} {'Total':>6}")
    print("─" * 42)

    train_counts = Counter(r["metadata"]["category"] for r in train)
    val_counts = Counter(r["metadata"]["category"] for r in val)

    for ns in NAMESPACES:
        t = train_counts.get(ns, 0)
        v = val_counts.get(ns, 0)
        print(f"{ns:<20} {t:>6} {v:>6} {t + v:>6}")

    print("─" * 42)
    print(f"{'Total':<20} {len(train):>6} {len(val):>6} {len(all_records):>6}")

    oracle = sum(1 for r in all_records if r["metadata"]["type"] == "oracle")
    free = len(all_records) - oracle
    print(f"\nOracle ratio:      {oracle}/{len(all_records)} ({oracle / len(all_records) * 100:.1f}%) — target: 70%")
    print(f"Oracle-free ratio: {free}/{len(all_records)} ({free / len(all_records) * 100:.1f}%) — target: 30%")


def main():
    parser = argparse.ArgumentParser(description="Generate v2 RAFT dataset (template-based)")
    parser.add_argument("--corpus-dir", default="knowledge/corpus_auto")
    parser.add_argument("--output-dir", default="data/v2/raft")
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
        print(f"\n=== DRY RUN ===\nTarget: {args.train_count}, {target_per_ns}/ns, oracle={ORACLE_RATIO:.0%}\n")
        for ns in (namespaces or NAMESPACES):
            ns_secs = sum(len(d.sections) for d in docs if d.namespace == ns)
            oracle = round(target_per_ns * ORACLE_RATIO)
            free = target_per_ns - oracle
            print(f"  {ns:<20} {ns_secs} sections → oracle={oracle}, free={free}")
        return

    print(f"\nGenerating RAFT examples (target: {args.train_count} train)...")
    all_records = generate_raft_dataset(docs, target_per_ns, namespaces, seed=args.seed)

    train, val = split_train_val(all_records, val_ratio=args.val_ratio, seed=args.seed)

    print(f"\nWriting to {args.output_dir}/...")
    write_jsonl(train, Path(args.output_dir) / "train.jsonl")
    write_jsonl(val, Path(args.output_dir) / "val.jsonl")

    print_summary(train, val)


if __name__ == "__main__":
    main()
