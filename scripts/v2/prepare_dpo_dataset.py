#!/usr/bin/env python3
"""Generate v2 DPO dataset from the knowledge corpus (template-based).

No API calls required. Creates preference pairs with template-based chosen
(correct) and rejected (flawed) responses across 5 failure categories.

Usage:
    python scripts/v2/prepare_dpo_dataset.py
    python scripts/v2/prepare_dpo_dataset.py --dry-run
    python scripts/v2/prepare_dpo_dataset.py --namespace engineering
"""

from __future__ import annotations

import argparse
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from corpus_utils import (
    NAMESPACES,
    NAMESPACE_PERSONAS,
    QUESTION_PREFIXES,
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
    extract_values_with_units,
    get_first_sentence,
    load_corpus,
    write_jsonl,
    SYSTEM_PROMPTS,
)

# ---------------------------------------------------------------------------
# Failure categories with weights
# ---------------------------------------------------------------------------

FAILURE_CATEGORIES = {
    "hallucination": 0.25,
    "tone": 0.20,
    "safety_scope": 0.20,
    "refusal": 0.15,
    "abstention": 0.20,
}

# Cross-namespace info for safety_scope violations
CROSS_NAMESPACE_LEAKS = {
    "customer_support": "Internal pricing shows dealer margin of 35% on this component (CONFIDENTIAL — dealer_sales namespace).",
    "engineering": "HR records indicate the lead engineer on this project is on performance improvement plan (CONFIDENTIAL — employee_hr namespace).",
    "dealer_sales": "Engineering TSB-2025-BR-001 flagged a potential safety recall on this product line — do not share with customers pending legal review (CONFIDENTIAL — engineering namespace).",
    "compliance": "The supplier's contract includes a penalty clause of $50,000 per day for non-compliance delays (CONFIDENTIAL — vendor namespace).",
    "employee_hr": "The engineering team's budget for this quarter has been cut by 20% due to underperformance on project milestones (CONFIDENTIAL — engineering namespace).",
    "vendor": "Customer complaint data shows a 15% return rate on this supplier's components in Q1 (CONFIDENTIAL — customer_support namespace).",
}

# Tone violations per namespace
WRONG_TONE = {
    "customer_support": "Yeah so basically that part's broken lol. Just take it to wherever and they'll sort it I guess. Not really a big deal tbh.",
    "engineering": "So like, the thingy might be a bit off? Maybe check it or something? I dunno, it's probably fine though!",
    "dealer_sales": "This product is acceptable. It meets minimum standards. There is nothing particularly notable about it compared to alternatives.",
    "compliance": "The regulation probably says something about this. You should maybe look into it when you get a chance. No rush though.",
    "employee_hr": "Look, just read the policy yourself. It's all in there somewhere. I don't have time to spell it out.",
    "vendor": "Eh, the supplier stuff is whatever. Just send them an email or something and they'll figure it out eventually.",
}


# ---------------------------------------------------------------------------
# Detail extraction (shared with SFT/RAFT)
# ---------------------------------------------------------------------------


def extract_details(section: Section, doc: CorpusDoc) -> list[str]:
    details: list[str] = []
    if section.heading and len(section.heading) > 3:
        details.append(section.heading)
    for k, v in extract_key_values(section.content):
        details.append(k)
    for item in extract_list_items(section.content)[:5]:
        if len(item) < 80:
            details.append(item)
    for dtc in extract_dtc_codes(section.content)[:2]:
        details.append(f"DTC {dtc}")
    for pn in extract_part_numbers(section.content)[:2]:
        details.append(f"part number {pn}")
    for reg in extract_reg_references(section.content)[:2]:
        details.append(reg)
    if not details:
        details.append(doc.title)
    return list(dict.fromkeys(details))  # dedupe preserving order


def generate_question(
    section: Section, doc: CorpusDoc, namespace: str, rng: random.Random
) -> str:
    details = extract_details(section, doc)
    templates = QUESTION_TEMPLATES.get(namespace, QUESTION_TEMPLATES["customer_support"])
    template = rng.choice(templates)
    detail = rng.choice(details)
    try:
        question = template.format(topic=doc.title, detail=detail)
    except (KeyError, IndexError):
        question = template.replace("{detail}", detail).replace("{topic}", doc.title)
    prefix = rng.choice(QUESTION_PREFIXES)
    return f"{prefix}{question}".strip()


# ---------------------------------------------------------------------------
# Rejected response generators
# ---------------------------------------------------------------------------


def generate_hallucination_rejected(
    section: Section, doc: CorpusDoc, chosen: str, rng: random.Random
) -> str:
    """Subtly alter facts in the chosen answer."""
    rejected = chosen

    # Alter numeric values
    values = extract_values_with_units(section.content)
    if values:
        for val, unit in values[:2]:
            try:
                num = float(val)
                # Alter by 15-40%
                factor = rng.choice([0.6, 0.7, 0.85, 1.15, 1.3, 1.5])
                new_val = f"{num * factor:.1f}" if "." in val else str(int(num * factor))
                rejected = rejected.replace(f"{val} {unit}", f"{new_val} {unit}", 1)
            except ValueError:
                pass

    # Alter part numbers
    part_nums = extract_part_numbers(section.content)
    if part_nums:
        pn = part_nums[0]
        # Change last digit/letter
        chars = list(pn)
        if chars[-1].isdigit():
            chars[-1] = str((int(chars[-1]) + rng.randint(1, 5)) % 10)
        else:
            chars[-1] = rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ")
        fake_pn = "".join(chars)
        rejected = rejected.replace(pn, fake_pn, 1)

    # Alter DTC codes
    dtcs = extract_dtc_codes(section.content)
    if dtcs:
        dtc = dtcs[0]
        # Change middle digits
        fake_dtc = dtc[0] + str(rng.randint(1, 9)) + dtc[2:]
        rejected = rejected.replace(dtc, fake_dtc, 1)

    # Alter regulation references
    regs = extract_reg_references(section.content)
    if regs:
        reg = regs[0]
        # Append a fake subsection
        rejected = rejected.replace(reg, f"{reg} Section 99.4(b)", 1)

    # If nothing was altered, fabricate a detail
    if rejected == chosen:
        rejected = chosen + " Additionally, this applies to all model years from 2018 onwards with production codes starting with 7X."

    return rejected


def generate_tone_rejected(
    namespace: str, section: Section, doc: CorpusDoc, rng: random.Random
) -> str:
    """Generate a response with wrong tone for the namespace."""
    return WRONG_TONE.get(namespace, WRONG_TONE["customer_support"])


def generate_safety_scope_rejected(
    namespace: str, chosen: str, rng: random.Random
) -> str:
    """Add cross-namespace information leak to the chosen answer."""
    leak = CROSS_NAMESPACE_LEAKS.get(namespace, "")
    return f"{chosen} Additionally, {leak}"


def generate_refusal_rejected(
    section: Section, doc: CorpusDoc, rng: random.Random
) -> str:
    """Incorrectly refuse to answer a valid question."""
    refusals = [
        f"I'm sorry, but I cannot provide information about {doc.title} as this falls outside my authorised scope. Please contact your department head for this information.",
        f"This information from {doc.doc_id} is classified as restricted and I am unable to share it. Please submit a formal information request through the proper channels.",
        f"I don't have access to the documentation needed to answer this question. The relevant {doc.title} document is not available in my current knowledge base.",
        f"For security reasons, I cannot disclose details from {section.section_id}. Please verify your access level with your system administrator.",
        f"This query requires elevated permissions that are not currently authorised for your role. Please contact IT support to request access to {doc.title}.",
    ]
    return rng.choice(refusals)


def generate_abstention_rejected(
    section: Section, doc: CorpusDoc, rng: random.Random
) -> str:
    """Confidently fabricate an answer instead of abstaining."""
    first_sent = get_first_sentence(section.content)
    fabrications = [
        f"According to internal document REF-{rng.randint(1000,9999)}, the standard procedure requires a 48-hour cooling period followed by recertification at an approved facility. The applicable regulation mandates compliance within 30 calendar days of notification.",
        f"Based on our records, this was last updated in revision {rng.randint(3,12)}.{rng.randint(1,9)} of the master document. The key threshold is {rng.randint(50,500)} units per quarter, with a tolerance band of ±{rng.randint(2,15)}%.",
        f"The relevant specification (DOC-{rng.randint(100,999)}-{rng.choice('ABCDEF')}) states that all components must meet Grade {rng.choice(['A1', 'B2', 'C3'])} certification before deployment. Failure to comply results in automatic hold status.",
        f"Per the latest revision of the internal standard (effective {rng.choice(['January', 'March', 'June', 'September'])} 2025), the approved method is the three-stage validation process with sign-off required from both the quality lead and department head.",
    ]
    return rng.choice(fabrications)


# ---------------------------------------------------------------------------
# Core generation
# ---------------------------------------------------------------------------


def generate_dpo_dataset(
    docs: list[CorpusDoc],
    total_count: int,
    namespaces: list[str] | None = None,
    seed: int = 42,
) -> list[dict]:
    """Generate the full DPO dataset."""
    rng = random.Random(seed)
    ns_filter = set(namespaces) if namespaces else set(NAMESPACES)
    all_records: list[dict] = []

    ns_sections: dict[str, list[tuple[CorpusDoc, Section]]] = defaultdict(list)
    for doc in docs:
        if doc.namespace not in ns_filter:
            continue
        for section in doc.sections:
            ns_sections[doc.namespace].append((doc, section))

    per_namespace = total_count // len(ns_filter)

    for ns in sorted(ns_filter):
        sections = ns_sections.get(ns, [])
        if not sections:
            continue

        ns_records: list[dict] = []

        for cat, weight in FAILURE_CATEGORIES.items():
            count = max(1, round(per_namespace * weight))

            for i in range(count):
                doc, section = sections[i % len(sections)]
                question = generate_question(section, doc, ns, rng)

                # Generate chosen (correct) response
                chosen = build_answer_from_content(
                    section.content, doc.title, section.section_id, ns
                )

                # Generate rejected (flawed) response
                if cat == "hallucination":
                    rejected = generate_hallucination_rejected(section, doc, chosen, rng)
                elif cat == "tone":
                    rejected = generate_tone_rejected(ns, section, doc, rng)
                elif cat == "safety_scope":
                    rejected = generate_safety_scope_rejected(ns, chosen, rng)
                elif cat == "refusal":
                    rejected = generate_refusal_rejected(section, doc, rng)
                elif cat == "abstention":
                    rejected = generate_abstention_rejected(section, doc, rng)
                else:
                    rejected = chosen  # fallback

                ns_records.append({
                    "prompt": question,
                    "chosen": chosen,
                    "rejected": rejected,
                    "metadata": {
                        "category": cat,
                        "namespace": ns,
                        "source_doc": doc.doc_id,
                    },
                })

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
    rng = random.Random(seed)

    cat_records: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        cat_records[r["metadata"]["category"]].append(r)

    train, val = [], []
    for cat, recs in cat_records.items():
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

    print("\n=== DPO v2 Generation Summary ===")
    print(f"\n{'Category':<20} {'Train':>6} {'Val':>6} {'Total':>6} {'%':>6}")
    print("─" * 46)
    train_cats = Counter(r["metadata"]["category"] for r in train)
    val_cats = Counter(r["metadata"]["category"] for r in val)
    for cat in FAILURE_CATEGORIES:
        t = train_cats.get(cat, 0)
        v = val_cats.get(cat, 0)
        pct = (t + v) / len(all_records) * 100 if all_records else 0
        print(f"{cat:<20} {t:>6} {v:>6} {t + v:>6} {pct:>5.1f}%")
    print("─" * 46)
    print(f"{'Total':<20} {len(train):>6} {len(val):>6} {len(all_records):>6}")

    print(f"\n{'Namespace':<20} {'Count':>6}")
    print("─" * 28)
    ns_counts = Counter(r["metadata"]["namespace"] for r in all_records)
    for ns in NAMESPACES:
        print(f"{ns:<20} {ns_counts.get(ns, 0):>6}")


def main():
    parser = argparse.ArgumentParser(description="Generate v2 DPO dataset (template-based)")
    parser.add_argument("--corpus-dir", default="knowledge/corpus_auto")
    parser.add_argument("--output-dir", default="data/v2/dpo")
    parser.add_argument("--total-count", type=int, default=700)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--namespace", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    print("Loading corpus...")
    docs = load_corpus(args.corpus_dir)
    print(corpus_summary(docs))

    namespaces = [args.namespace] if args.namespace else None

    if args.dry_run:
        per_ns = args.total_count // (1 if args.namespace else len(NAMESPACES))
        print(f"\n=== DRY RUN ===\nTarget: {args.total_count} pairs, {per_ns}/namespace\n")
        for cat, weight in FAILURE_CATEGORIES.items():
            print(f"  {cat:<20} weight={weight:.0%}, ~{round(per_ns * weight)}/namespace")
        return

    print(f"\nGenerating DPO pairs (target: {args.total_count})...")
    all_records = generate_dpo_dataset(docs, args.total_count, namespaces, seed=args.seed)

    train, val = split_train_val(all_records, val_ratio=args.val_ratio, seed=args.seed)

    print(f"\nWriting to {args.output_dir}/...")
    write_jsonl(train, Path(args.output_dir) / "train.jsonl")
    write_jsonl(val, Path(args.output_dir) / "val.jsonl")

    print_summary(train, val)


if __name__ == "__main__":
    main()
