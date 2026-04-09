# AlvinAI v2 — SOTA Architecture (Fine-Tuning + RAG Pipeline)

**Model:** Mistral Nemo 12B Instruct (128K context)
**Date:** 2026-04-09
**Status:** Planned

This is a **two-track upgrade** addressing both fine-tuning quality and RAG pipeline intelligence. See also:
- [FINE_TUNING_PLAN.md](FINE_TUNING_PLAN.md) — detailed training changes
- [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) — code-level changes for RAG pipeline

---

## Full v1 vs v2 Comparison

### Fine-Tuning

| Parameter | v1 | v2 (SOTA) |
|---|---|---|
| Data generation | Template-based | LLM-synthetic (Claude Sonnet) |
| SFT data | 9,398 pairs | 20,000+ pairs |
| RAFT data | 16,578 pairs | 25,000+ pairs |
| DPO data | 1,992 pairs (behavior-grouped) | 5,000+ pairs (namespace-balanced) |
| SFT epochs | 3 (overfit at 0.34) | 1 + early stopping |
| NEFTune | None | alpha=5.0 |
| LoRA rank | r=16, alpha=32 | r=32, alpha=64 |
| SFT dropout | 0.05 | 0.10 |
| Oracle-free ratio | 30% | 40% |
| RAFT negatives | Same domain, different doc | Harder: same paragraph, wrong values |
| DPO method | Offline only | Offline + iterative (self-correction) |
| DPO beta | 0.1 | 0.15 (preserve RAFT capabilities) |
| Compliance DPO pairs | 0 | 1,000 |
| Eval judge | Template keyword matching | Claude Sonnet (LLM-as-judge) |
| Eval with RAG | No | Yes — full pipeline |

### RAG Pipeline

| Component | v1 (Current) | v2 (SOTA) |
|---|---|---|
| Embeddings | MiniLM-L6 (384d, 22M params) | nomic-embed-v1.5 (768d, 137M params) |
| Chunking | Fixed-size per namespace | Semantic parent-child (retrieve child, inject parent) |
| Query Processing | Raw query direct to search | Rewrite + HyDE + sub-question decomposition |
| Dense Index | pgvector ivfflat | pgvector HNSW |
| Retrieval Depth | Fixed top-10 | Adaptive (5/15/25 by query type) |
| Reranker | MiniLM-L-6 (22M params) | MiniLM-L-12 (33M params) |
| Context Format | Flat "Source: title\ncontent" | Structured with headings, scores, CoT instructions |
| Chunks to LLM | 5 child chunks | 7-10 parent chunks (richer context) |
| LLM | Mistral 7B (32K context) | Mistral Nemo 12B (128K context) |
| Chat Templates | Mistral only | Mistral, Gemma, Phi, Qwen |
| Verification | Offline RAGAS only | Online faithfulness check (compliance: always, others: 10%) |

---

## Pipeline Flow Diagram

```
                         ALVINAI v2 — SOTA RAG PIPELINE
                         Mistral Nemo 12B, 128K Context

                         +------------------+
                         |   User Query     |
                         +--------+---------+
                                  |
                                  v
                    +---------------------------+
                    |    QUERY CLASSIFIER        |
                    |    (regex + heuristic)      |
                    +---+-------+-------+--------+
                        |       |       |
             greeting   |  search       |  general
                        v       |       v
               canned reply     |    LLM only (no docs)
                                |
============= NEW =============|==========================================
                                v
                    +---------------------------+
                    |    QUERY PROCESSOR         |
                    |                            |
                    |  1. REWRITE (LLM, 60 tok)  |
                    |     "brakes feel weird"     |
                    |     -> "brake system        |
                    |        malfunction          |
                    |        diagnosis"           |
                    |                            |
                    |  2. HyDE (LLM, 150 tok)    |
                    |     Generate hypothetical   |
                    |     answer paragraph ->     |
                    |     embed THAT for dense    |
                    |     search                  |
                    |                            |
                    |  3. DECOMPOSE (if compare)  |
                    |     "Compare 2022 vs 2023   |
                    |      brakes" ->             |
                    |     ["2022 brake specs",    |
                    |      "2023 brake specs"]    |
                    |                            |
                    |  All results cached (Redis) |
                    +-------------+--------------+
                                  |
============= UPGRADED ===========|========================================
                                  v
                    +---------------------------+
                    |    HYBRID RETRIEVER        |
                    |                            |
                    |  ADAPTIVE DEPTH:           |
                    |  +---------------------+   |
                    |  | simple   ->  5      |   |
                    |  | standard -> 15      |   |
                    |  | complex  -> 25      |   |
                    |  | compliance -> 20    |   |
                    |  +---------------------+   |
                    |                            |
                    |  +----------+ +----------+ |
                    |  |  DENSE   | |  SPARSE  | |
                    |  |          | |          | |
                    |  | nomic-   | | BM25Okapi| |
                    |  | embed    | | (in-mem) | |
                    |  | v1.5     | |          | |
                    |  | 768-dim  | | keyword  | |
                    |  | HNSW idx | | scoring  | |
                    |  | task-    | |          | |
                    |  | prefixed | |          | |
                    |  +----+-----+ +----+-----+ |
                    |       +------+-----+       |
                    |              v              |
                    |     +------------+         |
                    |     | RRF FUSION |         |
                    |     | k=60       |         |
                    |     +------+-----+         |
                    |            v               |
                    |     +-------------+        |
                    |     | THRESHOLD   |        |
                    |     | compliance  |        |
                    |     |   >= 0.80   |        |
                    |     | others      |        |
                    |     |   >= 0.72   |        |
                    |     +------+------+        |
                    |            v               |
                    |  +--------------------+    |
                    |  | PARENT RESOLUTION  |    |
                    |  | Retrieve CHILDREN  |    |
                    |  | -> resolve PARENTS |    |
                    |  | -> deduplicate     |    |
                    |  | Child: precise     |    |
                    |  |   match            |    |
                    |  | Parent: full       |    |
                    |  |   section context  |    |
                    |  +--------------------+    |
                    +-------------+--------------+
                                  | ~15 children -> ~7-10 unique parents
============= UPGRADED ===========|========================================
                                  v
                    +---------------------------+
                    |    RERANKER                |
                    |    MiniLM-L-12-v2          |
                    |    (33M params, CPU)       |
                    |                            |
                    |    Score each (query,      |
                    |    child.content) pair      |
                    |    top-15 -> top-7          |
                    +-------------+--------------+
                                  | top-7 reranked
============= UPGRADED ===========|========================================
                                  v
                    +-------------------------------+
                    |    PROMPT ASSEMBLY             |
                    |                                |
                    |  +---------------------------+ |
                    |  | SYSTEM PROMPT (per dept)   | |
                    |  | engineering -> precise     | |
                    |  | compliance -> legally      | |
                    |  |   strict                   | |
                    |  | customer -> friendly       | |
                    |  +---------------------------+ |
                    |                                |
                    |  +---------------------------+ |
                    |  | STRUCTURED CONTEXT         | |
                    |  |                            | |
                    |  | ### Doc 1: Service Manual  | |
                    |  | (Section 4.2 Brake System) | |
                    |  | Relevance: 0.94            | |
                    |  | ---                        | |
                    |  | {PARENT chunk content}     | |
                    |  |                            | |
                    |  | ### Doc 2: TSB-2024-015    | |
                    |  | Relevance: 0.91            | |
                    |  | ---                        | |
                    |  | {PARENT chunk content}     | |
                    |  |                            | |
                    |  | (7-10 parent chunks)       | |
                    |  +---------------------------+ |
                    |                                |
                    |  +---------------------------+ |
                    |  | CoT INSTRUCTIONS           | |
                    |  | 1. Analyze documents       | |
                    |  | 2. Cite by name            | |
                    |  | 3. Note conflicts          | |
                    |  | 4. State what's missing    | |
                    |  +---------------------------+ |
                    +---------------+----------------+
                                    |
============= HANDOFF ==============|======================================
            Hetzner CPU ------------+----------- RunPod/Azure GPU
                                    |
                                    v
                    +-------------------------------+
                    |    LLM GENERATION              |
                    |                                |
                    |  Mistral Nemo 12B (AWQ 4-bit)  |
                    |  128K context window            |
                    |                                |
                    |  Multi-template router:        |
                    |  +---------+--------+          |
                    |  | mistral | [INST]  |          |
                    |  | gemma   | <turn>  |          |
                    |  | phi     | <|user|>|          |
                    |  | qwen    | <|im_s|>|          |
                    |  +---------+--------+          |
                    |                                |
                    |  vLLM serving (blue/green)      |
                    +---------------+----------------+
                                    |
============= NEW ==================|======================================
                                    v
                    +-------------------------------+
                    |    FAITHFULNESS VERIFIER       |
                    |                                |
                    |  compliance: ALWAYS verify     |
                    |  others: 10% sample            |
                    |                                |
                    |  +------------------------+    |
                    |  | LLM checks answer vs   |    |
                    |  | source documents        |    |
                    |  |                         |    |
                    |  | FAITHFUL -> return      |    |
                    |  | UNFAITHFUL + compliance |    |
                    |  |   -> regenerate stricter|    |
                    |  | UNFAITHFUL + other      |    |
                    |  |   -> log warning, return|    |
                    |  +------------------------+    |
                    +---------------+----------------+
                                    |
                                    v
                    +-------------------------------+
                    |    RESPONSE                    |
                    |    {answer, sources, latency,  |
                    |     query_type, namespace}      |
                    +-------------------------------+
```

---

## Component Details

### 1. Query Classifier (unchanged)

Rule-based regex classification. No LLM cost.

| Route | Trigger | Action |
|---|---|---|
| Greeting | 30+ regex patterns (hi, thanks, bye...) | Canned reply, no LLM |
| Factual Lookup | 7 regexes (torque, price, part number...) | Direct DB lookup, no LLM |
| Document Search | 7 regexes (TSB, regulation, FMVSS...) | Full RAG pipeline |
| General | Fallback (? or >5 words) | LLM only, no retrieval |

### 2. Query Processor (NEW)

Three-stage intelligence layer before retrieval. All results cached in Redis.

**Query Rewriting:**
- LLM rewrites vague/colloquial queries into precise search terms
- Cost: ~60 tokens per query, cached
- Example: "brakes feel weird" -> "brake system malfunction diagnosis troubleshooting"
- Skips short specific queries (<=4 words)

**HyDE (Hypothetical Document Embedding):**
- LLM generates a hypothetical ideal answer paragraph (~150 tokens)
- That paragraph is embedded instead of the raw query
- Bridges the query-document asymmetry gap (short query vs long document)
- The hypothetical doc is closer in embedding space to real documents

**Sub-question Decomposition:**
- Only triggers for comparison queries (keywords: compare, difference, versus, both)
- Splits into 2-3 sub-questions, retrieves for each, merges results
- Example: "Compare 2022 vs 2023 brake specs" -> ["2022 brake specifications", "2023 brake specifications"]

### 3. Hybrid Retriever (UPGRADED)

**Embedding Model: nomic-embed-text-v1.5**
- 768-dim vectors (vs 384-dim MiniLM-L6)
- 137M params, runs on CPU
- 8192 token input (vs 256 for MiniLM) — handles larger chunks
- Task-prefixed: `search_query:` for queries, `search_document:` for documents
- MTEB retrieval score: ~53 vs MiniLM's ~41 (27% improvement)

**pgvector HNSW Index (replaces ivfflat):**
- More accurate approximate nearest neighbor search
- No `lists` parameter to tune as corpus grows
- Better recall at same query speed
- Parameters: m=16, ef_construction=64

**Adaptive Retrieval Depth:**
- Simple factual -> 5 children
- Standard document search -> 15 children
- Complex/comparison -> 25 children (per sub-question)
- Compliance -> 20 children (high recall needed)

**Parent-Child Architecture:**
```
Document
  +-- Parent Chunk (1500-2000 tokens)
  |     Full semantic section for LLM context
  |     Stored in parent_chunks table (no embedding)
  |     +-- Child Chunk (300-500 tokens)
  |     |     Precise segment for embedding retrieval
  |     |     Stored in document_chunks table (with embedding)
  |     +-- Child Chunk
  |     +-- Child Chunk
  +-- Parent Chunk
        +-- Child Chunk
        +-- Child Chunk
```

- **Retrieve** by child (precise semantic match on small chunks)
- **Inject** parent into prompt (full section context for LLM)
- Deduplicate parents when multiple children from same section match

### 4. Reranker (UPGRADED)

- Model: `cross-encoder/ms-marco-MiniLM-L-12-v2` (33M params, up from 22M L-6)
- 50% more accurate than L-6 variant
- Still runs on CPU in ~50ms for 15 pairs
- Drop-in replacement (same CrossEncoder API)
- Reranks top-15 -> top-7 (increased from top-5 to leverage 128K context)

### 5. Prompt Assembly (UPGRADED)

**Structured context format:**
```
## Retrieved Documents

### Document 1: Service Manual - Brake System (Section 4.2)
Relevance: 0.94
---
{full parent chunk content — 1500-2000 tokens of rich context}

### Document 2: TSB-2024-015 - Brake Pad Recall
Relevance: 0.91
---
{full parent chunk content}
```

**Chain-of-Thought instructions:**
```
1. Analyze the retrieved documents above to answer the question.
2. Cite specific documents by name when referencing information.
3. If documents contain conflicting information, note the discrepancy.
4. If the answer is not fully covered by the documents, state what is missing.
5. Do not reproduce the reference material verbatim — synthesize and explain.
```

**Why this matters for 12B+ models:**
- Structured headers help the model prioritize across 7-10 documents
- Relevance scores signal which documents to weight more heavily
- CoT instructions activate multi-step reasoning (12B models are capable enough)
- Document numbering enables precise citation in the answer

### 6. LLM Generation (UPGRADED)

**Model:** Mistral Nemo 12B Instruct (AWQ 4-bit via vLLM)
- 128K context window (vs 32K on 7B)
- Comfortably handles 7-10 parent chunks + system prompt + query
- Same Mistral instruct format: `<s>[INST]...[/INST]`
- Blue/green deployment via vLLM

**Multi-template router** for benchmarking other 12B-14B models:
| Template | Format | Models |
|---|---|---|
| mistral | `<s>[INST] {prompt} [/INST]` | Nemo 12B, Small 24B |
| gemma | `<start_of_turn>user\n{prompt}<end_of_turn>` | Gemma 3 12B |
| phi | `<\|user\|>\n{prompt}<\|end\|>` | Phi-4 14B |
| qwen | `<\|im_start\|>user\n{prompt}<\|im_end\|>` | Qwen3 14B |

### 7. Faithfulness Verifier (NEW)

Post-generation check that catches hallucinations before they reach the user.

**Compliance namespace:** Always verify (legal requirement)
- If UNFAITHFUL: regenerate with stricter prompt emphasizing "only state facts from documents"
- Adds ~300ms latency (one extra LLM call)

**Other namespaces:** 10% random sample (monitoring)
- If UNFAITHFUL: log warning, return original answer
- No added latency for 90% of queries

---

## Infrastructure (unchanged)

```
+---------------------------------------------------+
|               HETZNER SERVER (CPU)                  |
|                                                     |
|   FastAPI (orchestrator)                            |
|   +- nomic-embed-v1.5 (768d, in-process, ~300MB)   |
|   +- MiniLM-L-12 reranker (in-process, ~130MB)     |
|   +- Query processor (calls LLM for rewrite/HyDE)  |
|   +- Faithfulness verifier (calls LLM)              |
|                                                     |
|   PostgreSQL + pgvector (HNSW index)                |
|   Redis (cache: queries, rewrites, HyDE, results)  |
+---------------------------------------------------+
                        |
                        | HTTP (assembled prompt)
                        v
+---------------------------------------------------+
|               RUNPOD / AZURE (GPU)                  |
|                                                     |
|   vLLM serving Mistral Nemo 12B (AWQ 4-bit)        |
|   Blue/green deployment                             |
+---------------------------------------------------+
```

---

## Namespace Access Control (unchanged)

| Department | Can Access |
|---|---|
| Customer Support | Customer support docs |
| Dealers | Customer support, sales docs |
| Engineers | Customer support, engineering docs |
| Compliance/Legal | Compliance, engineering docs (read-only) |
| HR | Employee HR policies |
| Procurement | Vendor contracts, SLAs |
| Admin | Everything |

Enforced at retrieval layer. Compliance namespace: stricter threshold (0.80), mandatory audit logging, mandatory faithfulness verification.

---

## Performance Targets

| Query Type | v1 Target | v2 Target | Notes |
|---|---|---|---|
| Factual lookup | < 500ms | < 500ms | Unchanged (no LLM) |
| Doc search (cached) | < 200ms | < 250ms | Slight increase from parent resolution |
| Doc search (cold) | < 8s | < 10s | Query rewriting + HyDE add ~500ms |
| Doc search (cold + verify) | N/A | < 12s | Compliance only |

## RAGAS Targets

| Metric | v1 Target | v2 Target | Expected Gain |
|---|---|---|---|
| Faithfulness | >= 0.88 | >= 0.92 | Parent chunks + CoT + verification |
| Answer Relevancy | >= 0.85 | >= 0.90 | Query rewriting + better embeddings |
| Context Precision | >= 0.80 | >= 0.88 | Parent-child + upgraded reranker |
| Context Recall | >= 0.82 | >= 0.88 | nomic embeddings + HNSW + adaptive depth |
