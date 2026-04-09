# AlvinAI v2 — Complete SOTA Architecture and Implementation Plan

**Model:** Mistral Nemo 12B Instruct (12.3B params, 128K context)
**Experiment:** `mistral-nemo-12b-v2`
**Date:** 2026-04-09
**Status:** Planned

---

## Overview

v2 is a **two-track upgrade** — SOTA fine-tuning and SOTA RAG pipeline — designed to work together. v1 showed that fine-tuning alone on weak data with a basic RAG pipeline produces no net gain over the base model. v2 fixes both sides simultaneously.

```
+------------------------------------------------------+
|                    AlvinAI v2                          |
|                                                       |
|  Track 1: FINE-TUNING          Track 2: RAG PIPELINE  |
|  LLM-synthetic data            nomic-embed 768d       |
|  NEFTune + early stopping      Parent-child chunking  |
|  Harder RAFT negatives         Query rewriting + HyDE |
|  Namespace-balanced DPO        HNSW index             |
|  LLM judge (Claude)            Faithfulness verifier  |
|                                                       |
|  Both tracks feed into:                               |
|  +--------------------------------------------------+ |
|  | Production: fine-tuned Nemo 12B + enhanced RAG    | |
|  +--------------------------------------------------+ |
+------------------------------------------------------+
```

---

## 1. Full v1 vs v2 Comparison

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
| Verification | Offline RAGAS only | Online faithfulness check |

---

## 2. Fine-Tuning Architecture

### 2.1 Why v1 Fine-Tuning Underperformed

| Problem | Evidence | Root Cause |
|---|---|---|
| SFT overfit at epoch 0.34 | eval_loss 1.02 → 1.28+ after step 200 | Template-based data, only 9.4K examples, low diversity |
| DPO net 0% win rate | 12 wins / 12 losses / 26 ties | Only 1,992 pairs, no namespace balancing |
| Compliance regressed -27% | DPO had zero compliance pairs | Data organized by behavior type, not namespace |
| Base Nemo = fine-tuned with RAG | 5/5 on both | Corpus is public Wikipedia — already in pre-training |
| RAGAS all below target | Faithfulness 0.52, Relevancy 0.29 | Evaluated without RAG, template-based scoring |

### 2.2 Training Pipeline Flow

```
                    DATA GENERATION (Claude Sonnet)
                    +----------------------------------+
                    | LLM-synthetic, 6 namespaces      |
                    | 20K SFT + 25K RAFT + 5K DPO      |
                    | Diverse phrasing, hard negatives  |
                    +----------------+-----------------+
                                     |
            +------------------------+------------------------+
            |                        |                        |
            v                        v                        v
    +---------------+       +----------------+       +---------------+
    |   SFT Data    |       |   RAFT Data    |       |   DPO Data    |
    | 20K pairs     |       | 25K examples   |       | 5K pairs      |
    | OpenAI msgs   |       | oracle+distract|       | per namespace |
    | LLM-generated |       | 40% oracle-free|       | LLM-judged    |
    +-------+-------+       +-------+--------+       +-------+-------+
            |                        |                        |
            v                        v                        v
    ================        ================        ================
    ||   STAGE 1   ||       ||   STAGE 2   ||       ||   STAGE 3   ||
    ||    SFT      ||       ||    RAFT     ||       ||    DPO      ||
    ================        ================        ================
    |                       |                       |
    | 1 epoch               | 2 epochs              | 1 epoch
    | LR: 1e-4              | LR: 2e-5              | LR: 5e-6
    | NEFTune alpha=5       | Curriculum learning   | Beta: 0.15
    | Early stopping        | Harder negatives      | Namespace-balanced
    | LoRA r=32, alpha=64   | 40% oracle-free       | LLM judge (Claude)
    | Dropout: 0.10         |                       |
    |                       |                       |
    v                       v                       v
    +---------------+       +----------------+       +---------------+
    | SFT Checkpoint|------>| RAFT Checkpoint|------>| DPO Checkpoint|
    +-------+-------+       +-------+--------+       +-------+-------+
            |                        |                        |
            v                        v                        v
    +---------------+       +----------------+       +---------------+
    | Quality Gate  |       | Quality Gate   |       | Quality Gate  |
    |               |       |                |       |               |
    | eval_loss     |       | Citation >=0.85|       | Win rate >=15%|
    | not increasing|       | Rejection>=0.85|       | per namespace |
    | (early stop)  |       | Abstention>=0.85|     | LLM judge     |
    +---------------+       +----------------+       +---------------+
                                                              |
                                                              v
                                                     +----------------+
                                                     | AWQ Quantize   |
                                                     | 4-bit for vLLM |
                                                     +--------+-------+
                                                              |
                                                              v
                                                     +----------------+
                                                     | RAGAS Eval     |
                                                     | WITH full RAG  |
                                                     | LLM judge      |
                                                     +----------------+
```

### 2.3 Stage 1: SFT (Supervised Fine-Tuning)

```
v1: 3 epochs, 9.4K template data, LR 1.5e-4, dropout 0.05, LoRA r=16
    Result: Overfit at epoch 0.34, eval_loss 1.02 → 1.28+

v2: 1 epoch, 20K+ LLM-synthetic data, LR 1e-4, dropout 0.10,
    LoRA r=32, NEFTune alpha=5, early stopping
```

**SOTA techniques applied:**

| Technique | What | Why |
|---|---|---|
| NEFTune (alpha=5.0) | Adds uniform noise to embedding vectors during training | Proven 5-15% downstream improvement. Zero compute cost — just a config flag in TRL's SFTTrainer |
| Early stopping (patience=3) | Stops training when eval_loss stops improving | v1 wasted 80% of SFT compute training past the optimum |
| LoRA r=32, alpha=64 | Higher rank adaptation (114M params, 0.93% of 12.3B) | 12B model has capacity for higher rank. v1's r=16 was underparameterized |
| Dropout 0.10 | Doubled from 0.05 | More regularization to prevent co-adaptation |
| LR 1e-4 | Reduced from 1.5e-4 | Slower learning = better generalization |

**v2 SFT config:**
```yaml
model:
  base_model: "mistralai/Mistral-Nemo-Instruct-2407"
  max_seq_length: 4096
  load_in_4bit: true
  dtype: "bfloat16"

lora:
  r: 32                          # was: 16
  alpha: 64                      # was: 32
  dropout: 0.10                  # was: 0.05
  target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]

data:
  train_file: "data/v3/sft/train.jsonl"
  val_file: "data/v3/sft/val.jsonl"
  format: "openai_messages"
  chat_template: "mistral"

training:
  num_epochs: 1                    # was: 3
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 16
  learning_rate: 1.0e-4            # was: 1.5e-4
  lr_scheduler_type: "cosine"
  warmup_ratio: 0.05
  weight_decay: 0.01
  max_grad_norm: 1.0
  bf16: true
  seed: 42
  neftune_noise_alpha: 5.0         # NEW
  early_stopping_patience: 3       # NEW
```

### 2.4 Stage 2: RAFT (Retrieval-Augmented Fine-Tuning)

```
v1: 1 epoch, 16.5K data, LR 3e-5, 30% oracle-free
    Result: Excellent (eval_loss 0.414→0.113), citation 100%, abstention 94.6%

v2: 2 epochs, 25K+ data, LR 2e-5, 40% oracle-free,
    harder negatives, curriculum learning
```

RAFT was the strongest stage in v1. v2 pushes it further.

**Harder Negatives (NEW):**

| Difficulty | v1 | v2 |
|---|---|---|
| Easy | Random document from same namespace | Same |
| Medium | Same vehicle model, different system | Same |
| Hard (NEW) | Same paragraph topic, different spec values | NEW |
| Adversarial (NEW) | Paraphrased oracle with wrong numbers | NEW |

Adversarial negatives teach the model to verify specific numbers rather than pattern-matching.

**Curriculum Learning (NEW):**
```
Epoch 1: easy negatives first → medium → hard
Epoch 2: full mix (random order)
```

**40% Oracle-Free Ratio** (up from 30%): 12B models are more prone to confident hallucination.

**v2 RAFT config:**
```yaml
training:
  num_epochs: 2                     # was: 1
  learning_rate: 2.0e-5             # was: 3.0e-5

raft_thresholds:
  oracle_citation_rate: 0.85        # was: 0.80 (raised)
  distractor_rejection_rate: 0.85   # was: 0.80
  abstention_rate: 0.85             # was: 0.80

data:
  oracle_free_ratio: 0.40           # was: 0.30
  negative_difficulty: "curriculum"  # NEW
```

### 2.5 Stage 3: DPO (Direct Preference Optimization)

```
v1: 1 epoch, 1,992 pairs, LR 1e-5, beta 0.1, template judge
    Result: 0% net win rate, compliance -27% regression

v2: 1 epoch, 5,000+ pairs, LR 5e-6, beta 0.15,
    namespace-balanced, LLM judge (Claude), iterative
```

**Namespace-Balanced Data (NEW):**

| Namespace | v1 Pairs | v2 Pairs |
|---|---|---|
| customer_support | ~400 (mixed) | 800 |
| engineering | ~400 (mixed) | 800 |
| dealer_sales | ~400 (mixed) | 700 |
| compliance | 0 | 1000 (prioritized) |
| employee_hr | ~400 (mixed) | 700 |
| vendor | ~400 (mixed) | 700 |
| **Total** | **1,992** | **4,700+** |

**LLM-as-Judge (NEW):** Claude Sonnet evaluates on factual accuracy, citation correctness, appropriate uncertainty, conciseness, namespace-appropriate tone.

**Iterative DPO (NEW):**
```
Round 1: Train on 5K external pairs (1 epoch)
         |
         v
Round 2: Generate 1K responses from Round 1 model
         Judge with Claude → new preference pairs
         Train on those (0.5 epoch, lower LR 2e-6)
```

**Conservative Hyperparameters:**

| Parameter | v1 | v2 | Why |
|---|---|---|---|
| Learning rate | 1e-5 | 5e-6 | Prevent catastrophic forgetting of RAFT capabilities |
| Beta | 0.1 | 0.15 | Stronger KL penalty keeps model closer to RAFT reference |
| Max seq length | 1024 | 2048 | Allow longer context in preference pairs |

### 2.6 Data Generation: Template → LLM-Synthetic

**Why templates failed:**
1. Repetitive patterns: "What is the [SPEC] for [VEHICLE]?" × 20 variations
2. Limited diversity: 926 sections × ~10 templates = ~10K with high overlap
3. No paraphrasing: Real users say "brakes feel weird" not "What is the brake diagnostic procedure?"
4. Public content: Wikipedia-sourced corpus already in Nemo's pre-training

**v2 uses Claude Sonnet** to generate diverse, natural training data:
```
SFT: Source chunk + namespace + role → natural question + grounded answer
RAFT: Oracle + distractors → question + answer citing only oracle
DPO: Question + context → chosen (accurate) + rejected (subtle flaw)
```

Budget: ~$50-100 for 50K+ examples.

**Data Quality Gates (NEW):**

| Check | Threshold | Action |
|---|---|---|
| SFT answer grounded in source | 100% | Reject ungrounded |
| RAFT oracle citation present | 95%+ | Reject missing citations |
| RAFT distractor content absent | 95%+ | Reject contaminated answers |
| DPO chosen > rejected (Claude verify) | 90%+ | Re-judge disagreements |
| Namespace balance | ±10% | Rebalance if skewed |
| Duplicate detection | <5% similar | Deduplicate |

---

## 3. RAG Pipeline Architecture

### 3.1 Pipeline Flow Diagram

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
                    |  +--------------------+    |
                    +-------------+--------------+
                                  | ~15 children -> ~7-10 unique parents
============= UPGRADED ===========|========================================
                                  v
                    +---------------------------+
                    |    RERANKER                |
                    |    MiniLM-L-12-v2          |
                    |    (33M params, CPU)       |
                    |    top-15 -> top-7          |
                    +-------------+--------------+
                                  | top-7 reranked
============= UPGRADED ===========|========================================
                                  v
                    +-------------------------------+
                    |    PROMPT ASSEMBLY             |
                    |                                |
                    |  System prompt (per namespace) |
                    |  + Structured context          |
                    |    (headings, scores, parent   |
                    |     chunks, 7-10 documents)    |
                    |  + CoT instructions            |
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
                    |  Multi-template router          |
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
                    |  FAITHFUL -> return            |
                    |  UNFAITHFUL + compliance       |
                    |    -> regenerate stricter      |
                    |  UNFAITHFUL + other            |
                    |    -> log warning, return      |
                    +---------------+----------------+
                                    |
                                    v
                    +-------------------------------+
                    |    RESPONSE                    |
                    +-------------------------------+
```

### 3.2 Query Classifier (unchanged)

Rule-based regex classification. No LLM cost.

| Route | Trigger | Action |
|---|---|---|
| Greeting | 30+ regex patterns | Canned reply, no LLM |
| Factual Lookup | 7 regexes (torque, price, part number...) | Direct DB lookup, no LLM |
| Document Search | 7 regexes (TSB, regulation, FMVSS...) | Full RAG pipeline |
| General | Fallback (? or >5 words) | LLM only, no retrieval |

### 3.3 Query Processor (NEW)

Three-stage intelligence layer before retrieval. All results cached in Redis.

**Query Rewriting:** LLM rewrites vague queries into precise search terms (~60 tokens, cached). Skips short specific queries (<=4 words).

**HyDE (Hypothetical Document Embedding):** LLM generates a hypothetical ideal answer (~150 tokens). Embed that instead of the raw query — bridges query-document asymmetry.

**Sub-question Decomposition:** Triggers for comparison queries only. Splits into 2-3 sub-questions, retrieves for each, merges results.

### 3.4 Hybrid Retriever (UPGRADED)

**Embedding Model: nomic-embed-text-v1.5**
- 768-dim vectors (vs 384-dim MiniLM-L6), 137M params, CPU
- 8192 token input (vs 256 for MiniLM)
- Task-prefixed: `search_query:` / `search_document:`
- MTEB retrieval score: ~53 vs MiniLM's ~41 (27% improvement)

**pgvector HNSW Index** (replaces ivfflat): more accurate ANN, no tuning needed. m=16, ef_construction=64.

**Adaptive Retrieval Depth:** simple→5, standard→15, complex→25, compliance→20.

**Parent-Child Architecture:**
```
Document
  +-- Parent Chunk (1500-2000 tokens)
  |     Full semantic section — stored without embedding
  |     +-- Child Chunk (300-500 tokens) — stored with 768d embedding
  |     +-- Child Chunk
  +-- Parent Chunk
        +-- Child Chunk
```

Retrieve by child (precise match), inject parent into prompt (full context). Deduplicate parents when multiple children from the same section match.

### 3.5 Reranker (UPGRADED)

`cross-encoder/ms-marco-MiniLM-L-12-v2` (33M params). 50% more accurate than L-6, still ~50ms on CPU. Reranks top-15 → top-7.

### 3.6 Prompt Assembly (UPGRADED)

Structured context with metadata and CoT:
```
## Retrieved Documents

### Document 1: Service Manual - Brake System (Section 4.2)
Relevance: 0.94
---
{full parent chunk content — 1500-2000 tokens}

### Document 2: TSB-2024-015 - Brake Pad Recall
Relevance: 0.91
---
{full parent chunk content}

---
Instructions:
1. Analyze the retrieved documents to answer the question.
2. Cite specific documents by name.
3. Note conflicting information between documents.
4. State what is missing if answer is incomplete.
5. Synthesize — do not reproduce verbatim.
```

7-10 parent chunks (vs 5 child chunks in v1). Structured headers + relevance scores + CoT activate multi-step reasoning in 12B+ models.

### 3.7 LLM Generation (UPGRADED)

Mistral Nemo 12B (AWQ 4-bit via vLLM), 128K context. Blue/green deployment.

Multi-template router for benchmarking:

| Template | Format | Models |
|---|---|---|
| mistral | `<s>[INST] {prompt} [/INST]` | Nemo 12B, Small 24B |
| gemma | `<start_of_turn>user\n{prompt}<end_of_turn>` | Gemma 3 12B |
| phi | `<\|user\|>\n{prompt}<\|end\|>` | Phi-4 14B |
| qwen | `<\|im_start\|>user\n{prompt}<\|im_end\|>` | Qwen3 14B |

### 3.8 Faithfulness Verifier (NEW)

- **Compliance:** Always verify. UNFAITHFUL → regenerate with stricter prompt (+~300ms).
- **Others:** 10% sample. UNFAITHFUL → log warning, return original.

---

## 4. Evaluation Architecture

### 4.1 v2 Evaluation Upgrades

| Component | v1 | v2 |
|---|---|---|
| Judge | Template (keyword overlap) | Claude Sonnet |
| RAG in eval | No | Yes — full pipeline |
| Questions per namespace | 20 | 30 |
| DPO win rate eval | Template judge | Claude side-by-side |
| RAGAS judge | Template | Claude / OpenAI |
| Eval checkpoints | Final only | Every stage (SFT, RAFT, DPO) |
| Base model comparison | None | Always compare fine-tuned vs base + RAG |

### 4.2 Stage-by-Stage Quality Gates

```
After SFT:
  - eval_loss not increasing (early stopping)
  - Perplexity on held-out automotive text < base model
  - Qualitative spot-check: 10 questions per namespace

After RAFT:
  - Oracle citation rate >= 0.85
  - Distractor rejection rate >= 0.85
  - Abstention rate >= 0.85
  - NEW: Test WITHOUT RAG — model should abstain, not hallucinate training data

After DPO:
  - Win rate >= 15% per namespace (not just overall)
  - NEW: No namespace regresses more than -5%
  - NEW: DPO model must beat base Nemo + RAG by >= 10%

Final (with RAG pipeline):
  - RAGAS faithfulness >= 0.92
  - RAGAS answer relevancy >= 0.90
  - RAGAS context precision >= 0.88
  - RAGAS context recall >= 0.88
  - P95 latency within targets
```

---

## 5. Infrastructure

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

### Namespace Access Control (unchanged)

| Department | Can Access |
|---|---|
| Customer Support | Customer support docs |
| Dealers | Customer support, sales docs |
| Engineers | Customer support, engineering docs |
| Compliance/Legal | Compliance, engineering docs (read-only) |
| HR | Employee HR policies |
| Procurement | Vendor contracts, SLAs |
| Admin | Everything |

---

## 6. Performance Targets

| Query Type | v1 Target | v2 Target | Notes |
|---|---|---|---|
| Factual lookup | < 500ms | < 500ms | Unchanged (no LLM) |
| Doc search (cached) | < 200ms | < 250ms | Parent resolution adds slight overhead |
| Doc search (cold) | < 8s | < 10s | Query rewriting + HyDE add ~500ms |
| Doc search (cold + verify) | N/A | < 12s | Compliance only |

### RAGAS Targets

| Metric | v1 Target | v2 Target | Expected Gain From |
|---|---|---|---|
| Faithfulness | >= 0.88 | >= 0.92 | Parent chunks + CoT + verification + better fine-tuning |
| Answer Relevancy | >= 0.85 | >= 0.90 | Query rewriting + better embeddings + LLM-synthetic data |
| Context Precision | >= 0.80 | >= 0.88 | Parent-child chunking + upgraded reranker + HNSW |
| Context Recall | >= 0.82 | >= 0.88 | nomic embeddings + adaptive depth + harder RAFT negatives |

---

## 7. Timeline & Cost

```
Phase 0: Data Generation (1-2 days, $50-100 API credits)
  - v3 SFT data (20K+), RAFT data (25K+), DPO data (5K+)
  - Data quality gates

Phase 1: SFT Training (~2 hours on A40, ~$1.60)
  - 1 epoch, early stopping, NEFTune, LoRA r=32

Phase 2: RAFT Training (~6-8 hours on A40, ~$5-6)
  - 2 epochs, curriculum learning, harder negatives

Phase 3: DPO Round 1 (~30 min on A40, ~$0.40)
  - 5K pairs, namespace-balanced, Claude judge

Phase 4: DPO Round 2 — Iterative (~20 min on A40, ~$0.25)
  - Self-generated pairs from Round 1 mistakes

Phase 5: RAG Pipeline Upgrade (local development)
  - Embedding swap, parent-child chunking, query processor
  - DB migration, re-embedding

Phase 6: Evaluation (~1-2 hours)
  - AWQ quantize, full RAGAS eval with RAG
  - Compare: v2 vs v1 vs base Nemo + RAG

Total: ~$60-115 (data gen) + ~$10-15 (compute)
```

---

## 8. Summary: What Makes v2 SOTA

### Fine-Tuning

| Technique | Category | Impact |
|---|---|---|
| LLM-synthetic data | Data quality | Eliminates template repetitiveness |
| 20K+ SFT / 25K+ RAFT / 5K+ DPO | Data scale | Prevents overfitting on 12B model |
| NEFTune noise | Regularization | Proven 5-15% downstream improvement |
| Early stopping | Efficiency | Stops before overfitting, saves compute |
| LoRA r=32 | Capacity | Better adaptation for 12B model |
| Higher dropout (0.10) | Regularization | Prevents memorization |
| 40% oracle-free RAFT | Abstention | Stronger "I don't know" for 12B |
| Harder/adversarial negatives | Data quality | Model verifies values, not structure |
| Curriculum learning | Training strategy | Stable easy→hard progression |
| Namespace-balanced DPO | Data coverage | Fixes compliance regression |
| LLM judge (Claude) | Evaluation | Accurate semantic assessment |
| Iterative DPO | Training strategy | Model learns from own mistakes |
| Conservative DPO hyperparams | Stability | Preserves RAFT capabilities |

### RAG Pipeline

| Technique | Category | Impact |
|---|---|---|
| nomic-embed 768d | Embeddings | 27% MTEB improvement over MiniLM-L6 |
| Parent-child chunking | Chunking | Richer context without losing retrieval precision |
| Query rewriting + HyDE | Query intelligence | Fixes vague queries, bridges query-doc gap |
| HNSW index | Retrieval | Better recall, no tuning as corpus scales |
| Adaptive depth | Retrieval | Right context volume per query complexity |
| MiniLM-L-12 reranker | Reranking | 50% more accurate than L-6 |
| Structured prompt + CoT | Prompt engineering | Better reasoning from 12B+ models |
| Faithfulness verifier | Safety | Catches hallucinations before response |
| Multi-template router | Flexibility | Benchmark Gemma/Phi/Qwen alongside Mistral |
