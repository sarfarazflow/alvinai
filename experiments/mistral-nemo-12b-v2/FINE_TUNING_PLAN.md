# AlvinAI v2 — SOTA Fine-Tuning Plan

**Model:** Mistral Nemo 12B Instruct (128K context)
**Date:** 2026-04-09
**Status:** Planned

---

## Why v1 Fine-Tuning Underperformed

| Problem | Evidence | Root Cause |
|---|---|---|
| SFT overfit at epoch 0.34 | eval_loss 1.02 → 1.28+ after step 200 | Template-based data, only 9.4K examples, low diversity |
| DPO net 0% win rate | 12 wins / 12 losses / 26 ties | Only 1,992 pairs, no namespace balancing |
| Compliance regressed -27% | DPO had zero compliance pairs | Data organized by behavior type, not namespace |
| Base Nemo = fine-tuned with RAG | 5/5 on both | Corpus is public Wikipedia — already in pre-training |
| RAGAS all below target | Faithfulness 0.52, Relevancy 0.29 | Evaluated without RAG, template-based scoring |

---

## v1 vs v2 Fine-Tuning Comparison

| Parameter | v1 (Current) | v2 (SOTA) | Why |
|---|---|---|---|
| **Data generation** | Template-based | LLM-synthetic (Claude Sonnet) | Templates produce repetitive patterns; LLM generates diverse, natural phrasing |
| **SFT data size** | 9,398 | 20,000+ | 12B models need more data to generalize without overfitting |
| **RAFT data size** | 16,578 | 25,000+ | More diverse distractors, harder negatives |
| **DPO data size** | 1,992 | 5,000+ | Need per-namespace coverage, especially compliance |
| **SFT epochs** | 3 (overfit at 0.34) | 1 + early stopping | Prevent overfitting, save compute |
| **SFT LR** | 1.5e-4 | 1.0e-4 | Slower learning = better generalization |
| **SFT dropout** | 0.05 | 0.10 | More regularization for small dataset |
| **NEFTune** | None | alpha=5.0 | Adds noise to embeddings during training, proven 5-15% improvement |
| **LoRA rank** | r=16, alpha=32 | r=32, alpha=64 | 12B model has capacity for higher rank LoRA |
| **Oracle-free ratio** | 30% | 40% | 12B models need stronger abstention training |
| **DPO balancing** | By behavior type | By namespace | Ensures every namespace gets preference training |
| **DPO judge** | Template keyword matching | Claude Sonnet (LLM-as-judge) | Template judge misses semantic quality |
| **DPO LR** | 1.0e-5 | 5.0e-6 | More conservative to prevent catastrophic forgetting |
| **DPO beta** | 0.1 | 0.15 | Stronger KL penalty preserves RAFT capabilities |
| **Eval judge** | Template-based | Claude Sonnet | Accurate semantic evaluation |
| **Eval pipeline** | Standalone (no RAG) | Full RAG pipeline | Tests the real production path |

---

## SOTA Training Pipeline: SFT → RAFT → DPO

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
    | eval_loss     |       | Citation >=0.80|       | Win rate >=15%|
    | not increasing|       | Rejection>=0.80|       | per namespace |
    | (early stop)  |       | Abstention>=0.80|      | LLM judge     |
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

---

## Stage 1: SFT (Supervised Fine-Tuning)

### What Changes from v1

```
v1: 3 epochs, 9.4K template data, LR 1.5e-4, dropout 0.05, LoRA r=16
    Result: Overfit at epoch 0.34, eval_loss 1.02 → 1.28+

v2: 1 epoch, 20K+ LLM-synthetic data, LR 1e-4, dropout 0.10,
    LoRA r=32, NEFTune alpha=5, early stopping
```

### NEFTune (NEW)

NEFTune adds uniform noise to embedding vectors during forward pass of training. Paper shows 5-15% downstream improvement. Proven technique, supported natively in TRL's SFTTrainer.

```python
# In SFTTrainer config:
neftune_noise_alpha=5.0
```

Cost: zero additional compute. Just a config flag.

### Early Stopping (NEW)

v1 trained for 3 epochs when the best checkpoint was at epoch 0.34. That wasted 80% of SFT compute.

```python
# In TrainingArguments:
load_best_model_at_end=True
metric_for_best_model="eval_loss"

# Add EarlyStoppingCallback:
from transformers import EarlyStoppingCallback
callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
```

### Higher LoRA Rank (r=16 → r=32)

12B models have more capacity. With 20K+ diverse training data, the model can productively use a higher-rank adaptation. r=32 adds ~114M trainable params (0.93% of 12.3B) — still well within QLoRA VRAM budget on A40 48GB.

| LoRA Config | Trainable Params | % of 12.3B |
|---|---|---|
| v1: r=16, alpha=32 | 57M | 0.46% |
| v2: r=32, alpha=64 | ~114M | 0.93% |

### v2 SFT Config Changes

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
  train_file: "data/v3/sft/train.jsonl"    # LLM-synthetic, 20K+
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

---

## Stage 2: RAFT (Retrieval-Augmented Fine-Tuning)

### What Changes from v1

```
v1: 1 epoch, 16.5K data, LR 3e-5, 30% oracle-free
    Result: Excellent (eval_loss 0.414→0.113), citation 100%, abstention 94.6%

v2: 2 epochs, 25K+ data, LR 2e-5, 40% oracle-free,
    harder negatives, curriculum learning
```

RAFT was the strongest stage in v1 — no overfitting, continuous improvement. The upgrades focus on making it even better.

### Harder Negatives (NEW)

v1 distractors were "same domain, different document." v2 adds a harder tier:

| Difficulty | v1 | v2 |
|---|---|---|
| Easy | Random document from same namespace | Same |
| Medium | Same vehicle model, different system | Same |
| Hard (NEW) | Same paragraph topic, different spec values | NEW |
| Adversarial (NEW) | Paraphrased oracle with wrong numbers | NEW |

The adversarial negatives (correct structure, wrong values) teach the model to verify specific numbers rather than pattern-matching document structure.

### Curriculum Learning (NEW)

Order training examples from easy to hard within each epoch:

```
Epoch 1: easy negatives first → medium → hard
Epoch 2: full mix (random order)
```

This gives the model a foundation before introducing adversarial examples.

### 40% Oracle-Free Ratio

v1 used 30%. For 12B models, increase to 40% — larger models are more prone to confident hallucination. More oracle-free examples = stronger abstention training.

### v2 RAFT Config Changes

```yaml
training:
  num_epochs: 2                     # was: 1
  learning_rate: 2.0e-5             # was: 3.0e-5
  # ... rest same

raft_thresholds:
  oracle_citation_rate: 0.85        # was: 0.80 (raise the bar)
  distractor_rejection_rate: 0.85   # was: 0.80
  abstention_rate: 0.85             # was: 0.80

data:
  oracle_free_ratio: 0.40           # was: 0.30
  negative_difficulty: "curriculum"  # NEW
```

---

## Stage 3: DPO (Direct Preference Optimization)

### What Changes from v1

```
v1: 1 epoch, 1,992 pairs, LR 1e-5, beta 0.1, template judge
    Result: 0% net win rate, compliance -27% regression

v2: 1 epoch, 5,000+ pairs, LR 5e-6, beta 0.15,
    namespace-balanced, LLM judge (Claude)
```

DPO was the weakest stage in v1. The fixes target every identified failure.

### Namespace-Balanced Data (NEW)

v1 organized DPO pairs by behavior type (abstention, hallucination, refusal, tone). This left compliance with zero dedicated pairs.

v2 ensures coverage across all namespaces:

| Namespace | v1 Pairs | v2 Pairs |
|---|---|---|
| customer_support | ~400 (mixed) | 800 |
| engineering | ~400 (mixed) | 800 |
| dealer_sales | ~400 (mixed) | 700 |
| compliance | 0 | 1000 (prioritized) |
| employee_hr | ~400 (mixed) | 700 |
| vendor | ~400 (mixed) | 700 |
| **Total** | **1,992** | **4,700+** |

Compliance gets the most pairs because:
1. v1 showed -27% regression without training data
2. Compliance answers are legally critical — wrong preferences are dangerous
3. Compliance has the strictest quality requirements

### LLM-as-Judge for DPO Data (NEW)

v1 used template-based keyword matching to select chosen/rejected. This misses semantic quality.

v2 uses Claude Sonnet as judge with the criteria from CLAUDE.md:

```
1. Factual accuracy (correct answer from documents?)
2. Citation correctness (right source, section, page?)
3. Appropriate uncertainty (says "not found" when context insufficient?)
4. Conciseness (no padding, repetition, unnecessary hedging?)
5. Correct tone for namespace (technical/friendly/legally precise?)
```

### Iterative DPO (NEW)

After initial DPO training, generate responses from the DPO model on a held-out set, judge them, create new preference pairs from the model's own mistakes, and do a second short DPO pass.

```
Round 1: Train on 5K external pairs (1 epoch)
         |
         v
Round 2: Generate 1K responses from Round 1 model
         Judge with Claude → new preference pairs
         Train on those (0.5 epoch, lower LR 2e-6)
```

This is a simplified version of online DPO — the model learns from its own failure modes.

### Conservative Hyperparameters

v1's DPO LR (1e-5) and beta (0.1) were too aggressive — reward accuracy hit 100% (possible overfitting to reward signal). v2 is more conservative:

| Parameter | v1 | v2 | Why |
|---|---|---|---|
| Learning rate | 1e-5 | 5e-6 | Prevent catastrophic forgetting of RAFT capabilities |
| Beta | 0.1 | 0.15 | Stronger KL penalty keeps model closer to RAFT reference |
| Max seq length | 1024 | 2048 | Allow longer context in preference pairs |

### v2 DPO Config Changes

```yaml
training:
  learning_rate: 5.0e-6              # was: 1.0e-5
  max_seq_length: 2048               # was: 1024

dpo:
  beta: 0.15                         # was: 0.1
  loss_type: "sigmoid"
  iterative_rounds: 2                # NEW

data:
  namespace_balanced: true            # NEW
  compliance_pairs: 1000              # NEW
  judge_model: "claude-sonnet-4-6"    # was: template-based
```

---

## Data Generation: Template → LLM-Synthetic

### Why Templates Failed

v1 used template-based data generation from 70 docs / 926 sections. Problems:

1. **Repetitive patterns**: "What is the [SPEC] for [VEHICLE]?" × 20 variations → model memorizes template structure, not domain knowledge
2. **Limited diversity**: 926 source sections × ~10 templates = ~10K examples with high overlap
3. **No paraphrasing**: Real users ask "brakes feel weird" not "What is the brake diagnostic procedure?"
4. **Public content**: Wikipedia-sourced corpus is already in Nemo's pre-training data

### v2 LLM-Synthetic Generation

Use Claude Sonnet to generate training data with natural diversity:

```
Input: Source document chunk + namespace + difficulty level
Output: Natural question + grounded answer with citations
```

**SFT generation prompt:**
```
Given this automotive document excerpt, generate a realistic question
that a {role} would ask, and a helpful answer grounded in the document.

Vary the question style: some formal, some colloquial, some with typos,
some multi-part, some indirect ("I'm having trouble with..." not "What is...").

Document: {chunk_content}
Namespace: {namespace}
Role: {customer | engineer | dealer | legal | HR | procurement}
```

**RAFT generation prompt:**
```
Given this oracle document and these distractor documents, generate:
1. A question that the oracle document answers
2. An answer that cites ONLY the oracle document
3. Verify the answer does NOT reference any distractor content

Oracle: {oracle_chunk}
Distractors: {distractor_1}, {distractor_2}, {distractor_3}
Oracle-free: {true|false}  (if true, question has NO oracle)
```

**DPO generation prompt:**
```
Given this question and context, generate two responses:
- CHOSEN: factually accurate, properly cited, appropriate tone
- REJECTED: contains a specific flaw (hallucination | wrong citation |
  wrong tone | over-confident | missing uncertainty)

The flaw should be subtle and realistic, not obviously wrong.
```

**Budget estimate:** ~$50-100 for 50K+ examples with Claude Sonnet.

### Data Quality Gates (NEW)

Before training, validate generated data:

| Check | Threshold | Action |
|---|---|---|
| SFT answer grounded in source | 100% | Reject if answer contains facts not in source doc |
| RAFT oracle citation present | 95%+ | Reject if answer doesn't cite oracle |
| RAFT distractor content absent | 95%+ | Reject if answer references distractor |
| DPO chosen > rejected (GPT-4 verify) | 90%+ | Re-judge disagreements |
| Namespace balance | ±10% | Rebalance if skewed |
| Duplicate detection | <5% similar | Deduplicate by semantic similarity |

---

## Evaluation: Template → LLM Judge

### v1 Evaluation Problems

1. **Template-based judge**: Keyword overlap scoring misses semantically correct answers
2. **No RAG in eval**: Model evaluated standalone — doesn't test real production behavior
3. **Only 20 questions per namespace**: Too few for statistical significance

### v2 Evaluation Upgrades

| Component | v1 | v2 |
|---|---|---|
| Judge | Template (keyword overlap) | Claude Sonnet |
| RAG in eval | No | Yes — full pipeline |
| Questions per namespace | 20 | 30 |
| DPO win rate eval | Template judge | Claude side-by-side |
| RAGAS judge | Template | Claude / OpenAI |
| Eval checkpoints | Final only | Every stage (SFT, RAFT, DPO) |
| Base model comparison | None | Always compare fine-tuned vs base + RAG |

### Stage-by-Stage Quality Gates

```
After SFT:
  [x] eval_loss not increasing (early stopping)
  [x] Perplexity on held-out automotive text < base model
  [x] Qualitative spot-check: 10 questions per namespace

After RAFT:
  [x] Oracle citation rate >= 0.85
  [x] Distractor rejection rate >= 0.85
  [x] Abstention rate >= 0.85
  [x] NEW: Test WITHOUT RAG — model should still abstain (not hallucinate training data)

After DPO:
  [x] Win rate >= 15% per namespace (not just overall)
  [x] NEW: No namespace regresses more than -5% (catch compliance regression)
  [x] NEW: Compare DPO model vs base Nemo + RAG (must beat base by >= 10%)

Final (with RAG):
  [x] RAGAS faithfulness >= 0.92 (was 0.88 target)
  [x] RAGAS answer relevancy >= 0.90 (was 0.85)
  [x] RAGAS context precision >= 0.88 (was 0.80)
  [x] RAGAS context recall >= 0.88 (was 0.82)
  [x] P95 latency within targets
```

---

## Training Sequence & Timeline

```
Phase 0: Data Generation (1-2 days)
  - Generate v3 SFT data (20K+ pairs) with Claude
  - Generate v3 RAFT data (25K+ pairs) with harder negatives
  - Generate v3 DPO data (5K+ pairs) namespace-balanced
  - Run data quality gates
  - Estimated cost: $50-100 API credits

Phase 1: SFT Training (~2 hours on A40)
  - 1 epoch, 20K examples, early stopping
  - NEFTune alpha=5, dropout 0.10, LoRA r=32
  - Quality gate: eval_loss, spot-check

Phase 2: RAFT Training (~6-8 hours on A40)
  - 2 epochs, 25K examples, curriculum learning
  - 40% oracle-free, harder negatives
  - Quality gate: citation/rejection/abstention >= 0.85
  - NEW gate: test without RAG for abstention

Phase 3: DPO Round 1 (~30 min on A40)
  - 1 epoch, 5K pairs, namespace-balanced
  - LLM judge (Claude), beta=0.15
  - Quality gate: win rate >= 15% per namespace

Phase 4: DPO Round 2 — Iterative (~20 min on A40)
  - Generate 1K responses from Round 1 model
  - Judge with Claude, create new preference pairs
  - 0.5 epoch on self-generated pairs
  - Quality gate: no regression from Round 1

Phase 5: Evaluation (~1-2 hours)
  - AWQ quantize
  - Full RAGAS eval with RAG pipeline
  - Compare: v2 fine-tuned vs base Nemo + v2 RAG
  - Compare: v2 fine-tuned vs v1 fine-tuned

Total estimated training time: ~10-12 hours
Total estimated cost: ~$10-15 compute + $50-100 data generation
```

---

## Summary: What Makes v2 Fine-Tuning SOTA

| Technique | Category | Impact |
|---|---|---|
| LLM-synthetic data | Data quality | Eliminates template repetitiveness |
| 20K+ SFT / 25K+ RAFT / 5K+ DPO | Data scale | Prevents overfitting on 12B model |
| NEFTune noise | Regularization | Proven 5-15% downstream improvement |
| Early stopping | Efficiency | Stops before overfitting, saves compute |
| LoRA r=32 | Capacity | Better adaptation for 12B model |
| Higher dropout (0.10) | Regularization | Prevents memorization with more data |
| 40% oracle-free RAFT | Abstention | Stronger "I don't know" training for 12B |
| Harder/adversarial negatives | Data quality | Model verifies values, not just structure |
| Curriculum learning | Training strategy | Easy→hard ordering for stable learning |
| Namespace-balanced DPO | Data coverage | Fixes compliance regression |
| LLM judge (Claude) | Evaluation | Accurate semantic quality assessment |
| Iterative DPO | Training strategy | Model learns from own mistakes |
| Conservative DPO hyperparams | Stability | Preserves RAFT capabilities |
| Stage-by-stage quality gates | Safety | Catches problems before they compound |
| Full RAG eval | Realism | Tests actual production behavior |
