# Alvin AI Platform: Fine-Tuning + RAG Architecture & Implementation Plan

**Scope:** Single-region pilot  
**Use Cases:** Customer Support / Service Manuals · Engineer Knowledge Base · Dealer & Sales Enablement · Compliance & Regulatory Documentation  
**Base Model:** Mistral 7B (upgradeable to Mixtral 8x7B)  
**Pipeline:** SFT → RAFT → DPO → RAG

---

## 1. Executive Summary

This document defines the architecture and phased implementation plan for deploying a fine-tuned Mistral model with a Retrieval-Augmented Generation (RAG) layer across four use cases in an automotive company. The system is designed as a single-region pilot with a modular architecture that can be promoted to multi-region enterprise deployment.

The core design principle: **the fine-tuning pipeline (SFT → RAFT → DPO) produces a model that knows how to reason over automotive documents; the RAG pipeline supplies it with the right documents at query time.**

---

## 2. System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        CLIENT LAYER                                 │
│   Web App (SvelteKit)   |   Admin Dashboard   |   REST API            │
└────────────────────────────┬────────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────────┐
│                      API GATEWAY (FastAPI)                          │
│         Auth · Rate Limiting · Query Routing · Logging              │
└────┬──────────────────────┬──────────────────────┬──────────────────┘
     │                      │                      │
┌────▼──────┐     ┌─────────▼──────────┐  ┌───────▼────────────────┐
│  RETRIEVAL │     │  GENERATION        │  │  DOCUMENT INGESTION    │
│  SERVICE   │     │  SERVICE           │  │  PIPELINE              │
│            │     │                    │  │                        │
│ pgvector   │────▶│ vLLM + Mistral 7B  │  │ Parser → Chunker →     │
│ BM25 index │     │ (SFT+RAFT+DPO)     │  │ Embedder → Indexer     │
│ Reranker   │     │                    │  │                        │
└────────────┘     └────────────────────┘  └────────────────────────┘
     │                      │
┌────▼──────────────────────▼──────────────────────────────────────┐
│                    DATA LAYER                                     │
│  PostgreSQL + pgvector  |  MinIO (raw docs)  |  Redis (cache)    │
└───────────────────────────────────────────────────────────────────┘
     │
┌────▼──────────────────────────────────────────────────────────────┐
│                  FINE-TUNING PIPELINE (offline)                   │
│  RunPod A5000  |  Unsloth + LoRA  |  SFT → RAFT → DPO            │
└───────────────────────────────────────────────────────────────────┘
```

---

## 3. Data Architecture by Use Case

Each use case maps to a distinct document corpus, chunking strategy, and retrieval namespace.

### 3.1 Customer Support / Service Manuals

| Attribute | Detail |
|---|---|
| Document types | PDF service manuals, TSBs (Technical Service Bulletins), warranty docs, FAQ compilations |
| Chunking strategy | Hierarchical: section-level (1500 tokens) with sentence-level fallback (300 tokens) |
| Metadata | vehicle_model, year, system (engine/transmission/electrical), doc_type, language |
| Retrieval namespace | `ns:customer_support` |
| Access tier | Public (authenticated customers) |

### 3.2 Internal Knowledge Base (Engineers)

| Attribute | Detail |
|---|---|
| Document types | Engineering specs, CAD notes (text), internal wikis, R&D reports, supplier docs |
| Chunking strategy | Semantic chunking with overlap (800 tokens, 150 overlap) |
| Metadata | department, project_code, classification, author, version |
| Retrieval namespace | `ns:engineering` |
| Access tier | Internal only (role-gated) |

### 3.3 Dealer / Sales Enablement

| Attribute | Detail |
|---|---|
| Document types | Product brochures, pricing sheets, feature comparison matrices, objection handling guides |
| Chunking strategy | Fixed (512 tokens) — short, dense factual content |
| Metadata | model_line, market, effective_date, region |
| Retrieval namespace | `ns:dealer_sales` |
| Access tier | Dealer network (partner-gated) |

### 3.4 Compliance & Regulatory Documentation

| Attribute | Detail |
|---|---|
| Document types | FMVSS/ECE regulations, emissions standards, homologation records, audit trails |
| Chunking strategy | Clause-level chunking preserving regulatory structure (600 tokens) |
| Metadata | regulation_id, jurisdiction, effective_date, vehicle_category, status |
| Retrieval namespace | `ns:compliance` |
| Access tier | Legal/Compliance teams only (strict RBAC) |

---

## 4. Fine-Tuning Pipeline

### 4.1 Stage 1: Supervised Fine-Tuning (SFT)

**Objective:** Adapt Mistral 7B to automotive domain language, terminology, and response format.

**Dataset construction:**
- Extract 10,000–20,000 instruction-response pairs from existing service manual Q&A, closed support tickets, and internal knowledge base articles
- Format: `[INST] {question about automotive system} [/INST] {accurate technical answer}`
- Include all four use case domains proportionally — do not over-represent any single domain
- Clean for PII (customer names, VINs in support tickets)

**Training configuration:**
```
Base model:       mistralai/Mistral-7B-Instruct-v0.3
Method:           LoRA (r=16, alpha=32, dropout=0.05)
Framework:        Unsloth on RunPod A5000
Epochs:           3
Batch size:       4 (gradient accumulation = 8)
Learning rate:    2e-4 with cosine decay
Max seq length:   4096
```

**Validation:** Hold out 10% of dataset. Target: perplexity reduction of ≥20% vs base Mistral on automotive domain prompts.

---

### 4.2 Stage 2: RAFT (Retrieval-Augmented Fine-Tuning)

**Objective:** Train the SFT model to correctly use retrieved automotive documents — distinguishing relevant oracle chunks from distractor chunks.

**Dataset construction (critical stage):**

For each training example, construct a tuple:
```
{
  "question": "What is the torque spec for cylinder head bolts on the 2.0L turbo?",
  "oracle_docs": [chunk from correct service manual section],
  "distractor_docs": [2-3 chunks from plausible but wrong sections],
  "answer": "The correct answer citing the oracle doc"
}
```

- Generate 5,000–8,000 RAFT examples per use case namespace
- Distractor selection strategy: same vehicle model but wrong system, or same system but wrong model year — makes the task non-trivial
- 70% of examples include oracle doc; 30% are oracle-free (model must say "not found in provided context") — this prevents hallucination in production RAG

**Training configuration:**
- Continue from SFT checkpoint
- Same LoRA config; reduce learning rate to 5e-5
- Epochs: 2
- Batch size: 4 (consistent with SFT/DPO; if OOM during RAFT due to long oracle+distractor context, drop to 2 and increase grad_accum to 16)
- Gradient accumulation: 8 (effective batch = 32)

**Validation metric:** On held-out RAFT examples, measure: (a) does model cite oracle doc, (b) does model correctly ignore distractors, (c) does model correctly abstain on oracle-free examples.

---

### 4.3 Stage 3: DPO (Direct Preference Optimization)

**Objective:** Align RAFT model output with quality preferences — factual precision, appropriate uncertainty, concise formatting.

**Preference dataset construction:**
- For each query, generate 2 responses from the RAFT model using different sampling temperatures
- Human annotators (or GPT-4 as judge) mark preferred vs rejected response
- Target: 2,000–4,000 preference pairs
- Focus annotation effort on: (a) compliance queries where precision is legally critical, (b) customer-facing queries where tone matters, (c) cases where model retrieved correct chunk but answered incorrectly

**Training configuration:**
```
Method:     DPO (β=0.1)
Framework:  TRL DPOTrainer
Base:       RAFT checkpoint
Epochs:     1
LR:         1e-5
```

**Validation:** Side-by-side eval: DPO model vs RAFT-only model on 200 representative queries. Target: ≥15% preference win rate for DPO model.

---

## 5. RAG Pipeline Architecture

### 5.1 Document Ingestion Pipeline

```
Raw Document (PDF/DOCX/HTML)
        │
        ▼
   [Parser]
   pdfplumber / python-docx / BeautifulSoup
        │
        ▼
   [Chunker]
   LangChain RecursiveCharacterTextSplitter
   (strategy varies by namespace — see Section 3)
        │
        ▼
   [Metadata Extractor]
   vehicle_model, year, doc_type, namespace, source_id
        │
        ▼
   [Embedder]
   all-MiniLM-L6-v2 (self-hosted, CPU, 384-dim, L2-normalised)
   Note: mistral-embed and bge-large-en-v1.5 evaluated and rejected — see CLAUDE.md Decision Log.
        │
        ▼
   [Vector Store]
   PostgreSQL + pgvector
   (separate table per namespace for access control)
        │
        ▼
   [BM25 Index]
   rank_bm25 (BM25Okapi — lightweight, no Elasticsearch dependency for pilot)
```

### 5.2 Query Pipeline

```
User Query
    │
    ▼
[Query Classifier]
Lightweight classifier → routes to namespace(s)
e.g. "torque spec" → ns:customer_support + ns:engineering
    │
    ▼
[Hybrid Retriever]
Dense (pgvector cosine) + Sparse (BM25) → score fusion
top-k = 10 candidates
    │
    ▼
[Reranker]
cross-encoder/ms-marco-MiniLM-L-6-v2
Rerank to top-3 or top-5 chunks
    │
    ▼
[Context Assembly]
Construct prompt:
  [INST] Given the following documents:
  {chunk_1} ... {chunk_n}
  Answer: {user_query} [/INST]
    │
    ▼
[Mistral 7B — SFT+RAFT+DPO]
vLLM inference server
    │
    ▼
[Response + Source Citations]
Return answer + source document references to client
```

### 5.3 Hybrid Retrieval Configuration

| Parameter | Value |
|---|---|
| Dense retrieval | pgvector cosine similarity, top-10 |
| Sparse retrieval | BM25, top-10 |
| Fusion method | Reciprocal Rank Fusion (RRF) |
| Reranker | cross-encoder, rerank to top-5 |
| Similarity threshold | 0.72 (below this → "not found" response) |
| Cache layer | Redis, TTL 1 hour for frequent queries |

---

## 6. Infrastructure (Single-Region Pilot)

### 6.1 Compute

| Component | Spec | Purpose |
|---|---|---|
| Hetzner CCX33 (4 vCPU, 32GB) | Always-on | API Gateway, FastAPI services, PostgreSQL, Redis |
| Hetzner CCX53 (16 vCPU, 64GB) | Always-on | vLLM inference server (quantized Mistral 7B AWQ) |
| RunPod A5000 (24GB VRAM) | On-demand | Fine-tuning pipeline (SFT, RAFT, DPO) |

### 6.2 Storage

| Component | Purpose |
|---|---|
| PostgreSQL + pgvector | Vector store, metadata, access control |
| MinIO | Raw document storage (PDFs, DOCX) |
| Redis | Query cache, session state |

### 6.2b Model Hot-Swap Architecture

The production vLLM server supports **zero-downtime model replacement** via the vLLM REST API. A single vLLM instance serves one model slot; swaps pull new weights from MinIO and reload in-place without restarting the container.

```
MinIO (models bucket)
  └── models/alvinai-7b-{run_id}-awq/   ← versioned by DPO run ID
  └── models/alvinai-7b-latest-awq/     ← symlinked to current production

vLLM (single instance, port 8080)
  └── /v1/models                             ← lists currently loaded model
  └── POST /v1/models/load                   ← hot-swap endpoint

deploy.sh --model models/alvinai-7b-{run_id}-awq
  1. Pull new model weights from MinIO → /models/staging/ on inference node
  2. Run smoke test against staging path (5 inference checks via vLLM --model flag on temp port)
  3. POST /v1/models/load with new model path → vLLM drains in-flight requests, loads new weights
  4. Verify /v1/models returns new model name
  5. Update MinIO symlink: latest-awq → {run_id}-awq
  6. Log swap event to PostgreSQL model_versions table
```

**Rollback:** `deploy.sh --rollback` POSTs the previous run_id path to `/v1/models/load`. Previous weights remain on disk until explicitly purged.

**Model versioning:** `models/alvinai-7b-{run_id}-awq/` where `run_id` is the W&B or timestamp ID from the DPO training run. The `model_versions` table in PostgreSQL tracks: run_id, loaded_at, loaded_by, eval_scores, rollback_available.

### 6.3 Reverse Proxy & TLS

**Caddy** with automatic TLS (ACME/Let's Encrypt, zero-config). Routes:
- `api.{domain}` → FastAPI Gateway (port 8000)
- `ingest.{domain}` → Document ingestion service (internal only)
- `{domain}` → SvelteKit frontend (port 3000)

Configuration: `infra/caddy/Caddyfile`. No separate certbot required — Caddy manages certificate issuance and renewal automatically.

### 6.4 Containerisation

Docker Compose with services:
```
services:
  - backend            (FastAPI — API gateway + RAG pipeline)
  - celery             (Celery worker)
  - beat               (Celery beat scheduler)
  - vllm               (vLLM inference server — AWQ model, hot-swappable)
  - db                 (PostgreSQL + pgvector)
  - redis              (cache + Celery broker)
  - minio              (raw docs + model weights storage)
  - caddy              (reverse proxy + automatic TLS)

Note: Nginx and certbot are NOT used. Caddy replaces both.
```

---

## 7. Access Control & Security

```
User Role          Accessible Namespaces
─────────────────────────────────────────────────────
Customer           ns:customer_support
Dealer/Partner     ns:customer_support, ns:dealer_sales
Engineer           ns:customer_support, ns:engineering
Legal/Compliance   ns:compliance, ns:engineering (read)
Admin              All namespaces
```

- JWT-based auth at API Gateway
- Namespace filtering applied at retrieval layer — a user cannot retrieve chunks outside their permitted namespaces even if they craft a direct query
- Compliance namespace: all queries logged with user_id, timestamp, query text, retrieved source IDs for audit trail
- MinIO bucket policies mirror namespace access tiers

---

## 8. Evaluation Framework

### 8.1 Retrieval Metrics

These must pass **before** RAFT dataset generation begins. Run via `scripts/run_evaluation.py --stage retrieval`.

| Metric | Target | Notes |
|---|---|---|
| Recall@5 | ≥ 0.85 | Top-5 chunks contain the correct answer chunk |
| MRR | ≥ 0.75 | Mean Reciprocal Rank across held-out queries |
| NDCG@5 | ≥ 0.80 | Normalised Discounted Cumulative Gain |

### 8.2 Generation Metrics (RAGAS)

| Metric | Target |
|---|---|
| Faithfulness | ≥ 0.88 | Compliance namespace: ≥ 0.93 |
| Answer Relevance | ≥ 0.85 | All namespaces |
| Context Precision | ≥ 0.80 | All namespaces |
| Context Recall | ≥ 0.82 | All namespaces |

### 8.3 Use-Case Specific

| Use Case | Key Eval |
|---|---|
| Service manuals | Torque/spec accuracy vs ground truth |
| Engineering KB | Citation correctness, version accuracy |
| Dealer/Sales | Feature claim accuracy vs product sheets |
| Compliance | Clause citation precision, jurisdiction accuracy |

---

## 9. ML Pipeline Implementation Plan (Weeks 1–12 of 22-week full build)

> This section covers the ML pipeline only. For the complete build timeline including frontend, hardening, monitoring, and E2E tests, see TASKS.md Complete Build Summary.

### Phase 1 — Data & Infrastructure (Weeks 1–3)

- [ ] Audit and collect document corpus per namespace
- [ ] Set up Hetzner VPS + Docker Compose + Caddy + PostgreSQL + pgvector + MinIO + Redis
- [ ] Build and test document ingestion pipeline (parser → chunker → embedder → pgvector)
- [ ] Ingest pilot corpus: 500–1000 documents per namespace
- [ ] Build BM25 index on pilot corpus
- [ ] Validate retrieval pipeline end-to-end with manual queries

### Phase 2 — SFT (Weeks 4–5)

- [ ] Construct SFT dataset (10,000–20,000 pairs) from existing support tickets, manuals, internal docs
- [ ] Set up RunPod A5000 with Unsloth environment
- [ ] Run SFT training on Mistral 7B
- [ ] Evaluate on held-out set; iterate on dataset quality if perplexity target not met
- [ ] Export and quantize SFT checkpoint (AWQ 4-bit)

### Phase 3 — RAFT (Weeks 6–7)

- [ ] Construct RAFT dataset (5,000+ examples with oracle + distractor chunks per namespace)
- [ ] Run RAFT training from SFT checkpoint
- [ ] Evaluate on RAFT-specific metrics (oracle citation, distractor rejection, abstention)
- [ ] Deploy RAFT model to vLLM on Hetzner; run first end-to-end RAG tests

### Phase 4 — DPO (Week 8)

- [ ] Generate preference pairs using RAFT model outputs
- [ ] Annotate preferred vs rejected responses (focus on compliance and customer-facing)
- [ ] Run DPO training
- [ ] Side-by-side eval: DPO vs RAFT-only
- [ ] Deploy final model checkpoint to production vLLM server

### Phase 5 — RAG Integration & Evaluation (Weeks 9–10)

- [ ] Integrate hybrid retriever (dense + BM25 + reranker) with generation service
- [ ] Build query classifier for namespace routing
- [ ] Run RAGAS evaluation across all four use cases
- [ ] Tune retrieval parameters (top-k, threshold, RRF weights) based on eval results
- [ ] Implement Redis caching layer

### Phase 6 — Access Control, UI & Pilot Launch (Weeks 11–12)

- [ ] Implement JWT auth + namespace RBAC
- [ ] Build single SvelteKit web app with role-based access per namespace
- [ ] Set up query logging for compliance namespace audit trail
- [ ] Internal pilot with 20–30 users per use case
- [ ] Collect feedback, identify failure modes, prioritise next DPO iteration

---

## 10. Feedback Loop & Post-Pilot Roadmap

Production is the source of the next training cycle:

```
Production queries
      │
      ▼
Log failures (wrong retrieval / wrong answer / user thumbs down)
      │
      ▼
Failure triage: retrieval failure vs generation failure
      │
      ├── Retrieval failure → improve chunking, embeddings, reranker
      │
      └── Generation failure → add to next DPO preference dataset
              │
              ▼
         Next DPO round (monthly cadence recommended)
```

**Post-pilot upgrade triggers:**
- Scale to Mixtral 8x7B if Mistral 7B accuracy plateaus on engineering KB
- Add multilingual support (dealer markets) via multilingual embedding model swap
- Promote to multi-region if pilot success metrics are met across all four use cases

---

## 11. Key Risks & Mitigations

| Risk | Mitigation |
|---|---|
| RAFT dataset quality poor | Invest 60% of dataset construction time here; use GPT-4 to auto-generate distractor candidates |
| Compliance namespace hallucination | Strict similarity threshold + mandatory source citation in response format + human review layer for high-stakes queries |
| vLLM cold start on quantized model | Pre-load model on container start; use Hetzner always-on (not serverless) for inference |
| Document corpus version drift | MinIO versioning + metadata `effective_date` field; ingestion pipeline checks for updated docs weekly |
| Access control bypass | Namespace filtering at retrieval layer (not just API layer); pen test before pilot launch |

---

*Document version: 1.0 | Pilot scope | Single-region deployment*
