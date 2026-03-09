# AlvinAI — Tasks V2 (Revised Execution Plan)

**Created:** 2026-03-09
**Purpose:** Track actual progress and divergences from the original TASKS.md plan.

---

## Divergences from Original Plan

| Decision | Original Plan | Revised Approach | Reason |
|---|---|---|---|
| Infrastructure timing | Hetzner setup implicit before Phase 2 | Defer Hetzner; train model first on RunPod | Datasets ready; no need to block on infra |
| Hetzner server | CCX33 (dedicated, 32GB) + CCX53 (64GB) | Single server TBD; existing CPX22 too loaded | Existing server runs 32 containers; need fresh server |
| vLLM on Hetzner | CCX53 for inference | Deferred — Hetzner CCX has no GPU | vLLM requires NVIDIA GPU; Hetzner CPX/CCX are CPU-only |
| Task ordering | Sequential 1.1→1.2→1.3→...→1.12 | Skip to training (1.5b→1.7→1.8→1.10→1.12) | SFT/RAFT/DPO datasets already exist; backend code not needed for training |
| HF org | `tbqguy` | `sarfarazflow` | Owner's GitHub/HF org |
| Docker runtime verification | Local `docker compose up -d` | Deferred to Hetzner deployment | No GPU locally; saves bandwidth on multi-GB image pulls |

---

## Revised Task Sequence

### Phase A — Training Pipeline (Current Focus)

| # | Task | Status | Notes |
|---|---|---|---|
| 1.1 | Monorepo Scaffold | [x] Complete | Commit `44bd350` |
| 1.2 | Docker Compose Environment | [x] Complete | Config validated; runtime deferred to Hetzner |
| 1.5b | Training Configuration YAMLs | [ ] Next | sft_config.yaml, raft_config.yaml, dpo_config.yaml |
| 1.7 | SFT Dataset → project structure | [ ] Next | Copy existing datasets into data/sft/, data/raft/, data/dpo/ |
| 1.8 | SFT Training on RunPod | [ ] Blocked on 1.5b, 1.7 | 1,600 train + 400 test (automotive) + 1,600 HR SFT |
| 1.10 | RAFT Training on RunPod | [ ] Blocked on 1.8 | 3,200 train + 800 test |
| 1.12 | DPO Training on RunPod | [ ] Blocked on 1.10 | 700 preference pairs |
| 1.12b | AWQ Export & HF Upload | [ ] Blocked on 1.12 | Push to sarfarazflow/alvinai-7b-awq-YYYYMMDD |

### Phase B — Backend & Infrastructure (After Model Trained)

| # | Task | Status | Notes |
|---|---|---|---|
| 1.2a | Hetzner Server Provisioning | [ ] Not started | Need fresh server (CPX31 8GB or CCX33 32GB) |
| 1.2-verify | Docker Compose Runtime Verification | [ ] Blocked on 1.2a | pg_isready, redis ping, MinIO, Caddy on Hetzner |
| 1.3 | Python Backend Skeleton | [ ] Not started | FastAPI + pyproject.toml |
| 1.4 | Database Models & Migration | [ ] Not started | SQLAlchemy models + Alembic |
| 1.5 | Document Ingestion Pipeline | [ ] Not started | PDF/DOCX parser → chunker → embedder → pgvector |
| 1.6 | Hybrid Retrieval Service | [ ] Not started | pgvector + BM25 + reranker |

### Phase C — Integration & Frontend (After Backend + Model)

| # | Task | Status | Notes |
|---|---|---|---|
| 2.1 | Authentication & RBAC | [ ] Not started | |
| 2.2 | RAG Pipeline Core (SSE) | [ ] Not started | |
| 2.3 | vLLM Inference Server | [ ] Not started | Needs GPU server (not Hetzner CCX) |
| 2.4 | RAGAS Evaluation Pipeline | [ ] Not started | |
| 2.5 | Conversation History & Feedback | [ ] Not started | |
| 3.1 | SvelteKit Frontend Scaffold | [ ] Not started | |
| 3.2 | AI Chat Interface | [ ] Not started | |
| 3.3 | Admin Dashboard | [ ] Not started | |
| 3.4 | Caddy + Production Deployment | [ ] Not started | |

### Phase D — Hardening (After Pilot)

| # | Task | Status | Notes |
|---|---|---|---|
| 4.1 | Drift Detection | [ ] Not started | |
| 4.2 | Metrics + Monitoring | [ ] Not started | |
| 4.3 | E2E Test Suite | [ ] Not started | |
| 4.4 | Email Alerting & Digest | [ ] Not started | |
| 4.5 | Performance Profiling | [ ] Not started | |
| 4.6 | Final Documentation | [ ] Not started | |

---

## Dataset Inventory

| Dataset | File | Train | Test | Format |
|---|---|---|---|---|
| SFT (automotive) | alvin_sft_{train,test}.jsonl | 1,600 | 400 | OpenAI messages (system/user/assistant) |
| SFT (HR) | hr_sft_{train,test}.jsonl | 1,600 | 400 | OpenAI messages (system/user/assistant) |
| SFT (HR CSV) | alvin_sft_employee_hr.csv | — | — | CSV alternate format |
| RAFT | alvin_raft_{train,test}.jsonl | 3,200 | 800 | Messages + oracle/distractor metadata |
| DPO | alvin_dpo.csv | ~700 | — | CSV: chosen/rejected pairs |

**Source documents (for RAG):**
- HR-POL-001-Leave-Attendance.docx
- HR-POL-002-Code-of-Conduct.docx
- HR-POL-003-Remote-Work.docx
- HR-POL-004-Compensation-Benefits.docx

---

## Open Questions

1. **GPU for vLLM inference:** Hetzner CCX is CPU-only. Options: RunPod persistent pod, Lambda, or managed GPU provider. Decision needed before Task 2.3.
2. **Dataset size:** SFT has 3,200 total (vs architecture target of 10-20K). May need augmentation after first training run evaluation.
3. **DPO split:** 700 pairs in single CSV, no train/test split yet. Need to split before Task 1.12.

---

## Infra Notes

- **Existing Hetzner CPX22** (46.224.190.110): Fully loaded with 32 containers (ERPNext, Forgejo, study apps, etc). NOT suitable for AlvinAI.
- **SSH key:** `~/.ssh/id_ed25519` connects to existing server. New server will need key uploaded to Hetzner.
- **RunPod:** Not yet set up. Need account + SSH key + network volume.
