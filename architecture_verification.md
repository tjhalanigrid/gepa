# Architecture Verification — What Was Actually Built

> Cross-reference of the two architecture diagrams (Context Engineering + Data Flow)
> against the actual codebase state. Written for honest gap analysis before MVP demo.

---

## Image 1 — Context Engineering Diagram

### TURN 1: Initial Assessment

| Component | Built? | Reality |
|---|---|---|
| User Uploads Photos | ✅ Done | FastAPI `/assess` endpoint accepts file upload |
| LLM Receives Context | ✅ Done | Orchestrator builds message with image + claim data |
| **Thinking with Images** | ⚠️ Inverted | YOLO runs FIRST unconditionally, VLM gets results pre-loaded. The "VLM sees image first" intent is backwards from the diagram |
| CV Tool Calls Dispatched (5 parallel) | ❌ Wrong | Only YOLO runs (unconditionally, not via VLM decision). Tools are sequential, not parallel |
| Segmentation | ❌ Stub | `models/part_segmentation/infer.py` exists — 3 lines, returns `[]`, nothing implemented |
| Depth Estimation | ❌ Missing | Not anywhere in the codebase |
| OCR | ❌ Disconnected | Plate detection stub exists in `models/plate_rc_detection/` but not connected to anything |
| Object Detection | ✅ Done | YOLO works, trained model, mAP50 = 0.749 |
| Damage Classifier | ✅ Done | Same YOLO model, 6 damage classes |
| Results Aggregate | ⚠️ Partial | Code exists to merge results but VLM never successfully produced output (proven by `outputs/tool_test_report.json`: `tool_call_log: []`) |

---

### Automated Approvals (Temporal)

| Component | Built? | Reality |
|---|---|---|
| Damage < threshold → Auto-approve / Escalate | ✅ Done | Approval gate at ₹50,000 implemented in `pipeline/orchestrator.py run()` |
| Temporal (durable workflow engine) | ❌ Missing | Explicitly listed as out of scope in `CLAUDE.md` |
| Timeout Escalation Ladder (24h → 48h → 72h) | ❌ Missing | Not built anywhere |
| Request More Info / Additional Photos flow | ❌ Missing | Not built |

---

### RLM + CodeAct Execution Layer

| Component | Built? | Reality |
|---|---|---|
| CodeAct loop structure | ✅ Done | Loop, JSON parser, retry logic all built in `pipeline/orchestrator.py` |
| Phase 1: Reasoning | ⚠️ Partial | Coded correctly but VLM output wrong format in actual test run |
| Phase 2: Code Generation | ⚠️ Partial | VLM supposed to write Python using COST_DB — never triggered in practice |
| Phase 3: Execution (Sandboxed runtime) | ✅ Done | `models/vlm_reasoning/sandbox.py` — AST validation, whitelisted builtins, 10s timeout |
| Phase 4: Synthesis | ⚠️ Partial | Code path exists in orchestrator but VLM never reached `Terminate` successfully |

---

### TURN 2+: Follow-up Interactions

| Component | Built? | Reality |
|---|---|---|
| Context Management (PINNED / RETAINED / DROPPED) | ✅ Built | `pipeline/context_manager.py` — `ClaimContext` with `pin()`, `add_turn()`, `build_messages()` fully implemented |
| Sliding Window (last 3 turns + compressed history) | ✅ Built | Works exactly as diagram shows — drops oldest turn to compressed summary when window full |
| Wired to backend / orchestrator | ❌ Disconnected | `context_manager.py` is built but **never called** from FastAPI or the orchestrator loop |

---

## Image 2 — Data Flow Architecture Diagram

### Ingestion Layer

| Component | Built? | Reality |
|---|---|---|
| User Application (Mobile / Web / API) | ✅ Done | Streamlit frontend (`frontend/app.py`) + direct FastAPI |
| API Gateway (Auth, Rate Limit, Route) | ⚠️ Partial | FastAPI routing done (`backend/app/main.py`), no real auth or rate limiting |

---

### Processing Layer

| Component | Built? | Reality |
|---|---|---|
| Image Store | ✅ Done | Local filesystem `data/uploads/` with `annotated/`, `masked/`, `yolo_annotated/` subdirs |
| CV Model Registry | ❌ Missing | No registry — models called directly by orchestrator |
| Instance Segmentation (Grounding DINO + SAM2) | ❌ Stub | `models/part_segmentation/infer.py` — 18 lines, returns `[]` |
| Depth Estimation | ❌ Missing | Not in codebase at all |
| Object Detection | ✅ Done | YOLOv8 `best.pt` — fully trained and working |
| OCR | ❌ Disconnected | `models/plate_rc_detection/infer.py` exists, zero integration |
| Damage Classifier | ✅ Done | YOLOv8 6-class model (dent, scratch, crack, glass_shatter, lamp_broken, tire_flat) |

---

### Intelligence Layer

| Component | Built? | Reality |
|---|---|---|
| LLM Orchestrator (Multimodal Reasoning + CV Tool Orchestration + Thinking with Images) | ⚠️ Partial | Built (~2400 lines, `pipeline/orchestrator.py`) but VLM did not follow CodeAct format in test run — `tool_call_log: []` |
| Vehicle DB (Specs + History) | ❌ Disconnected | PostgreSQL schema written (`backend/migrations/schema.sql`), in-memory sessions only, nothing connected |
| Pricing DB (Parts + Labor) | ✅ Done | `models/vlm_reasoning/cost_db.py` — full `COST_DB` dict, all 6 damage types × 16 parts |

---

### Output Layer

| Component | Built? | Reality |
|---|---|---|
| Report Generator (PDF / DOCX / API) | ⚠️ Partial | JSON report via API only — no PDF, no DOCX generation |
| Temporal Workflow (Durable Orchestration) | ❌ Missing | Not built, explicitly out of scope per `CLAUDE.md` |
| Notification Service (Email / SMS / Push) | ❌ Missing | Not built, explicitly out of scope per `CLAUDE.md` |
| User Application (Dashboard / Mobile) | ⚠️ Partial | Basic Streamlit dashboard exists, not fully connected to pipeline output |

---

## Overall Score

| Area | Diagram Intent | Reality |
|---|---|---|
| Core AI loop | VLM drives all tool calls | YOLO hardcoded before VLM; VLM failed to call any tool in test |
| CV tools | 5 models running in parallel | 1 model working (YOLO); rest are stubs or missing |
| Cost computation | VLM generates Python code → sandbox executes | Sandbox built correctly; VLM never reached it |
| Context / Multi-turn | Full sliding window per claim | `context_manager.py` built but never wired to backend |
| Output pipeline | PDF report → workflow → notifications | JSON only; workflow and notifications not built |
| What genuinely works end-to-end | Full pipeline | **YOLO detection + approval gate only** |

---

## What Was Genuinely Well Built

These components are solid and production-quality regardless of integration gaps:

- **`pipeline/orchestrator.py`** — loop structure, JSON parser, retry logic, trajectory saving
- **`models/vlm_reasoning/sandbox.py`** — AST-validated restricted Python execution, correct security design
- **`models/vlm_reasoning/cost_db.py`** — complete pricing database with normalisation
- **`pipeline/context_manager.py`** — full sliding window implementation, correct three-tier memory model
- **`shared/bbox_canvas.py`** — full interactive SVG bounding box editor in pure JS (drag, resize, draw, delete)
- **`models/damage_detection/`** — trained YOLOv8 model, frozen, mAP50 = 0.749
- **`pipeline/schema.py`** — complete Pydantic contracts for all data shapes

---

## Key Evidence of What Did Not Work

**File:** `outputs/tool_test_report.json`

```json
{
  "tool_call_log": [],
  "damage_part_map": [],
  "total_min": 0,
  "total_max": 0,
  "approval_decision": "UNKNOWN",
  "total_inference_s": 66.45,
  "raw_vlm_response": "```json\n{\"detections\": [...]}```<|im_end|>"
}
```

- `tool_call_log: []` — VLM called zero tools
- `<|im_end|>` leaked — `skip_special_tokens=True` missing from tokenizer decode
- VLM returned YOLO-format JSON instead of CodeAct `{thought, actions}` format
- 66 seconds inference time — model loaded and ran, but produced wrong output format

---

## What the Pending Refactor Fixes

The planned changes to `pipeline/orchestrator.py` and `configs/global_config.yaml` address:

1. Removing eager YOLO (Stage 1) so VLM sees raw image first
2. Adding `execute_cost_computation` as a real CodeAct tool
3. Rewriting `CODEACT_SYSTEM_PROMPT` to match the diagram's "Thinking with Images" flow
4. Increasing `max_iterations` from 2 → 6, token limits from 120 → 512
