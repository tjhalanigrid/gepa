# CLAUDE.md — veh_dmg_detection

> This file is the authoritative context document for this codebase.
> Read it fully before touching anything. Keep it updated when decisions change.
> Written for a senior engineer. No hand-holding.

---

## What This Project Is

Car damage detection and cost estimation MVP. A user uploads a photo of a damaged
vehicle. The system identifies what is damaged, which part it belongs to, and
estimates repair cost in INR. Output is a structured JSON report with an
auto-approve or escalate-to-human decision.

Built by a 6-intern team at Grid Dynamics. No budget. No paid APIs. No cloud
services. Open source only. Scope creep is the enemy — if a component is not in
this file, it is not in scope.

**Org:** akorellagrid  
**Repo:** `https://github.com/akorellagrid/veh_dmg_detection`  
**Active branch:** `damage-detection`  
**Language:** Python 3.10 throughout. No exceptions.

---

## Architecture: What It Actually Does

### The Mental Model

Qwen (the VLM) is the **sole brain**. It sees the image, classifies damage itself,
and may call a few OPTIONAL vision aids — its choice, any order, any count, or none.
There is no fixed workflow. The trained `best.pt` detector is **retired**
(`models/_archive/best.pt`). Repair cost is computed in the **backend** (plain
Python over `COST_DB`), not by the LLM. **SAM2** runs in the backend only, to render
the segmentation masks and the **merged-union (VLM ∪ SAM2) boxes** in the UI.

```
image
  │
  ▼
orchestrator.run()
  │
  ▼  Stage 1 — VLM brain loop (free-form)
PiAgent (qwen3.5:9b via Ollama) ── sees the image ── classifies damage itself
  │   optional tools, agent's choice:
  │     run_damage_detection · zoom_region · detect_part
  └─ Terminate → damage_items [{damage_type, part, severity, confidence, bbox_pct}]
  │
  ▼  Stage 2-3 — BACKEND (deterministic, no LLM)
  ├─ detections_with_bbox  (bbox_pct → pixels)
  ├─ cost  = lookup_cost() over COST_DB        → damage_part_map, totals
  │
  ▼  Stage 4 — SAM2 (backend) + merged union
  ├─ SAM2 region boxes  ∪  VLM damage boxes    → merged_detections (source-tagged:
  │     vlm / both / sam2)   +  SAM2 masks (generate_masked_image)
  │
  ▼  Stage 5 — annotated image + approval + iteration log
  └─ FinalDamageReport
        • detections_with_bbox  → "Detected Damage" boxes
        • merged_detections     → "Merged (VLM ∪ SAM2)" boxes
        • masked_image          → "Damage masks (SAM2)"
        • iterations            → iteration-logs panel (tool, why, result)
        • approval: AUTO_APPROVED, or ESCALATE_TO_HUMAN
```

**Anti-hallucination contract:** approval is based on the VISUAL assessment quality
(what the AI sees), NEVER on cost. A VLM final claim is trusted only when the VLM's
*independent* `run_damage_detection` pass corroborated the same damage class.
It escalates when a final claim is uncorroborated, or no usable assessment was
produced. SAM2's region boxes are advisory context in the merged view; they never
assert damage. Pricing has a SINGLE source — `cost_db.py` (underscore part keys).

### Why This Architecture

The old sequential pipeline (YOLOv8 → SAM2 → OpenCV intersection → VLM) failed
because: (1) OpenCV center-point intersection logic silently produced null mappings
on deformed geometry, (2) all three models ran on every image regardless of what
was in it, (3) the VLM was passive and had no ability to ask for more information
or handle ambiguity.

The tool-calling architecture fixes (1) by replacing deterministic math with VLM
spatial reasoning, fixes (2) via conditional dispatch, and fixes (3) by making
the VLM the decision-making agent throughout.

The cost: VLM inference is 3–8s per forward pass. Total pipeline is 8–35s
depending on number of tool calls. This is acceptable for an MVP upload workflow.

---

## Stack — Confirmed and Frozen

> **Mentor directive (current):** Qwen is the SOLE brain. No YOLO, no DINOv2 in the
> decision path. Tools are free-form (the agent picks). Cost is computed in the
> BACKEND from the agent's final JSON. SAM2 runs in the backend only, to render the
> segmentation masks and the merged-union (VLM ∪ SAM2) boxes shown in the UI.

| Layer | Technology | Notes |
|---|---|---|
| Damage detection + classification | `qwen3.5:9b` VLM brain | The VLM sees the image and reports damage_items (with bbox_pct). Sole perception. |
| Repair cost | Backend Python | `lookup_cost()` over `COST_DB` — deterministic, NOT the LLM, NOT a sandbox. |
| Segmentation masks + merged union | SAM2 (ultralytics, backend) | stock `sam2.1_b.pt`. UI-only: masks + VLM∪SAM2 boxes. No labels, not a decision tool. |
| ~~Damage detection (trained)~~ | ~~YOLOv8 `best.pt`~~ | **RETIRED** → `models/_archive/best.pt` |
| ~~Vehicle detect (YOLO) / DINOv2~~ | — | **Dropped from the path** (kept dormant on disk) |
| VLM brain | `qwen3.5:9b` via Ollama | HTTP to `localhost:11434` (see `ollama_client.py`) |
| Backend | FastAPI | Single app, not microservices |
| Database | PostgreSQL | Local instance. Images stored as BYTEA in `claim_images` table. |
| Image storage | PostgreSQL BYTEA | NOT filesystem — images are persisted to DB after pipeline, temp files deleted |
| Dashboard | Streamlit | Not started |
| Python | 3.10 | Hard requirement throughout |

**Do not introduce:** microservices, Docker Compose multi-service, Celery,
Redis, Kafka, Temporal, Pinecone, Weaviate, Keycloak, any paid API, any cloud
storage.

---

## Repo Structure

```
veh_dmg_detection/
├── CLAUDE.md                            ← this file
├── context.md                           ← full project context reference
├── configs/
│   └── global_config.yaml              ← single source of truth for all params
├── models/
│   ├── _archive/
│   │   └── best.pt                     ← RETIRED YOLOv8 weights
│   ├── damage_detection/               ← FROZEN/RETIRED. Do not modify.
│   ├── part_segmentation/              ← Grounding DINO + SAM2. DORMANT.
│   ├── vehicle_detection/              ← Stock YOLO ROI. DORMANT.
│   ├── plate_rc_detection/             ← In progress. Not integrated.
│   └── vlm_reasoning/                  ← Active architecture. The brain.
│       ├── pi_agent.py                 ← VLM agentic loop + all prompts + tool dispatch
│       ├── ollama_client.py            ← stdlib-only Ollama REST wrapper
│       ├── cost_db.py                  ← COST_DB pricing table + lookup_cost()
│       ├── tool_registry.py            ← DEAD CODE (old design). Do not use.
│       └── sandbox.py                  ← CodeAct exec sandbox. DORMANT (not called).
├── pipeline/
│   ├── orchestrator.py                 ← Main entry point. Runs all 5 stages.
│   ├── schema.py                       ← All Pydantic contracts. Append only.
│   ├── context_manager.py             ← Multi-turn sliding window. Not active.
│   ├── trajectory_filter.py           ← SFT data filtering.
│   └── feedback_reader.py             ← Reads corrections_log for few-shot.
├── backend/
│   └── app/
│       ├── main.py                     ← FastAPI app factory + lifespan
│       ├── state.py                    ← In-memory jobs + sessions dicts
│       ├── db.py                       ← SQLAlchemy engine
│       ├── models.py                   ← ORM: User, Vehicle, Claim, ClaimImage, etc.
│       ├── auth.py                     ← Token auth
│       ├── schemas.py                  ← Router request/response models
│       ├── core/config.py              ← Paths, constants, load_config()
│       ├── routers/
│       │   ├── assessment.py           ← POST /assess, GET /job/{id}
│       │   ├── sessions.py             ← HITL session endpoints
│       │   ├── images.py               ← Image serving from DB
│       │   ├── feedback.py             ← /recalculate, /feedback/stats
│       │   ├── health.py               ← GET /health
│       │   ├── accounts.py, vehicles.py, claims.py, insurance.py, settings.py
│       └── services/
│           ├── assessment.py           ← Async job runner
│           ├── cost.py                 ← apply_cost_lookup() with severity fallback
│           ├── imaging.py              ← generate_annotated_image(), write_yolo_labels()
│           └── feedback.py             ← write_feedback() to feedback_log.jsonl
├── shared/
│   ├── sam_mask.py                     ← SAM2 mask overlay (ultralytics). Best-effort.
│   ├── image_utils.py
│   ├── bbox_canvas.py
│   └── logger.py
├── dashboard/
│   └── app.py                          ← Streamlit. Not started.
├── data/
│   ├── uploads/                        ← Temp files only (deleted after pipeline)
│   ├── trajectories/raw/               ← Trajectory JSONs for SFT
│   ├── feedback/feedback_log.jsonl     ← HITL approvals
│   ├── feedback/corrections_log.jsonl  ← Per-item correction records
│   └── yolo_corrections/               ← YOLO fine-tune samples from HITL
└── scripts/
    ├── sft_train.py
    └── prepare_sft_dataset.py
```

---

## The One Rule Every Model Must Follow

Every model exposes exactly one public function:

```python
def run(image_path: str, config: dict) -> dict:
    ...
```

- Input: absolute image path + config dict (loaded from yaml by caller)
- Output: `.model_dump()` of the relevant Pydantic schema from `pipeline/schema.py`
- No model imports from another model
- No model imports from `pipeline/` except `schema.py`
- No model has side effects beyond writing to `outputs/` if configured to do so
- The orchestrator is the only thing that calls `run()` functions

---

## Pipeline Schemas — `pipeline/schema.py`

All schemas below are **already implemented**. Append only — never overwrite.

```python
# CodeAct loop schemas
class CodeActAction(BaseModel):
    name: str
    arguments: Dict[str, Any]

class CodeActTurn(BaseModel):
    thought: str
    uncertainty: List[str]
    actions: List[CodeActAction]
    confidence: Optional[float]

class TrajectoryStep(BaseModel):
    turn_index: int
    action: CodeActAction
    observation_type: str          # "image" | "json" | "error"
    observation_summary: str
    observation_image_path: Optional[str]
    observation_data: Optional[Dict]
    elapsed_s: float

class Trajectory(BaseModel):
    trajectory_id: str
    image_path: str
    steps: List[TrajectoryStep]
    final_damage_map: List[DamagePartEntry]
    ...

# Tool call tracking
class ToolCallRecord(BaseModel):
    tool: str
    args_summary: str
    elapsed_s: float
    result_keys: List[str]
    success: bool

# Final orchestrator output
class DamagePartEntry(BaseModel):
    damage: str
    part: str
    severity: str
    cost_min: int    # INR
    cost_max: int    # INR

class DetectionWithBBox(BaseModel):
    index: int
    bbox: List[float]       # [x1, y1, x2, y2] pixel coords
    damage: str
    part: str
    severity: str
    confidence: float
    source: str             # "vlm" | "both" | "sam2" | "human"
    cost_min: int
    cost_max: int
    grounded: bool
    needs_review: bool

class FinalDamageReport(BaseModel):
    image_path: str
    damage_part_map: List[DamagePartEntry]
    detections_with_bbox: List[DetectionWithBBox]
    merged_detections: List[Dict[str, Any]]
    total_min: int
    total_max: int
    currency: str = "INR"
    approval_decision: str       # AUTO_APPROVED | ESCALATE_TO_HUMAN | UNKNOWN
    tool_call_log: List[ToolCallRecord]
    iterations: List[Dict[str, Any]]
    total_inference_s: float
    warnings: List[str]
    raw_vlm_response: Optional[str]
    annotated_image_path: Optional[str]
    merged_image_path: Optional[str]
    masked_image_path: Optional[str]

# HITL
class SessionState(BaseModel):
    session_id: str
    status: str              # pending_review | approved | rejected
    report: FinalDamageReport
    created_at: str
    claim_id: Optional[str]
    job_id: Optional[str]
    corrected_map: Optional[List[DamagePartEntry]]
    correction_notes: Optional[str]
```

---

## VLM Tool Calling — Implementation Details

### Model

`qwen3.5:9b` via Ollama. The agent outputs a JSON object with an `actions` array.
The response parser in `pi_agent.py` uses bracket-matching (not `<tool_call>` tags).
If JSON cannot be parsed after 3 recovery strategies + 2 retries, the iteration is
abandoned and the loop continues or terminates.

### Tool Definitions (active — in `pi_agent.py`)

| Tool | Arguments | Returns | Notes |
|---|---|---|---|
| `run_damage_detection` | `{reason: str}` | Annotated image + detection list | Second VLM call at 1024px. Populates `_vlm_detections` side-channel for approval. |
| `zoom_region` | `{bbox: [x1,y1,x2,y2] pct, reason: str}` | Cropped image | Pure PIL — no Ollama call. Blocked if bbox > 80% of image. |
| `detect_part` | `{part_query: str, reason: str}` | Annotated part image | Third Ollama call at 640px. |
| `Terminate` | `{damage_items: [...]}` | Exits loop | Each item needs damage_type, part, severity from valid vocab. |

**`tool_registry.py` is dead code.** The old YOLO-backed tool registry is never
imported in the active pipeline. Do not use it or add to it.

### Valid Vocabulary (enforced in `pi_agent.py` and prompt)

```
damage_type: dent | scratch | crack | glass_shatter | lamp_broken | tire_flat |
             mirror_broken | paint_damage | scuff | bent | crumpled |
             missing_part | detached_part | wheel_damage | structural_damage

part:        front_bumper | rear_bumper | hood | grill |
             windshield | rear_windshield |
             left_fender | right_fender |
             front_left_door | front_right_door | rear_left_door | rear_right_door |
             front_left_window | front_right_window | rear_left_window | rear_right_window |
             roof_panel | trunk_lid | tailgate | quarter_panel |
             headlight | taillight | fog_lamp | side_mirror |
             wheel | tire | rocker_panel | radiator_support

severity:    minor | moderate | severe
```

If the VLM outputs a value outside these sets on Terminate, the turn is rejected
and the VLM is asked to retry (up to `max_retry = 2`).

### Loop Behavior

```
max_iterations = 6        ← hard cap from config (vlm.max_iterations)
max_retries    = 2        ← JSON parse / policy violation recovery per iteration
wall_timeout   = 480s     ← per full run
```

**VLM sampling (set in `configs/global_config.yaml` `vlm:`):**
```
thinking:         false   ← CRITICAL. Thinking mode spends the whole token budget
                            on <think> and never emits the JSON action.
temperature:      0.7
top_p:            0.8
top_k:            20
presence_penalty: 1.5     ← curbs endless-repetition failure mode
num_ctx:          8192    ← images cost ~1700 tok each; default ~4096 evicts system prompt
```

**Token budgets per call type:**
```
Main agent loop:        1024 tokens  (vlm.max_new_tokens_tool)
run_damage_detection:    900 tokens  (hardcoded in _vlm_damage_detection)
detect_part:             400 tokens  (hardcoded in _vlm_detect_part)
```

**Image sizes per call type:**
```
Main agent loop:        640px  (vlm.image_max_dim)
run_damage_detection:  1024px  (vlm.image_detect_dim) — tighter bbox localization
detect_part:            640px
```

History must NOT contain thinking content (`pi_agent` strips `<think>` blocks
before appending assistant messages to history).

If `max_iterations` is hit without Terminate, the agent salvages findings from
the `run_damage_detection` side-channel. If that is also empty, returns empty
damage_items. Always ESCALATE_TO_HUMAN in this case. Do not raise an exception —
a partial report is better than a 500 error.

### The Three Prompts (all in `pi_agent.py`)

1. **`CODEACT_SYSTEM_PROMPT`** — system message at the start of every run. Tells
   Qwen it is the sole brain, defines the JSON output format, lists all tools with
   arguments, gives anti-hallucination rules, lists valid vocabulary.

2. **`DAMAGE_DETECTION_PROMPT`** — used in `_vlm_damage_detection()` only.
   Stateless single call (no conversation history). Asks Qwen to find ALL damage
   regions and return `{"detections": [...]}` with tight bboxes.

3. **`_PART_DETECTION_PROMPT`** — used in `_vlm_detect_part()` only. Stateless
   single call. Asks Qwen to locate a specific part and return its bbox + condition.

Keep all three prompts in sync with the valid vocabulary lists above. If you add a
damage class or part, update BOTH the Python `VALID_*` frozensets AND the prompt
text — they must always match.

---

## COST_DB — `models/vlm_reasoning/cost_db.py`

Single source of truth for all pricing. Both `orchestrator.py` (Stage 3) and
`backend/app/services/cost.py` (HITL recalculation) import from here.

Structure: `COST_DB[damage_class][part_label] = (cost_min_inr, cost_max_inr)`

All values are tuples of INR integers. Part labels use UNDERSCORE format to match
the VLM vocabulary (`front_bumper`, not `front bumper`).

`lookup_cost(damage, part)` normalises keys (lowercase, `_` = space) and falls
back to `(3000, 8000)` for unknown pairs.

`apply_cost_lookup()` in `services/cost.py` additionally applies severity
multipliers (minor=0.6×, moderate=1.0×, severe=1.6×) for fallback cases — used
in HITL recalculation only, NOT in the main pipeline.

**`sandbox.py` is dormant.** It imports COST_DB but `execute_cost_computation`
is never called in the active pipeline. Cost goes directly to `lookup_cost()`.

---

## Segmentation — `shared/sam_mask.py`

SAM2 runs in the backend only (never in the VLM reasoning path).

Called from `pipeline/orchestrator.py` at Stage 4:
- `_sam2_damage()` — SAM2 prompted by VLM damage boxes
- `_sam2_masked_overlay()` — SAM2 prompted by merged (VLM ∪ SAM2) boxes

**Weights search order:**
1. `configs/global_config.yaml` → `part_segmentation.sam2.weights_path`
2. `models/damage_detection/models/sam2.1_b.pt`
3. `weights/sam2.1_hiera_base_plus.pt`

**Degradation chain (best-effort, never raises):**
```
SAM2 masks  →  GrabCut within bbox  →  bbox outline only
```

If weights are missing, `_sam_failed = True` is set once and stays True for the
entire server process. Server restart required after fixing weights path.

Both `_sam2_damage()` and `_sam2_masked_overlay()` return `([], None)` / `None`
on any failure — caller falls back gracefully.

---

## Context Management — Multi-Turn

`pipeline/context_manager.py` implements the sliding window for follow-up
questions on an existing claim. **Not active in the current pipeline.**

```
PINNED   → system prompt + vehicle ID + damage summary. Never dropped.
RETAINED → last 3 message pairs + tool results.
DROPPED  → older turns, compressed to one-line summaries.
```

---

## Database Schema

Tables auto-created on startup via `Base.metadata.create_all()`.

| Table | Purpose |
|---|---|
| `users` | Auth accounts |
| `sessions` | Auth tokens |
| `vehicles` | Vehicle records (JSONB payload) |
| `claims` | Claim records (JSONB payload) |
| `insurance_claims` | Insurance records (JSONB payload) |
| `user_settings` | Per-user settings (JSONB) |
| `claim_images` | All pipeline images as BYTEA — original, annotated, masked, merged |

Images are stored in `claim_images` as BYTEA. Temp files on disk are deleted after
persistence. No filesystem dependency for images after job completion.

---

## In-Memory State — `backend/app/state.py`

```python
jobs:     dict[str, dict]             # job_id → {status, result?, error?, ...}
sessions: dict[str, SessionState]     # session_id → SessionState (ESCALATE only)
```

**Both are lost on server restart.** Sessions have no DB backing — pending HITL
reviews are dropped if the server restarts. Known limitation, post-MVP fix.

---

## HITL Flow

Triggered when `approval_decision = "ESCALATE_TO_HUMAN"`.

1. `SessionState` created in `state.sessions`
2. Dashboard loads session via `GET /session/{id}`
3. Human reviews, optionally corrects via:
   - `POST /session/{id}/update_detections` — update bbox list in-place
   - `POST /session/{id}/save_correction` — persist diff + YOLO labels
4. `POST /session/{id}/approve` → recalculate costs, write `feedback_log.jsonl`

Correction data written to:
- `data/feedback/feedback_log.jsonl` — HITL approvals (few-shot injection source)
- `data/feedback/corrections_log.jsonl` — per-item diffs + bbox annotations
- `data/yolo_corrections/` — YOLO-format labels for future fine-tuning

---

## Configs — `configs/global_config.yaml`

Single source of truth. All paths, thresholds, and model parameters live here.
No hardcoded paths anywhere in Python code (except SAM2 fallback search paths).
Load with `yaml.safe_load()`.

Required top-level keys:

```yaml
vlm:
  model_id: "qwen3.5:9b"
  ollama_base_url: "http://localhost:11434"
  thinking: false                         # MUST be false
  temperature: 0.7
  top_p: 0.8
  top_k: 20
  presence_penalty: 1.5
  num_ctx: 8192
  max_new_tokens_tool: 1024
  max_new_tokens_final: 1024
  max_iterations: 6
  codeact_max_retries: 2
  image_max_dim: 640
  image_detect_dim: 1024

approval:
  auto_approve_threshold_inr: 50000

part_segmentation:
  sam2:
    weights_path: "models/damage_detection/models/sam2.1_b.pt"
    device: "cuda"

database:
  url: "postgresql://localhost:5432/veh_dmg_db"
```

---

## Known Issues (as of 2026-06-17)

| # | Issue | Location |
|---|---|---|
| 1 | Approval bypass: VLM skipping `run_damage_detection` → `AUTO_APPROVED` with zero corroboration | `orchestrator.py:534` |
| 2 | `tool_registry.py` is dead code, contradicts active system | `models/vlm_reasoning/tool_registry.py` |
| 3 | No NMS/dedup on Terminate items — overlapping boxes inflate cost | `orchestrator.py` Stage 2 |
| 4 | Trajectory saved before image persistence — stale file paths in JSON | `orchestrator.py:563` |
| 5 | `state.sessions` in-memory only — HITL reviews lost on restart | `backend/app/state.py` |
| 6 | `severity` has no effect on pricing in the main pipeline (only in HITL recalc) | `orchestrator.py` Stage 3 |
| 7 | SAM2 `_sam_failed` flag is sticky — server restart needed after fixing weights | `shared/sam_mask.py` |

---

## Git — Rules

- `damage-detection` branch is the active integration branch
- `main` is clean at all times — only merge when end-to-end demo works
- All `.pt` files must be tracked via Git LFS before committing
  ```bash
  git lfs track "*.pt"
  git lfs track "*.pth"
  git add .gitattributes
  ```
- Never commit test images to `examples/` if they are > 5MB
- Never commit API keys, credentials, or database URLs — use `.env` + `python-dotenv`

---

## What Is Built vs In Progress vs Not Started

| Component | Status |
|---|---|
| VLM brain loop (PiAgent) | ✅ Built — `models/vlm_reasoning/pi_agent.py` |
| Ollama client | ✅ Built — `models/vlm_reasoning/ollama_client.py` |
| COST_DB + lookup_cost() | ✅ Built — `models/vlm_reasoning/cost_db.py` |
| Pipeline orchestrator (5-stage) | ✅ Built — `pipeline/orchestrator.py` |
| All Pydantic schemas | ✅ Built — `pipeline/schema.py` |
| FastAPI backend + all routers | ✅ Built — `backend/app/` |
| PostgreSQL schema + ORM models | ✅ Built — `backend/app/models.py` |
| SAM2 mask overlay | ✅ Built — `shared/sam_mask.py` |
| Merged union (VLM ∪ SAM2) | ✅ Built — `orchestrator._merge_union` |
| HITL session flow | ✅ Built — `backend/app/routers/sessions.py` |
| Feedback + correction logging | ✅ Built — `services/feedback.py`, JSONL files |
| YOLO label writer (HITL) | ✅ Built — `services/imaging.py` |
| Trajectory persistence | ✅ Built — `data/trajectories/raw/` |
| YOLOv8 damage detection | 🗄️ RETIRED — `models/_archive/best.pt` |
| Grounding DINO + SAM2 parts | 🔄 DORMANT — `models/part_segmentation/` |
| Plate/RC detection | 🔄 In progress — not integrated |
| Streamlit dashboard | ❌ Not started |
| Multi-turn context manager | ❌ Not active |
| Session persistence to DB | ❌ Not started (post-MVP) |

---

## Things Not to Build

These are explicitly out of scope for this MVP:

- Temporal workflow engine
- API Gateway (Kong, NGINX as dedicated service)
- Notification service (email, SMS, push)
- Vector store (Pinecone, Weaviate)
- Depth estimation model
- MLflow model registry
- Go/gRPC CV gateway
- Keycloak SSO / RBAC
- Multi-tenant support
- Docker multi-service Compose
- Kafka / audit logger
- Any paid API (OpenAI, Anthropic API, Google Vision, AWS Rekognition)
- `execute_cost_computation` as a VLM tool (cost is backend-only)

---

## Things That Will Go Wrong and How to Handle Them

**VLM outputs malformed JSON**
`_parse_codeact_turn()` tries 3 recovery strategies then 2 retries with error
feedback. If all fail, the iteration is skipped. If no Terminate fires,
fallback to `_vlm_detections`. Log a warning. Do not crash.

**SAM2 weights missing**
`_load_sam()` sets `_sam_failed = True`. All mask calls degrade to GrabCut then
bbox outline. `_sam2_damage()` returns `([], None)`. Pipeline continues — masks
are UI-only, not a decision input. Restart server after fixing weights.

**VLM loop hits `max_iterations`**
Salvage from `_vlm_detections` side-channel. If empty, return empty damage_items.
Set `approval_decision = ESCALATE_TO_HUMAN`. Return 200 with partial report.

**First request takes 60+ seconds (VLM cold load)**
Expected. The `/health` startup event pre-warms the model. Add a "Processing..."
state to the dashboard. Do not add a client timeout shorter than 90 seconds.

**Vocabulary mismatch between prompts and VALID_* sets**
If you add a damage class or part, you MUST update BOTH the Python `VALID_*`
frozensets in `pi_agent.py` AND the prompt text in `CODEACT_SYSTEM_PROMPT` and
`DAMAGE_DETECTION_PROMPT`. If they diverge, Qwen cannot use the new class
(prompt doesn't mention it) but validation rejects it if Qwen guesses it anyway.

---

## How to Run the Pipeline (Current State)

```bash
# 1. Activate env
source .venv/bin/activate

# 2. Start Ollama and pull model
ollama serve
ollama pull qwen3.5:9b

# 3. Run backend (from repo root)
uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload

# 4. Health check (triggers VLM warmup)
curl http://localhost:8000/health

# 5. Submit image
curl -X POST http://localhost:8000/assess \
  -F "image=@data/examples/test_car.jpg" \
  -F "claim_id=CLAIM_001"

# 6. Poll for result
curl http://localhost:8000/job/<job_id>
```

---

## Definition of MVP Done

The MVP is done when all of the following are true:

1. A user can upload a single car damage photo via the dashboard or API
2. The system returns a `FinalDamageReport` JSON with at least one
   `DamagePartEntry` and a cost range in INR
3. The `approval_decision` field is populated (`AUTO_APPROVED` or `ESCALATE_TO_HUMAN`)
4. The full pipeline completes without crashing on the 10 test images in `data/examples/`
5. The `damage-detection` branch passes a PR review with all 6 team members

Nothing else is required for MVP. Resist any addition to this list.
