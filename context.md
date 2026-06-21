# Project Context — veh_dmg_detection
> Complete reference for the whole project. Updated 2026-06-16.
> All paths are relative to `veh_dmg_detection/`.

---

## What This Project Does

A user uploads a photo of a damaged vehicle. The system:
1. Runs a VLM agent (Qwen3.5:9b) to identify all damage — what type, which part, how severe, where (bbox).
2. Prices each damage deterministically using a hardcoded cost table (COST_DB in INR).
3. Runs SAM2 to generate tight segmentation masks over the VLM's damage boxes.
4. Produces a `FinalDamageReport` with damage map, cost totals, annotated images, and an AUTO_APPROVED / ESCALATE_TO_HUMAN decision.
5. If escalated, stores the report in a HITL session for human correction via the dashboard.

**Stack:** Python 3.10 · FastAPI · PostgreSQL · Ollama (Qwen3.5:9b) · SAM2 (ultralytics) · PIL · OpenCV

---

## Repo Structure

```
veh_dmg_detection/
├── CLAUDE.md                          ← project rules and architecture decisions
├── context.md                         ← this file
├── configs/
│   └── global_config.yaml             ← single source of truth for all params
├── models/
│   ├── vlm_reasoning/
│   │   ├── pi_agent.py                ← VLM agentic loop (THE brain)
│   │   ├── tool_registry.py           ← DEAD CODE (old design, never called)
│   │   ├── sandbox.py                 ← CodeAct exec sandbox (currently unused)
│   │   ├── cost_db.py                 ← COST_DB pricing table + lookup_cost()
│   │   └── ollama_client.py           ← stdlib Ollama REST wrapper
│   ├── damage_detection/              ← RETIRED YOLOv8 (archived, not called)
│   ├── part_segmentation/             ← Grounding DINO + SAM2 (dormant)
│   ├── vehicle_detection/             ← stock YOLO ROI (dormant)
│   └── plate_rc_detection/            ← plate OCR (not integrated)
├── pipeline/
│   ├── orchestrator.py                ← main pipeline entry point
│   ├── schema.py                      ← all Pydantic models
│   ├── context_manager.py             ← multi-turn sliding window (not active)
│   ├── trajectory_filter.py           ← SFT data filtering
│   └── feedback_reader.py             ← reads corrections_log for few-shot
├── backend/
│   └── app/
│       ├── main.py                    ← FastAPI app factory + lifespan
│       ├── state.py                   ← in-memory jobs + sessions dicts
│       ├── db.py                      ← SQLAlchemy engine + SessionLocal
│       ├── models.py                  ← ORM: User, Session, Vehicle, Claim, ClaimImage, etc.
│       ├── auth.py                    ← token auth helpers
│       ├── schemas.py                 ← request/response Pydantic models for routers
│       ├── core/
│       │   └── config.py              ← paths, constants, load_config(), CORS
│       ├── routers/
│       │   ├── assessment.py          ← POST /assess, GET /job/{id}
│       │   ├── sessions.py            ← HITL: GET/POST /session/{id}/*
│       │   ├── images.py              ← image serving from DB (BYTEA)
│       │   ├── feedback.py            ← POST /recalculate, GET /feedback/stats
│       │   ├── health.py              ← GET /health, GET /
│       │   ├── accounts.py            ← user auth (register/login)
│       │   ├── vehicles.py            ← vehicle CRUD
│       │   ├── claims.py              ← claim CRUD
│       │   ├── insurance.py           ← insurance claim CRUD
│       │   └── settings.py            ← per-user settings
│       └── services/
│           ├── assessment.py          ← async job runner
│           ├── cost.py                ← apply_cost_lookup() with severity fallback
│           ├── imaging.py             ← generate_annotated_image(), write_yolo_labels()
│           └── feedback.py            ← write_feedback() to feedback_log.jsonl
├── shared/
│   ├── sam_mask.py                    ← SAM2 mask overlay helper (ultralytics)
│   ├── image_utils.py
│   ├── bbox_canvas.py
│   └── logger.py
├── dashboard/
│   └── app.py                         ← Streamlit dashboard (not started)
├── data/
│   ├── uploads/                       ← temp upload files (deleted after job)
│   ├── trajectories/raw/              ← trajectory JSONs for SFT
│   ├── feedback/feedback_log.jsonl    ← HITL approvals log
│   ├── feedback/corrections_log.jsonl ← per-item correction records
│   └── yolo_corrections/              ← YOLO fine-tune samples from HITL
└── scripts/
    ├── sft_train.py                   ← SFT training on trajectory data
    └── prepare_sft_dataset.py
```

---

## Active Models (in the running pipeline)

| Model | Where | Purpose |
|---|---|---|
| **Qwen3.5:9b** | Ollama at localhost:11434 | VLM brain — all damage detection, part naming, severity, bbox estimation |
| **SAM2** (`sam2.1_b.pt`) | `shared/sam_mask.py` via ultralytics | Tight segmentation masks over VLM damage boxes (UI only, best-effort) |

**Retired / dormant (on disk, not called):**
- `models/damage_detection/best.pt` — trained YOLOv8 (archived)
- `models/vehicle_detection/` — stock YOLO ROI detector
- `models/part_segmentation/` — Grounding DINO + SAM2 part segmentation
- `models/plate_rc_detection/` — plate OCR (not integrated)

---

## Where PiAgent Is Used

**`PiAgent` is instantiated and called in exactly ONE place:**

```
pipeline/orchestrator.py  →  run()  →  line 463-464

    from models.vlm_reasoning.pi_agent import PiAgent
    agent    = PiAgent(config)
    loop_out = agent.run(image_path=image_path, trajectory_steps=trajectory_steps)
```

That `orchestrator.run()` is called from exactly ONE place:

```
backend/app/services/assessment.py  →  run_assessment_job()  →  line 50

    from pipeline.orchestrator import run as orchestrator_run
    report = await asyncio.wait_for(
        run_in_threadpool(orchestrator_run, str(save_path.resolve()), config, claim_metadata),
        timeout=JOB_TIMEOUT_S,
    )
```

That `run_assessment_job()` is triggered from exactly ONE place:

```
backend/app/routers/assessment.py  →  POST /assess  →  line 96

    asyncio.create_task(
        run_assessment_job(job_id, Path(tmp_path), cfg, claim_id, vehicle_id)
    )
```

**Full call chain:**
```
POST /assess
  └─ run_assessment_job()          [services/assessment.py]
       └─ orchestrator.run()       [pipeline/orchestrator.py]
            └─ PiAgent.run()       [models/vlm_reasoning/pi_agent.py]
                 └─ ollama_chat()  [models/vlm_reasoning/ollama_client.py]
```

PiAgent is **not** used anywhere else. There is no batch mode, no direct script invocation in the active path, no other router calling it.

---

## End-to-End Pipeline

### Stage 0 — Image intake (`routers/assessment.py`)
- Validate MIME type (jpeg/png/webp only)
- Write image bytes to temp file (pipeline needs a file path)
- Save original image BYTEA to `claim_images` DB table immediately
- Create job entry in `state.jobs` with status `"queued"`
- Fire `asyncio.create_task(run_assessment_job(...))` — non-blocking
- Return `{job_id, status: "processing"}` immediately

### Stage 1 — PiAgent VLM brain loop (`pi_agent.py`)

Conversation starts: system prompt (`CODEACT_SYSTEM_PROMPT`) + vehicle image (base64, resized to 640px).

**Loop** (up to `max_iterations=6`, wall-clock cap 480s):
1. Call Ollama → raw string
2. Parse into `CodeActTurn` (3-strategy JSON recovery: bracket-match → repair → regex)
3. Canonicalize tool names (case/space insensitive)
4. Validate turn policy (vocab on Terminate, non-empty actions otherwise)
5. On failure: retry up to `max_retry=2` with error feedback injected into conversation
6. Execute each action via `_dispatch_action()`:
   - `run_damage_detection` → second Ollama call (1024px, 900 tokens) → annotated image + detection list
   - `zoom_region` → PIL crop (+12% padding, upscale to 320–512px)
   - `detect_part` → Ollama call (640px, 400 tokens) → annotated part image
   - `Terminate` → extract `damage_items`, exit loop immediately
7. Append observation (image or error) as new user message, repeat

**Returns to orchestrator:**
```python
{
  "damage_items":         list,   # from Terminate: [{damage_type, part, severity, confidence, bbox_pct}]
  "vlm_detections":       list,   # side-channel from run_damage_detection: [{class, confidence, bbox, bbox_pct, part, severity}]
  "annotated_image_path": str,    # PIL-annotated image from run_damage_detection
  "tool_calls":           int,
  "warnings":             list,
  "raw_vlm_response":     str,    # last Ollama raw response
}
```

**Fallback if Terminate never fires:** salvages damage items from `_vlm_detections` (run_damage_detection side-channel). If that's also empty, returns empty list. Always escalates in this case.

### Stage 2 — Build detections_with_bbox (`orchestrator.py`)
- For each `damage_item` in Terminate output: `bbox_pct → pixels` via `_pct_to_px()`
- Clamps to [0,100], fixes corner ordering, enforces minimum 6% frame size
- Calls `lookup_cost(damage_type, part)` → per-item cost for the `DetectionWithBBox`

### Stage 3 — Backend cost (deterministic, no LLM)
- For each `damage_item`: `lookup_cost(damage_type, part)` → `DamagePartEntry(cost_min, cost_max)`
- Sums to `total_min`, `total_max`
- COST_DB is the only pricing source — hardcoded INR tuples in `cost_db.py`
- `lookup_cost()` normalizes keys (lowercase, underscores = spaces), falls back to (3000, 8000)
- `apply_cost_lookup()` in `services/cost.py` (used by HITL recalc) additionally applies severity multipliers: minor=0.6×, moderate=1.0×, severe=1.6× (only for unknown (damage, part) pairs that fall back to class average)

### Stage 4 — SAM2 + merged union (`orchestrator.py`)
- `_sam2_damage()`: runs SAM2 (ultralytics) prompted by VLM damage boxes → tight mask boxes
- `_merge_union()`: tags each box as `"vlm"` / `"both"` / `"sam2"` by IoU >= 0.3
- `_sam2_masked_overlay()`: draws SAM2 masks from merged boxes (fallback to VLM-prompted overlay)
- Everything is best-effort — returns `([], None)` on any failure, silently

### Stage 5 — Images, approval, report (`orchestrator.py`)
- `_draw_boxes()` → annotated image (class-coloured) + merged image (source-coloured)
- **Approval gate** (corroboration — NOT cost):
  - `det_classes` = damage classes seen in `run_damage_detection` pass
  - `uncorroborated` = final Terminate classes NOT in `det_classes`
  - If no `vlm_damage_items` → ESCALATE
  - If `uncorroborated` not empty → ESCALATE (with warning)
  - Otherwise → AUTO_APPROVED
  - ⚠️ BUG: if VLM never called `run_damage_detection`, `det_classes` is empty set, short-circuit makes `uncorroborated=[]` → AUTO_APPROVED with zero corroboration
- `_build_iterations()` → compact UI log per tool call
- `_save_trajectory()` → `data/trajectories/raw/<uuid>.json`
- `FinalDamageReport(**...).model_dump()` returned

### Stage 6 — Persistence (`services/assessment.py`)
- `_persist_generated_images()`: reads annotated/masked/merged JPEG files → writes BYTEA rows to `claim_images` → deletes temp files
- If ESCALATE_TO_HUMAN: creates `SessionState` entry in `state.sessions`
- Updates `state.jobs[job_id]` to `"complete"` with full report

---

## API Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/` | Service banner |
| `GET` | `/health` | VLM warmup status |
| `POST` | `/assess` | Submit image → returns `{job_id}` |
| `GET` | `/job/{id}` | Poll status / get result |
| `GET` | `/job/{id}/iterations` | Tool call log for UI |
| `GET` | `/job/{id}/plain_image` | Original upload from DB |
| `GET` | `/job/{id}/annotated_image` | Damage boxes annotated |
| `GET` | `/job/{id}/masked_image` | SAM2 mask overlay |
| `GET` | `/job/{id}/merged_image` | VLM ∪ SAM2 merged boxes |
| `GET` | `/session/{id}` | HITL session state |
| `POST` | `/session/{id}/approve` | Human approval with corrections |
| `POST` | `/session/{id}/update_detections` | Update bbox detections in-place |
| `POST` | `/session/{id}/save_correction` | Save full correction record + YOLO labels |
| `POST` | `/recalculate` | Recompute costs without re-running pipeline |
| `GET` | `/feedback/stats` | Aggregated correction stats |
| `POST` | `/api/feedback` | Direct feedback log write |
| `POST` | `/register` / `GET /login` | User auth |
| `CRUD` | `/vehicles`, `/claims`, `/insurance` | Persistence routers |
| `CRUD` | `/settings` | Per-user settings |

---

## Database Schema (PostgreSQL)

All tables auto-created on startup via `Base.metadata.create_all()`.

| Table | Key columns | Notes |
|---|---|---|
| `users` | `id` (uuid), `name`, `phone` (unique), `password_hash` | Auth |
| `sessions` | `token` (PK), `user_id` | Auth tokens |
| `vehicles` | `id`, `client_id`, `user_id`, `data` (JSONB) | Vehicle records |
| `claims` | `id`, `client_id`, `user_id`, `data` (JSONB) | Claim records |
| `insurance_claims` | `id`, `client_id`, `user_id`, `data` (JSONB) | Insurance records |
| `user_settings` | `user_id` (PK), `data` (JSONB) | Per-user settings |
| `claim_images` | `id`, `job_id`, `image_type`, `mime_type`, `data` (BYTEA) | All images from pipeline |

`claim_images.image_type` values: `"original"` `"annotated"` `"masked"` `"merged"`

Connection string from `DATABASE_URL` env var, defaults to `postgresql+psycopg2://localhost:5432/veh_dmg_db`.

---

## In-Memory State (`backend/app/state.py`)

```python
jobs: dict[str, dict]             # job_id → {status, result?, error?, started_at?, ...}
sessions: dict[str, SessionState] # session_id → SessionState (ESCALATE_TO_HUMAN only)
```

Both are lost on server restart. Sessions have no DB backing — a restart drops all pending HITL reviews.

---

## Pydantic Schema Map (`pipeline/schema.py`)

| Schema | Purpose |
|---|---|
| `CodeActAction` | One action in a VLM turn: `{name, arguments}` |
| `CodeActTurn` | One VLM output: `{thought, uncertainty, actions, confidence}` |
| `TrajectoryStep` | One (action, observation) pair in the agent loop |
| `Trajectory` | Full tool-use trajectory for one image (SFT training data) |
| `DamageItem` | Legacy schema (old pipeline, still in schema.py, not used in active path) |
| `ClaimAnalysisSchema` | Legacy schema (old pipeline) |
| `DamagePartEntry` | `{damage, part, severity, cost_min, cost_max}` — one item in the final map |
| `DetectionWithBBox` | `{index, bbox, damage, part, severity, confidence, source, cost_min, cost_max, grounded, needs_review, anomaly_score}` — full spatial detection |
| `RegionEvidence` | SAM2/YOLO region with no damage label (grounding anchor) |
| `FinalDamageReport` | Top-level pipeline output (all fields, see below) |
| `SessionState` | HITL session: report + correction state |
| `ToolCallRecord` | Per-tool log entry in the final report |
| `FeedbackEntry` | Written to `feedback_log.jsonl` on HITL approval |
| `CorrectionEntry` | Written to `corrections_log.jsonl` on save_correction |
| `CorrectionAction` | One diff action: `keep \| edit \| remove \| add` |
| `BBoxAnnotation` | Human-drawn bbox for missed damage → YOLO label |
| `BBoxCorrectionRequest` | Request body for `/session/{id}/update_detections` |
| `SaveCorrectionRequest` | Request body for `/session/{id}/save_correction` |
| `RecalculateRequest/Response` | Request/response for `/recalculate` |
| `ApproveRequest` | Request body for `/session/{id}/approve` |
| `FewShotExample` | VLM prompt injection from correction history |

---

## FinalDamageReport Fields

```python
image_path:           str                    # preprocessed image path
damage_part_map:      List[DamagePartEntry]  # from Terminate items + COST_DB
detections_with_bbox: List[DetectionWithBBox]# same items with pixel bbox + per-item costs
merged_detections:    List[dict]             # VLM ∪ SAM2 union, source-tagged
total_min:            int                    # INR
total_max:            int                    # INR
currency:             str = "INR"
approval_decision:    str                    # AUTO_APPROVED | ESCALATE_TO_HUMAN | UNKNOWN
tool_call_log:        List[ToolCallRecord]   # one per TrajectoryStep
iterations:           List[dict]             # compact UI log: {turn, tool, reason, summary, elapsed_s, ok}
total_inference_s:    float
warnings:             List[str]              # deduped via dict.fromkeys
raw_vlm_response:     Optional[str]          # last Ollama content string
annotated_image_path: Optional[str]          # temp path (deleted after DB persist)
merged_image_path:    Optional[str]          # temp path (deleted after DB persist)
masked_image_path:    Optional[str]          # temp path (deleted after DB persist)
```

---

## Pricing — COST_DB (`models/vlm_reasoning/cost_db.py`)

Single source of truth. Structure: `COST_DB[damage_class][part_label] = (cost_min_inr, cost_max_inr)`.

Active damage classes: `dent | scratch | crack | glass_shatter | lamp_broken | tire_flat`

Active part labels: `front_bumper | rear_bumper | hood | windshield | rear_windshield | front_left_door | front_right_door | rear_left_door | rear_right_door | left_fender | right_fender | trunk_lid | roof_panel | headlight | taillight | tire`

`lookup_cost(damage, part)` normalizes keys (lowercase, `_` = space), falls back to `(3000, 8000)` for unknown pairs.

`apply_cost_lookup()` in `services/cost.py` additionally applies severity multipliers (minor=0.6×, moderate=1.0×, severe=1.6×) for fallback cases — used in HITL recalculation only.

---

## VLM Tools (pi_agent.py — active)

| Tool | Arguments | What happens | Side effects |
|---|---|---|---|
| `run_damage_detection` | `{reason: str}` | Second Ollama call at 1024px. Returns annotated image + detection list. | Populates `_vlm_detections` side-channel (used for approval + annotated image fallback) |
| `zoom_region` | `{bbox: [x1,y1,x2,y2] pct, reason: str}` | PIL crop + upscale to 320–512px. Blocked if bbox > 80% of image. | None |
| `detect_part` | `{part_query: str, reason: str}` | Ollama call at 640px. Draws blue box on found part. | None |
| `Terminate` | `{damage_items: [...]}` | Exits loop. Each item needs `damage_type`, `part`, `severity` from valid vocab. | Saves TrajectoryStep, returns to orchestrator |

**Dead tool definitions** in `tool_registry.py` (never called): `run_damage_detection` (YOLO), `run_part_segmentation`, `execute_cost_computation`, `run_plate_detection`.

---

## Ollama Config (`configs/global_config.yaml` → `vlm:`)

| Param | Value | Why |
|---|---|---|
| `model_id` | `qwen3.5:9b` | The VLM brain |
| `thinking` | `false` | Thinking mode burns the token budget on `<think>` and never emits the JSON action |
| `temperature` | `0.7` | Instruct-mode sampling |
| `top_p` | `0.8` | Qwen3.5 best-practices |
| `top_k` | `20` | Qwen3.5 best-practices |
| `presence_penalty` | `1.5` | Curbs endless-repetition failure mode |
| `num_ctx` | `8192` | Images cost ~1700 tokens each; default ~4096 evicts system prompt mid-run |
| `max_iterations` | `6` | Hard agent loop cap |
| `max_new_tokens_tool` | `1024` | Per tool-call turn |
| `max_new_tokens_final` | `1024` | Final synthesis |
| `image_max_dim` | `640` | Main agent image resize |
| `image_detect_dim` | `1024` | run_damage_detection sub-call resize |

---

## HITL (Human-in-the-Loop) Flow

Triggered when `approval_decision = "ESCALATE_TO_HUMAN"`.

1. `run_assessment_job` creates a `SessionState` in `state.sessions`
2. Dashboard calls `GET /session/{id}` to load the report
3. Dashboard calls `GET /session/{id}/annotated_image` for the damage image
4. Human reviews detections, optionally:
   - `POST /session/{id}/update_detections` → update bbox list in-place, regenerate annotated image
   - `POST /session/{id}/save_correction` → persist full diff record + YOLO labels for fine-tuning
5. `POST /session/{id}/approve` → recalculate costs, mark `"approved"`, write `feedback_log.jsonl`

Correction data written to:
- `data/feedback/feedback_log.jsonl` — HITL approval records (for few-shot injection)
- `data/feedback/corrections_log.jsonl` — per-item diffs + bbox annotations (for SFT data)
- `data/yolo_corrections/` — YOLO-format label files (for future fine-tuning)

---

## Approval Logic

```
det_classes = {d["class"] for d in vlm_detections}   # from run_damage_detection
uncorroborated = [
    item["damage_type"] for item in vlm_damage_items
    if det_classes and item["damage_type"] not in det_classes
]

if not vlm_damage_items:          → ESCALATE_TO_HUMAN
elif uncorroborated:              → ESCALATE_TO_HUMAN
else:                             → AUTO_APPROVED
```

⚠️ Known bug: if `run_damage_detection` was never called, `det_classes` is empty, the `if det_classes and ...` short-circuits, `uncorroborated=[]`, result is `AUTO_APPROVED` with zero corroboration.

---

## SAM2 Degradation Chain

`shared/sam_mask.py`:
1. Try SAM2 (ultralytics) with prompted boxes
2. Fallback: GrabCut within the bbox
3. Last resort: bbox outline only (no fill)

`orchestrator._sam2_damage()` / `_sam2_masked_overlay()`:
- Returns `([], None)` on any failure — no error surfaced to report

---

## Key Known Issues

| # | Issue | Severity | Location |
|---|---|---|---|
| 1 | Approval bypass: VLM skip of `run_damage_detection` → AUTO_APPROVED with no corroboration | High | `orchestrator.py:534` |
| 2 | `tool_registry.py` is dead code, contradicts active system | Medium | `models/vlm_reasoning/tool_registry.py` |
| 3 | CLAUDE.md tool table lists old tools, not active ones | Medium | `CLAUDE.md` |
| 4 | No NMS/dedup on Terminate items — overlapping boxes inflate cost | Medium | `orchestrator.py` Stage 2 |
| 5 | Trajectory saved before image persistence — stale file paths in JSON | Medium | `orchestrator.py:563-573` |
| 6 | `state.sessions` in-memory only — HITL reviews lost on restart | Medium | `backend/app/state.py` |
| 7 | `sandbox.py` / `execute_cost_computation` unreachable in active path | Low | `models/vlm_reasoning/sandbox.py` |
| 8 | `severity` has no effect on pricing in the orchestrator path | Low | `orchestrator.py` Stage 3 |
| 9 | `_vlm_detect_part` returns `str` not `dict` — inconsistent with other tools | Low | `pi_agent.py:1212` |

---

## How to Run

```bash
# 1. Activate env
source .venv/bin/activate

# 2. Start Ollama and pull the model
ollama serve
ollama pull qwen3.5:9b

# 3. Start the backend (from repo root)
uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload

# 4. Health check (triggers VLM warmup)
curl http://localhost:8000/health

# 5. Submit image
curl -X POST http://localhost:8000/assess \
  -F "image=@data/examples/test_car.jpg"

# 6. Poll for result
curl http://localhost:8000/job/<job_id>
```

First request takes 60+ seconds (VLM cold load). Subsequent requests: 8–35s depending on tool calls.
