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

The VLM is not a post-processor. It is the orchestrator. It sees the image first,
decides which CV tools to invoke, calls them, receives structured results, reasons
over spatial overlap internally, generates Python cost computation code, executes
it in a sandbox, and returns the final report. Every CV model is a tool the VLM
can call or skip.

```
image
  │
  ▼
orchestrator.run()
  │
  ▼
VLM (Qwen2-VL-7B) ── sees raw image ── forms hypothesis
  │
  ├─ tool call → run_damage_detection(image_path)
  │     └─ YOLOv8 best.pt → damage bboxes JSON → returned to VLM
  │
  ├─ tool call → run_part_segmentation(image_path)
  │     └─ Grounding DINO → SAM2.1 → part masks JSON → returned to VLM
  │
  ├─ VLM reasons: damage bbox ∩ part bbox → damage-to-part map
  │   (spatial reasoning, not OpenCV math)
  │
  ├─ tool call → execute_cost_computation(python_code)
  │     └─ sandbox.py restricted exec → result dict → returned to VLM
  │
  └─ VLM synthesizes → FinalDamageReport JSON
        │
        └─ auto-approve gate → AUTO_APPROVED or ESCALATE_TO_HUMAN
```

### Why This Architecture

The old sequential pipeline (YOLOv8 → SAM2 → OpenCV intersection → VLM) failed
because: (1) OpenCV center-point intersection logic silently produced null mappings
on deformed geometry — a crumpled bumper mask has no clean center point, (2) all
three models ran on every image regardless of what was in it, (3) the VLM was
passive and had no ability to ask for more information or handle ambiguity.

The tool-calling architecture fixes (1) by replacing deterministic math with VLM
spatial reasoning, fixes (2) via conditional dispatch, and fixes (3) by making
the VLM the decision-making agent throughout.

The cost: VLM inference is 3–8s per forward pass. Total pipeline is 8–35s
depending on number of tool calls. This is acceptable for an MVP upload workflow.
It is not acceptable for batch or real-time. Do not let anyone add a batch
processing requirement without flagging this.

---

## Stack — Confirmed and Frozen

| Layer | Technology | Notes |
|---|---|---|
| Damage detection | YOLOv8 (`ultralytics`) | `best.pt`, 6 classes |
| Part segmentation | Grounding DINO + SAM2.1 | GDino SwinT + `sam2.1_hiera_base_plus.pt` |
| Plate/RC detection | YOLOv8-based | In progress, not yet integrated |
| VLM orchestrator | Qwen2-VL-7B-Instruct | Via `transformers` + `qwen-vl-utils` |
| Backend | FastAPI | Single app, not microservices |
| Database | PostgreSQL | Local instance |
| Image storage | Local filesystem | `outputs/`, not S3, not GCS |
| Dashboard | Streamlit | Not started |
| Python | 3.10 | Hard requirement throughout |

**Do not introduce:** microservices, Docker Compose multi-service, Celery,
Redis, Kafka, Temporal, Pinecone, Weaviate, Keycloak, any paid API, any cloud
storage. The technical magazine document describes an enterprise target state,
not this MVP.

---

## Repo Structure

```
veh_dmg_detection/
├── CLAUDE.md                        ← this file
├── models/
│   ├── damage_detection/            ← FROZEN. YOLOv8. Do not modify.
│   │   ├── __init__.py              ← exposes run()
│   │   └── best.pt                  ← tracked via Git LFS
│   ├── part_segmentation/           ← Grounding DINO + SAM2. In generation.
│   │   ├── __init__.py
│   │   ├── run.py                   ← exposes run()
│   │   ├── gdino_infer.py
│   │   ├── sam2_infer.py
│   │   ├── postprocess.py
│   │   ├── visualize.py
│   │   ├── ensure_weights.py
│   │   └── tests/
│   ├── plate_rc_detection/          ← In progress. Exposes run() when ready.
│   └── vlm_reasoning/               ← New architecture. Build this.
│       ├── tool_registry.py         ← tool definitions + dispatcher
│       └── sandbox.py               ← CodeAct restricted execution
├── pipeline/
│   ├── orchestrator.py              ← FULL REWRITE. VLM tool-calling loop.
│   ├── schema.py                    ← Pydantic contracts. Append only.
│   └── context_manager.py          ← Sliding window for Turn 2+ follow-ups.
├── backend/
│   └── app.py                       ← FastAPI. Partial rewrite needed.
├── dashboard/                        ← Streamlit. Not started.
├── shared/                           ← logging, utils. Touch only if needed.
├── configs/
│   ├── global_config.yaml           ← single source of truth for all paths/params
│   └── seg_config.yaml              ← Grounding DINO + SAM2 config
└── data/
    └── examples/                    ← test images. Do not commit large files.
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

This contract exists so any model can be swapped or mocked independently. Violating
it breaks the orchestrator's ability to treat models as interchangeable tools.

---

## Pipeline Schemas — `pipeline/schema.py`

Append to this file. Never overwrite it. Existing schemas:

```python
# Existing — part segmentation output
class PartSegment(BaseModel):
    part_label: str
    bbox: List[float]           # [x1, y1, x2, y2] absolute pixels
    mask_rle: Optional[str]
    gdino_confidence: float
    sam2_score: float
    area_px: int

class SegmentationOutput(BaseModel):
    image_path: str
    image_width: int
    image_height: int
    parts: List[PartSegment]
    total_parts_detected: int
    model_versions: Dict[str, str]
    warnings: List[str]
```

Schemas to add:

```python
# Tool call tracking
class ToolCallRecord(BaseModel):
    tool: str
    elapsed_s: float
    result_keys: List[str]

# Final orchestrator output
class DamagePartEntry(BaseModel):
    damage: str       # one of: dent, scratch, crack, glass_shatter, lamp_broken, tire_flat
    part: str         # one of: front_bumper, rear_bumper, hood, windshield,
                      #   rear_windshield, front_left_door, front_right_door,
                      #   rear_left_door, rear_right_door, left_fender, right_fender,
                      #   trunk_lid, roof_panel, left_headlight, right_headlight, tire
    severity: str     # minor, moderate, severe
    cost_min: int     # INR
    cost_max: int     # INR

class FinalDamageReport(BaseModel):
    image_path: str
    damage_part_map: List[DamagePartEntry]
    total_min: int
    total_max: int
    currency: str = "INR"
    approval_decision: str       # AUTO_APPROVED or ESCALATE_TO_HUMAN
    tool_call_log: List[ToolCallRecord]
    total_inference_s: float
    warnings: List[str]
    raw_vlm_response: Optional[str]   # keep during MVP for debugging
```

---

## Model Performance — Damage Detection

Training is complete. These numbers are the ceiling for the current dataset.
Do not attempt retraining on CarDD without new data.

| Class | Precision | Recall | Notes |
|---|---|---|---|
| glass_shatter | 0.98 | — | Strong. High confidence detections are reliable. |
| tire_flat | 0.92 | — | Strong. |
| lamp_broken | 0.84 | — | Strong. |
| dent | 0.61 | — | Weak. VLM must compensate for misses. |
| scratch | 0.62 | — | Weak. |
| crack | 0.55 | — | Weakest. Highest false negative rate. |
| **Overall** | **0.737** | **0.719** | **mAP50: 0.749** |

The VLM's role on weak classes: when damage_detection returns low-confidence or
no detections in a region the VLM visually identifies as damaged, the VLM should
note this discrepancy in warnings and base its assessment on visual reasoning.
It should not blindly trust YOLOv8 on dents, scratches, and cracks.

---

## VLM Tool Calling — Implementation Details

### Model

Qwen2-VL-7B-Instruct. Tool calls are wrapped in `<tool_call>...</tool_call>` tags.
The response parser in `orchestrator.py` must look for this tag. If it is absent,
the loop assumes the VLM has produced a final answer and terminates.

If the team switches to InternVL2 or any other model, the tag format changes and
`_extract_tool_call()` must be updated. Document that change here.

### Tool Definitions

Defined in `models/vlm_reasoning/tool_registry.py`. Current tools:

| Tool Name | Wraps | Returns |
|---|---|---|
| `run_damage_detection` | `models/damage_detection.run()` | Damage bboxes + classes |
| `run_part_segmentation` | `models/part_segmentation.run()` | Part labels + masks |
| `run_plate_detection` | `models/plate_rc_detection.run()` | Plate text (wire when ready) |
| `execute_cost_computation` | `models/vlm_reasoning/sandbox.py` | Cost estimate dict |

Do not add tools for things that don't exist yet. The VLM will attempt to call
any tool defined — a tool that returns an error poisons the context for the
entire loop.

### Loop Behavior

```
max_iterations = 6   ← hard cap, prevents infinite loops
temperature = 0.1    ← near-greedy, deterministic enough for structured output
max_new_tokens = 512 ← for tool call turns. Use 1024 for final synthesis turn only.
```

If `max_iterations` is hit without a final answer, log a warning and return
whatever partial state exists. Do not raise an exception — a partial report
is better than a 500 error.

### CodeAct Sandbox — `sandbox.py`

The VLM generates Python to compute repair cost. Two execution engines exist,
selected by `vlm.sandbox_engine` in `global_config.yaml`:

- **`monty`** (default) — `pydantic-monty`, a minimal secure Python interpreter
  written in Rust. Filesystem / network / env access are blocked by default;
  `COST_DB` is injected as a read-only input. Monty returns the value of the
  final expression, so it accepts both `result = {...}` (we append a trailing
  `result`) and a final bare dict expression. Optional dependency — if not
  installed, `execute_sandboxed()` falls back to `exec`.
- **`exec`** — restricted CPython `exec()` namespace (the original engine).
  Allowed builtins: `abs`, `min`, `max`, `sum`, `round`, `len`, `range`, `int`,
  `float`, `str`, `list`, `dict`, `zip`, `enumerate`. Allowed imports: `math`,
  `json`, `statistics`, `decimal`. Blocked: `open`, `exec`, `eval`,
  `__import__`, `compile`, `getattr`, `setattr`, dunder access, any non-whitelist
  import. Timeout: 10s via a daemon-thread `join()` (not `signal.alarm()`, which
  cannot fire in the threadpool worker).

**Output contract (both engines):** the produced value MUST be a dict with keys
`damage_part_map`, `total_min`, `total_max`, `currency`. A bare price table or
any other shape is rejected — the model cannot bypass `COST_DB` by returning
arbitrary data. On failure: `{"error": "..."}`.

If the VLM cannot produce valid code, fix the system prompt first (a worked
`result = ...` template lives in `CODEACT_SYSTEM_PROMPT`). Do not weaken the
output-contract validation — that is what keeps pricing anchored to `COST_DB`.

### COST_DB

Hardcoded in `sandbox.py`. The VLM's generated code reads from this dict.
Structure: `COST_DB[damage_class][part_label] = (cost_min_inr, cost_max_inr)`.
All values are tuples of INR integers. Update this dict as domain knowledge
improves — it is the only place pricing lives.

---

## Context Management — Multi-Turn

`pipeline/context_manager.py` implements the sliding window from the architecture
diagram.

```
PINNED   → system prompt + vehicle ID + damage summary. Never dropped.
RETAINED → last 3 message pairs + tool results.
DROPPED  → older turns, compressed to one-line summaries.
```

Use `ClaimContext` when the backend receives a follow-up question on an existing
claim. Do not use it for fresh uploads — start a new context per claim.

---

## Configs — `configs/global_config.yaml`

Single source of truth. All paths, thresholds, and model parameters live here.
No hardcoded paths anywhere in Python code. Load with `yaml.safe_load()`.

Required top-level keys:

```yaml
vlm:
  model_id: "Qwen/Qwen2-VL-7B-Instruct"
  device: "cuda"                          # or "cpu"
  max_new_tokens_tool: 512
  max_new_tokens_final: 1024
  temperature: 0.1
  max_iterations: 6

approval:
  auto_approve_threshold_inr: 50000

damage_detection:
  weights_path: "models/damage_detection/best.pt"
  confidence_threshold: 0.25             # intentionally low — VLM filters
  device: "cuda"

part_segmentation:
  grounding_dino:
    config_path: "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    weights_path: "weights/groundingdino_swint_ogc.pth"
    box_threshold: 0.30
    text_threshold: 0.25
    text_prompt: "front bumper . rear bumper . hood . car door . windshield . rear windshield . fender . trunk lid . roof panel . headlight . taillight"
  sam2:
    config: "sam2_hiera_base_plus.yaml"
    weights_path: "weights/sam2.1_hiera_base_plus.pt"
    device: "cuda"
  postprocess:
    min_mask_area_px: 500
    allow_duplicate_labels: false
  output:
    save_annotated_images: true
    annotated_output_dir: "outputs/part_segmentation/"

plate_rc_detection:
  weights_path: "models/plate_rc_detection/plate.pt"   # update when ready
  device: "cuda"

database:
  url: "postgresql://localhost:5432/veh_dmg_db"

storage:
  image_upload_dir: "data/uploads/"
  output_dir: "outputs/"
```

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

| Component | Status | Owner note |
|---|---|---|
| YOLOv8 damage detection | ✅ FROZEN | Do not retrain |
| Git LFS + monorepo | ✅ Done | On `damage-detection` branch |
| `ensure_weights()` guard | ✅ Done | In `app.py` |
| Local batch pipeline | ✅ Done | Processes `examples/`, writes `outputs/` |
| Part segmentation (Grounding DINO + SAM2) | 🔄 Generating | Codex prompt written |
| Plate/RC detection | 🔄 In progress | Not integrated yet |
| `models/vlm_reasoning/tool_registry.py` | ❌ Build now | Highest priority |
| `models/vlm_reasoning/sandbox.py` | ❌ Build now | Highest priority |
| `pipeline/orchestrator.py` (new) | ❌ Full rewrite | Blocks everything downstream |
| `pipeline/schema.py` additions | ❌ Append now | `FinalDamageReport` missing |
| `pipeline/context_manager.py` | ❌ Build | After orchestrator |
| `backend/app.py` (partial rewrite) | ❌ After orchestrator | Replace pipeline call |
| PostgreSQL schema + models | ❌ Unknown state | Confirm with DB intern |
| Streamlit dashboard | ❌ Not started | Lowest priority |

---

## Things That Will Go Wrong and How to Handle Them

**VLM outputs malformed JSON in tool call**
`_extract_tool_call()` returns `None`. Loop treats it as a final answer turn.
If no actual report JSON is found in `_extract_final_report()`, the response
will contain `{"raw_vlm_response": <raw text>}`. Log a warning. Do not crash.

**SAM2 CUDA OOM during part segmentation**
`sam2_infer.py` catches `RuntimeError`, falls back to CPU for that call,
appends `"SAM2 fell back to CPU due to OOM"` to `warnings`. The run continues.

**Grounding DINO returns 0 detections**
`run.py` returns `SegmentationOutput` with empty `parts` list and a warning.
The orchestrator receives this, VLM notes the empty result and falls back to
visual reasoning for part assignment. Do not return an error — empty is valid.

**All postprocess filters remove all detections**
Same as above. Warning: `"All detections filtered by postprocess thresholds"`.
Lower `min_mask_area_px` in config if this happens consistently.

**VLM loop hits `max_iterations`**
Log warning. Return partial state. The `approval_decision` will be `UNKNOWN`.
FastAPI returns 200 with the partial report — do not 500.

**First request takes 60+ seconds (VLM cold load)**
Expected. The `/health` startup event pre-warms the model. If a request arrives
before warmup completes, the user waits. Add a "Processing..." state to the
dashboard. Do not add a timeout on the client that is shorter than 90 seconds.

**text_prompt in seg_config uses commas instead of dots**
Grounding DINO requires dot-separated prompts. `run.py` auto-corrects and logs
a warning. The config should always use dots — fix it at the source.

---

## Things Not to Build

These appear in the technical magazine document or early architecture discussions.
They are explicitly out of scope for this MVP:

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

If a stakeholder asks for any of these, the answer is: post-MVP.

---

## System Prompt for VLM — Current Version

Location: hardcoded string in `pipeline/orchestrator.py` as `SYSTEM_PROMPT`.
Update it here when it changes so there is a tracked history.

```
You are an expert vehicle damage assessment AI.
You have access to computer vision tools. Use them to accurately assess damage.

WORKFLOW:
1. Examine the image with your vision first. Form an initial hypothesis.
2. Call run_damage_detection to get precise damage locations and classes.
3. Call run_part_segmentation to get vehicle part masks.
4. Cross-reference damage locations with part locations to map damage to parts.
5. Call execute_cost_computation with Python code that uses COST_DB to estimate
   repair costs.
6. Return a structured JSON report.

COST COMPUTATION: Your code must set result = {
    "damage_part_map": [{"damage": str, "part": str, "severity": str,
                         "cost_min": int, "cost_max": int}],
    "total_min": int,
    "total_max": int,
    "currency": "INR"
}

Always be conservative with severity.
Only call tools you actually need.
If part segmentation returns empty results, use your visual assessment of
part locations from the image.
If damage detection confidence is low on a region you visually assess as
damaged, note it in warnings and include it in the report with lower
confidence.
```

---

## Logging Convention

```python
import logging
logger = logging.getLogger(__name__)

# Levels:
# DEBUG  → per-detection results, tensor shapes, raw model outputs
# INFO   → stage entry/exit, tool calls, inference times, model load events
# WARNING → fallbacks (CPU OOM, empty detections, prompt corrections, loop cap hit)
# ERROR  → unrecoverable failures (weight missing, image not found)

# Never use print(). Logger only.
# Format set once in backend/app.py or shared/logging_config.py:
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s"
)
```

---

## How to Run the Pipeline (Current State)

```bash
# 1. Activate env
source .venv/bin/activate

# 2. Verify weights exist
python -c "from models.damage_detection import ensure_weights; ensure_weights(config)"

# 3. Run backend
uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload

# 4. Health check (triggers VLM warmup)
curl http://localhost:8000/health

# 5. Submit image
curl -X POST http://localhost:8000/assess \
  -F "image=@data/examples/test_car.jpg" \
  -F "claim_id=CLAIM_001"
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