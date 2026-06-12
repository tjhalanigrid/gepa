# 🚗 Vehicle Damage Assessment — VLM Brain + Untrained CV Models

An automotive-claims MVP that inspects a photo of a damaged vehicle and returns a
structured, costed damage report — with a human-review gate.

The core idea: **a Vision-Language Model is the brain**, and **untrained,
off-the-shelf computer-vision models (YOLO + SAM2) are supporting tools** — never
the decision-maker. Nothing here is fine-tuned on damage data. The brain reasons;
the CV models only help with *where* (boxes/masks). Repair cost is computed
deterministically in the backend, not by the LLM.

---

## 🧠 Why "VLM brain + untrained models"

Traditional pipelines train a damage detector (e.g. a YOLO fine-tuned on a damage
dataset). This MVP deliberately uses **no trained damage model**:

- **The VLM (Qwen `qwen3.5:9b` via Ollama) is the sole reasoning brain.** It looks
  at the raw pixels and decides *what* the damage is, *which part* it is on, and
  *how severe* it is — entirely from vision.
- **The untrained CV models add spatial grounding only — they emit no damage
  labels:**
  - **SAM2** (stock `sam2.1_b.pt`, ultralytics) — segments the damage regions the
    brain found into precise **masks**.
  - **YOLO** (stock `yolov8n.pt`, COCO) — an optional, untrained vehicle detector
    that can locate the car (ROI). *Available in the repo as
    `models/vehicle_detection/`; the default pipeline runs VLM + SAM2.*

This keeps the system honest: an untrained model can never assert damage it was
never trained to recognise — only the VLM names damage, and only on what it can
actually see.

---

## 🏗️ Architecture (live flow)

```mermaid
graph TD
    A["Damage photo uploaded"] --> B["FastAPI backend (backend/app/main.py)"]
    B --> C["Orchestrator (pipeline/orchestrator.py)"]

    subgraph brain["VLM BRAIN — Qwen via Ollama (free-form)"]
        C --> D["PiAgent loop (models/vlm_reasoning/pi_agent.py)"]
        D --> D1["run_damage_detection — VLM classifies damage"]
        D --> D2["zoom_region / detect_part — verify uncertain damage"]
        D1 --> D3["Terminate → damage_items {type, part, severity, bbox_pct}"]
        D2 --> D3
    end

    D3 --> E["Backend cost — lookup_cost() over COST_DB (no LLM)"]
    D3 --> F["SAM2 (untrained) — masks on the VLM boxes"]
    F --> G["Merged union: VLM ∪ SAM2 (source-tagged boxes)"]

    E --> H["FinalDamageReport"]
    G --> H
    H --> I{"Approval (image-based, not cost)"}
    I -->|corroborated| J["AUTO_APPROVED"]
    I -->|uncorroborated / no result| K["ESCALATE_TO_HUMAN → review & correct"]
    H --> L["Dashboard (dashboard/app.py)"]
```

**Pipeline stages** (`pipeline/orchestrator.py::run`):

1. **VLM brain loop** — Qwen runs a free-form agent loop. It chooses which tools to
   call, in any order, any number of times (or none), then `Terminate`s with its
   final `damage_items`. There is **no fixed workflow** and the LLM never computes
   cost.
2. **Backend cost** — each damage item is priced in plain Python via `COST_DB`
   (`models/vlm_reasoning/cost_db.py`). Deterministic, auditable, single source.
3. **SAM2 segmentation** — SAM2 is prompted by the VLM's damage boxes (never
   "segment-everything"), producing tight **masks** on the actual damage. The
   **merged union** (VLM boxes ∪ SAM2 mask-boxes) is rendered source-coloured.
4. **Approval (image-based)** — the report is `AUTO_APPROVED` when the VLM's final
   claims are corroborated by its own detection pass; otherwise it
   `ESCALATE_TO_HUMAN`. **Cost never gates approval.**

---

## 🛡️ Anti-hallucination contract

- **Untrained models emit no damage labels** — they cannot hallucinate damage they
  were never trained to recognise.
- **The VLM is corroborated against itself** — a final claim is trusted only if the
  VLM's independent detection pass saw the same damage class. Uncorroborated claims
  → human review.
- **Cost is deterministic** — priced from `COST_DB` in the backend, not invented by
  the LLM.

---

## 🖥️ Dashboard

Per assessment the Streamlit UI shows:

- **Detected Damage** — the VLM's damage bounding boxes.
- **Merged (VLM ∪ SAM2)** — union boxes, coloured by source (🟩 VLM · 🟧 VLM
  confirmed by SAM2 · 🟦 SAM2 region).
- **Damage masks (SAM2)** — precise segmentation overlay (auto-generated).
- **Agent iteration logs** — every tool the brain chose to call, *why*, and the
  result (free-form, not a fixed pipeline).
- **Damage summary** — part, severity, and INR cost range.
- **Human-in-the-loop review** — escalated claims open an editable Review → Correct
  → Annotate → Submit flow; the result is logged as `HUMAN_APPROVED`.

---

## 🚀 Setup & Run

### Prerequisites
- Python 3.10+ (developed on 3.13), `pip install -r requirements-base.txt`
- **Ollama** with a vision-capable model:
  ```bash
  ollama serve                 # Terminal 1 — keep running
  ollama pull qwen3.5:9b       # vision-capable VLM brain
  ```
- SAM2 weight `models/damage_detection/models/sam2.1_b.pt` (stock ultralytics SAM2).
  `yolov8n.pt` auto-downloads on first use if vehicle detection is enabled.

### Run (3 terminals)
```bash
# Terminal 1 — VLM brain
ollama serve

# Terminal 2 — backend API
uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload

# Terminal 3 — dashboard
streamlit run dashboard/app.py        # http://localhost:8501
```

Upload a vehicle photo and review the assessment. First request is slower while the
VLM loads into memory; subsequent runs are faster.

### Quick CLI check (no UI)
```bash
python - <<'PY'
import yaml, json
from pipeline.orchestrator import run
cfg = yaml.safe_load(open("configs/global_config.yaml"))
rep = run("data/uploads/<some_image>.jpg", cfg)
print("approval:", rep["approval_decision"], "| INR", rep["total_min"], "-", rep["total_max"])
print("tools used:", [it["tool"] for it in rep["iterations"]])
PY
```

---

## 📁 Repo structure (key paths)

```
models/vlm_reasoning/
  pi_agent.py        ← the VLM brain: free-form agent loop, tools, prompt
  ollama_client.py   ← Ollama HTTP client (vision messages)
  cost_db.py         ← COST_DB + lookup_cost() (single pricing source)
models/part_segmentation/
  infer.py           ← SAM2 segmentation (untrained)
models/vehicle_detection/
  __init__.py        ← stock YOLOv8 vehicle ROI (untrained, optional)
shared/sam_mask.py   ← SAM2 mask overlay (ultralytics)
pipeline/
  orchestrator.py    ← run(): brain loop → backend cost → SAM2 masks → approval
  schema.py          ← Pydantic contracts (FinalDamageReport, etc.)
backend/app/main.py  ← FastAPI: /assess, /job, HITL /session endpoints, images
dashboard/app.py     ← Streamlit UI (tabs, iteration logs, review flow)
configs/global_config.yaml  ← model id, sampling, SAM2 weights, etc.
```

---

## ⚙️ Configuration (`configs/global_config.yaml`)

- `vlm.model_id: qwen3.5:9b` — the Ollama brain tag (vision-capable).
- `vlm.thinking: false`, instruct sampling (`temperature 0.7`, `top_p 0.8`,
  `top_k 20`, `presence_penalty 1.5`), `num_ctx 8192` — Qwen "Best Practices"
  (thinking mode is off so the JSON action is never truncated).
- `part_segmentation.sam2.weights_path` — the stock SAM2 weight used for masks.

---

## ⚠️ Known limitation

Because the system is fully **untrained**, bounding-box *placement* is approximate —
the boxes/masks land on the right damage area but are not pixel-tight, since they
follow the VLM's coordinates (a 9B VLM's weak spot). **Damage type, part, severity,
and cost are reliable.** Pixel-tight localisation would require a trained damage
detector, which is intentionally out of scope for this untrained MVP.
