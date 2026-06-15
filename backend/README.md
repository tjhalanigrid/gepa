# Backend — Vehicle Damage Assessment API

FastAPI service that accepts a vehicle image, runs the VLM-orchestrated damage
pipeline, and returns a structured cost report. The API is a thin HTTP layer over
`pipeline/` and `models/`; it owns no model logic itself.

## Run

From the **repository root** (not from `backend/`):

```bash
uvicorn backend.app:app --host 0.0.0.0 --port 8000
```

Requires a running Ollama instance with the VLM from `configs/global_config.yaml`
(`vlm.model_id`). See the repo root README for the full stack (Ollama + frontend).

## Structure

```
backend/
├── Dockerfile             # build from repo root: docker build -f backend/Dockerfile .
├── requirements.txt       # API-layer deps (ML deps in repo-root requirements-base.txt)
└── app/
    ├── __init__.py        # exposes `app` for `uvicorn backend.app:app`
    ├── main.py            # app factory: logging, CORS, lifespan, router mounting
    ├── state.py           # in-memory stores (jobs, sessions) — MVP only
    ├── core/
    │   └── config.py      # config loader + constants (paths, CORS, API metadata)
    ├── services/          # business logic (no FastAPI types)
    │   ├── assessment.py  # async pipeline job runner
    │   ├── cost.py        # COST_DB lookup
    │   ├── imaging.py     # annotated-image + YOLO-label generation
    │   └── feedback.py    # feedback-log persistence
    └── routers/           # HTTP routes (one APIRouter per concern)
        ├── health.py      # GET /health, GET /
        ├── assessment.py  # POST /assess, GET /job/{id}, GET /job/{id}/iterations
        ├── images.py      # session/job image serving (plain/annotated/masked/merged)
        ├── sessions.py    # HITL: /session/{id} approve / corrections
        └── feedback.py    # /recalculate, /feedback/stats, /api/feedback
```

## Request flow

`POST /assess` is **asynchronous**: it saves the upload, queues a background job,
and returns `{job_id, status}`. Poll `GET /job/{job_id}` until:

- `{status: "complete", result: <FinalDamageReport | {session_id, report, status: "pending_review"}>}`
- `{status: "failed", error}`

Escalated (low-confidence) reports create a HITL **session** for human review via
the `/session/*` endpoints.

## Tests

```bash
pytest backend/tests
```

The smoke test verifies route wiring and config loading without running model
inference.
