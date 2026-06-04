"""
dashboard/app.py

Streamlit HITL dashboard for vehicle damage assessment.

Stages:
  upload    — file upload form → POST /assess (returns job_id immediately)
  polling   — polling GET /job/{job_id} every 5s while pipeline runs
  approved  — AUTO_APPROVED result display
  escalated — ESCALATE_TO_HUMAN editable review interface
  done      — post-approval confirmation
"""

import json
import sys
import time
from pathlib import Path

# Ensure repo root is on sys.path so `shared` is importable regardless of
# which directory Streamlit is launched from.
_REPO_ROOT = str(Path(__file__).parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import requests
import streamlit as st

API_BASE = "http://localhost:8000"

CAR_REFERENCE_SVG = """
<svg viewBox="0 0 400 260" xmlns="http://www.w3.org/2000/svg"
     style="width:100%;max-width:400px;background:#f8f8f6;border-radius:8px;padding:8px;">
  <rect x="60" y="60" width="280" height="140" rx="18"
        fill="none" stroke="#888" stroke-width="1.5"/>
  <rect x="110" y="30" width="180" height="60" rx="12"
        fill="none" stroke="#888" stroke-width="1.2"/>
  <rect x="120" y="36" width="80" height="34" rx="4"
        fill="#ddeeff" stroke="#6699cc" stroke-width="1" opacity="0.7"/>
  <rect x="200" y="36" width="80" height="34" rx="4"
        fill="#ddeeff" stroke="#6699cc" stroke-width="1" opacity="0.7"/>
  <circle cx="112" cy="205" r="22" fill="#ddd" stroke="#888" stroke-width="1.5"/>
  <circle cx="288" cy="205" r="22" fill="#ddd" stroke="#888" stroke-width="1.5"/>
  <circle cx="112" cy="205" r="11" fill="#bbb" stroke="#888" stroke-width="1"/>
  <circle cx="288" cy="205" r="11" fill="#bbb" stroke="#888" stroke-width="1"/>
  <rect x="64" y="172" width="80" height="22" rx="4"
        fill="#E6F1FB" stroke="#378ADD" stroke-width="1"/>
  <text x="104" y="187" text-anchor="middle" font-size="9"
        font-family="sans-serif" fill="#0C447C" font-weight="600">front bumper</text>
  <rect x="256" y="172" width="80" height="22" rx="4"
        fill="#E6F1FB" stroke="#378ADD" stroke-width="1"/>
  <text x="296" y="187" text-anchor="middle" font-size="9"
        font-family="sans-serif" fill="#0C447C" font-weight="600">rear bumper</text>
  <rect x="80" y="64" width="70" height="20" rx="4"
        fill="#E1F5EE" stroke="#1D9E75" stroke-width="1"/>
  <text x="115" y="78" text-anchor="middle" font-size="9"
        font-family="sans-serif" fill="#085041" font-weight="600">hood</text>
  <rect x="250" y="64" width="70" height="20" rx="4"
        fill="#E1F5EE" stroke="#1D9E75" stroke-width="1"/>
  <text x="285" y="78" text-anchor="middle" font-size="9"
        font-family="sans-serif" fill="#085041" font-weight="600">trunk lid</text>
  <rect x="160" y="32" width="80" height="18" rx="4"
        fill="#FAEEDA" stroke="#BA7517" stroke-width="1"/>
  <text x="200" y="45" text-anchor="middle" font-size="9"
        font-family="sans-serif" fill="#633806" font-weight="600">roof panel</text>
  <rect x="64" y="64" width="52" height="18" rx="4"
        fill="#FBEAF0" stroke="#D4537E" stroke-width="1"/>
  <text x="90" y="77" text-anchor="middle" font-size="8"
        font-family="sans-serif" fill="#72243E" font-weight="600">L headlight</text>
  <rect x="284" y="64" width="52" height="18" rx="4"
        fill="#FBEAF0" stroke="#D4537E" stroke-width="1"/>
  <text x="310" y="77" text-anchor="middle" font-size="8"
        font-family="sans-serif" fill="#72243E" font-weight="600">R headlight</text>
  <rect x="110" y="100" width="78" height="56" rx="4"
        fill="none" stroke="#aaa" stroke-width="1" stroke-dasharray="4 2"/>
  <text x="149" y="132" text-anchor="middle" font-size="8"
        font-family="sans-serif" fill="#555">FL door</text>
  <rect x="212" y="100" width="78" height="56" rx="4"
        fill="none" stroke="#aaa" stroke-width="1" stroke-dasharray="4 2"/>
  <text x="251" y="132" text-anchor="middle" font-size="8"
        font-family="sans-serif" fill="#555">FR door</text>
  <text x="80" y="130" text-anchor="middle" font-size="8"
        font-family="sans-serif" fill="#777" transform="rotate(-90,80,130)">L fender</text>
  <text x="322" y="130" text-anchor="middle" font-size="8"
        font-family="sans-serif" fill="#777" transform="rotate(90,322,130)">R fender</text>
  <text x="112" y="210" text-anchor="middle" font-size="8"
        font-family="sans-serif" fill="#555" font-weight="600">tire</text>
  <text x="288" y="210" text-anchor="middle" font-size="8"
        font-family="sans-serif" fill="#555" font-weight="600">tire</text>
</svg>
"""

CLASS_LEGEND_ICON = {
    "dent": "🔵", "scratch": "🟢", "crack": "🟡",
    "glass_shatter": "🟣", "lamp_broken": "🟠", "tire_flat": "⚫",
}
POLL_INTERVAL_S = 5

DAMAGE_CLASSES = ["dent", "scratch", "crack", "glass_shatter", "lamp_broken", "tire_flat"]

PART_LABELS = [
    "front bumper", "rear bumper", "hood", "windshield", "rear windshield",
    "front_left_door", "front_right_door", "rear_left_door", "rear_right_door",
    "left fender", "right fender", "trunk lid", "roof panel",
    "headlight", "taillight", "tire",
]

SEVERITY_LEVELS = ["minor", "moderate", "severe"]


def _init_state() -> None:
    defaults = {
        "stage": "upload",
        "job_id": None,
        "session_id": None,
        "report": None,
        "edited_map": None,
        "recalc_result": None,
        "uploaded_image_bytes": None,
        "claim_id": None,
        "poll_start": None,
        # Bbox annotation UI
        "detections_with_bbox": [],
        "annotated_image_bytes": None,
        "bbox_edit_mode": False,
        # Correction workflow
        "correction_step": 1,
        "correction_actions": [],
        "working_damage_map": [],
        "bbox_annotations": [],
        "intern_name": "",
        "added_items": [],
        # SAM2 mask display
        "masked_image_bytes": None,
        # Feedback stats cache
        "feedback_stats": None,
        "feedback_stats_at": 0.0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _reset() -> None:
    for k in list(st.session_state.keys()):
        del st.session_state[k]


# ── State renderers ───────────────────────────────────────────────────────────

def render_upload() -> None:
    st.title("Vehicle Damage Assessment")
    st.markdown("Upload a vehicle photo to assess damage and estimate repair cost.")

    uploaded = st.file_uploader("Vehicle Image", type=["jpg", "jpeg", "png", "webp"])
    claim_id = st.text_input("Claim ID (optional)")
    vehicle_id = st.text_input("Vehicle ID (optional)")

    submit = st.button("Assess Damage", disabled=(uploaded is None), type="primary")

    if submit and uploaded is not None:
        img_bytes = uploaded.read()
        st.session_state.uploaded_image_bytes = img_bytes
        st.session_state.claim_id = claim_id or None

        try:
            resp = requests.post(
                f"{API_BASE}/assess",
                files={"image": (uploaded.name, img_bytes, uploaded.type)},
                data={
                    "claim_id": claim_id or "",
                    "vehicle_id": vehicle_id or "",
                },
                timeout=30,
            )
        except requests.exceptions.ConnectionError:
            st.error(
                "Cannot connect to backend at http://localhost:8000. "
                "Start it with: uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload"
            )
            return
        except requests.exceptions.Timeout:
            st.error("Backend did not respond within 30s. Is it running?")
            return

        if resp.status_code != 200:
            st.error(f"Submission failed ({resp.status_code}): {resp.text}")
            return

        data = resp.json()
        st.session_state.job_id = data["job_id"]
        st.session_state.poll_start = time.time()
        st.session_state.stage = "polling"
        st.rerun()


def render_polling() -> None:
    st.title("Vehicle Damage Assessment")

    st.info(
        "Waiting for pipeline result — polling every 5s.\n\n"
        "VLM cold start can take 2–5 minutes on first run. "
        "Subsequent runs are faster once the model is loaded."
    )

    if st.session_state.uploaded_image_bytes:
        st.image(
            st.session_state.uploaded_image_bytes,
            caption="Submitted Image",
            width=400,
        )

    st.caption(f"Job ID: `{st.session_state.job_id}` — checking every {POLL_INTERVAL_S}s")

    job_id = st.session_state.job_id

    try:
        resp = requests.get(f"{API_BASE}/job/{job_id}", timeout=10)
    except requests.exceptions.ConnectionError:
        st.error("Lost connection to backend.")
        return
    except requests.exceptions.Timeout:
        st.warning("Poll request timed out — retrying.")
        time.sleep(POLL_INTERVAL_S)
        st.rerun()
        return

    if resp.status_code != 200:
        st.error(f"Job poll failed ({resp.status_code}): {resp.text}")
        return

    job = resp.json()
    status = job.get("status")

    if status == "failed":
        st.error("❌ Pipeline failed")
        err = job.get("error", "Unknown error")
        st.code(err, language="text")
        elapsed = job.get("elapsed_s")
        if elapsed:
            st.caption(f"Failed after {elapsed:.1f}s")
        if st.button("🔄 Start Over"):
            _reset()
            st.rerun()
        st.stop()

    elif status in ("queued", "processing"):
        elapsed = job.get("elapsed_s", 0)
        st.info(f"Pipeline running... ({elapsed}s elapsed)")
        if elapsed > 300:
            st.warning(
                "This is taking longer than expected. "
                "If it exceeds 600s it will be automatically cancelled."
            )
        time.sleep(POLL_INTERVAL_S)
        st.rerun()
        return

    # status == "complete"
    result = job["result"]

    if "session_id" in result:
        st.session_state.session_id = result["session_id"]
        st.session_state.report = result["report"]
        st.session_state.edited_map = [dict(e) for e in result["report"]["damage_part_map"]]
        st.session_state["detections_with_bbox"] = result.get("report", {}).get("detections_with_bbox", [])
        st.session_state["annotated_image_bytes"] = None  # fetch fresh on Step 2 load
        st.session_state.stage = "escalated"
    else:
        st.session_state.report = result
        st.session_state["detections_with_bbox"] = result.get("detections_with_bbox", [])
        st.session_state.stage = "approved"

    st.rerun()


def _to_canvas_dets(detections_with_bbox: list) -> list:
    """Convert DetectionWithBBox objects or dicts to canvas-compatible format."""
    result = []
    for i, det in enumerate(detections_with_bbox):
        if hasattr(det, "model_dump"):
            d = det.model_dump()
        elif isinstance(det, dict):
            d = det
        else:
            continue
        result.append({
            "id":         d.get("id", f"yolo_{i}"),
            "index":      d.get("index", i + 1),
            "bbox":       d.get("bbox", [0, 0, 100, 100]),
            "damage":     d.get("damage", "dent"),
            "part":       d.get("part", "front_bumper"),
            "severity":   d.get("severity", "minor"),
            "confidence": d.get("confidence", 0.5),
            "source":     d.get("source", "yolo"),
            "cost_min":   d.get("cost_min", 0),
            "cost_max":   d.get("cost_max", 0),
        })
    return result


def _render_detection_legend(damage_map: list) -> None:
    """Numbered colour-coded legend matching YOLO annotated image boxes."""
    if not damage_map:
        return
    CLASS_COLORS = {
        "dent": "#378ADD", "scratch": "#1D9E75", "crack": "#BA7517",
        "glass_shatter": "#D4537E", "lamp_broken": "#D85A30", "tire_flat": "#888780",
    }
    cols = st.columns(min(len(damage_map), 3))
    for i, entry in enumerate(damage_map):
        damage = entry.get("damage", "")
        part   = entry.get("part", "").replace("_", " ")
        sev    = entry.get("severity", "")
        color  = CLASS_COLORS.get(damage, "#888")
        with cols[i % 3]:
            st.markdown(
                f"<div style='padding:8px;border-left:3px solid {color};"
                f"margin-bottom:6px;border-radius:0 4px 4px 0'>"
                f"<b style='color:{color}'>Box {i+1}</b><br>"
                f"{damage.replace('_', ' ')} · {part}<br>"
                f"<small>{sev}</small></div>",
                unsafe_allow_html=True,
            )


def render_approved() -> None:
    report = st.session_state.report

    st.title("Vehicle Damage Assessment")
    st.success("✓ AUTO APPROVED")

    job_id = st.session_state.get("job_id", "")
    tab_orig, tab_ann = st.tabs(["Original", "Detected Damage"])

    with tab_orig:
        if st.session_state.uploaded_image_bytes:
            st.image(
                st.session_state.uploaded_image_bytes,
                caption="Submitted Image",
                use_container_width=True,
            )

    with tab_ann:
        ann_key = f"annotated_bytes_{job_id}"
        if not st.session_state.get(ann_key) and job_id:
            try:
                r = requests.get(
                    f"{API_BASE}/job/{job_id}/annotated_image", timeout=10
                )
                if r.status_code == 200:
                    st.session_state[ann_key] = r.content
            except Exception:
                pass
        ann_bytes = st.session_state.get(ann_key)
        if ann_bytes:
            st.image(ann_bytes, use_container_width=True,
                     caption="YOLO detected damage regions (numbered)")
            _render_detection_legend(report.get("damage_part_map", []))
        else:
            st.info("Annotated image not available")

    st.subheader("Damage Summary")

    entries = report.get("damage_part_map", [])
    if entries:
        st.table({
            "Damage":         [e["damage"] for e in entries],
            "Part":           [e["part"] for e in entries],
            "Severity":       [e["severity"] for e in entries],
            "Min Cost (INR)": [f"₹{e['cost_min']:,}" for e in entries],
            "Max Cost (INR)": [f"₹{e['cost_max']:,}" for e in entries],
        })
    else:
        st.info("No damage entries detected.")

    col1, col2 = st.columns(2)
    col1.metric("Total Min (INR)", f"₹{report['total_min']:,}")
    col2.metric("Total Max (INR)", f"₹{report['total_max']:,}")

    with st.expander("Tool Call Log"):
        for record in report.get("tool_call_log", []):
            st.write(
                f"**{record['tool']}** — {record['elapsed_s']}s "
                f"— keys: {record.get('result_keys', [])}"
            )

    if report.get("warnings"):
        with st.expander("Warnings"):
            for w in report["warnings"]:
                st.warning(w)

    st.download_button(
        "Download Report JSON",
        data=json.dumps(report, indent=2),
        file_name=f"damage_report_{report.get('image_path', 'unknown').split('/')[-1]}.json",
        mime="application/json",
    )

    if st.button("Assess Another Image"):
        _reset()
        st.rerun()


def _render_feedback_sidebar() -> None:
    """Fetch and display feedback loop stats in sidebar (60s cache)."""
    now = time.time()
    if (
        st.session_state.get("feedback_stats") is None
        or now - st.session_state.get("feedback_stats_at", 0) > 60
    ):
        try:
            r = requests.get(f"{API_BASE}/feedback/stats", timeout=5)
            if r.status_code == 200:
                st.session_state["feedback_stats"] = r.json()
                st.session_state["feedback_stats_at"] = now
        except Exception:
            pass

    stats = st.session_state.get("feedback_stats") or {}
    if stats:
        with st.sidebar.expander("Feedback Loop", expanded=False):
            st.metric("Total corrections", stats.get("total_corrections", 0))
            st.metric("Missed damages found", stats.get("total_missed_damages_found", 0))
            st.metric("False positives caught", stats.get("total_false_positives_removed", 0))
            n_ann = stats.get("total_bbox_annotations", 0)
            st.metric("YOLO annotations", n_ann)
            st.progress(min(n_ann / 50, 1.0), text=f"{n_ann}/50 to enable fine-tune")


def render_escalated() -> None:
    report = st.session_state.report
    damage_map = report.get("damage_part_map", [])

    _render_feedback_sidebar()

    st.title("Vehicle Damage Assessment")
    st.warning("⚠ ESCALATE TO HUMAN — Manual review required before approval")

    step = st.session_state.correction_step
    step_labels = ["1 Review", "2 Correct", "3 Annotate", "4 Submit"]
    st.markdown(
        " → ".join(
            f"**{s}**" if i + 1 == step else s
            for i, s in enumerate(step_labels)
        )
    )
    st.divider()

    # ── Step 1: Review ────────────────────────────────────────────────────────
    if step == 1:
        session_id_s1 = st.session_state.session_id
        if st.session_state.uploaded_image_bytes:
            tab_orig, tab_ann, tab_masked = st.tabs(
                ["Original", "Detected Damage", "Damage masks (SAM2)"]
            )
            with tab_orig:
                st.image(st.session_state.uploaded_image_bytes, use_container_width=True)
            with tab_ann:
                job_id_s1 = st.session_state.get("job_id", "")
                ann_key = f"annotated_bytes_{job_id_s1}"
                if not st.session_state.get(ann_key) and job_id_s1:
                    try:
                        r = requests.get(
                            f"{API_BASE}/job/{job_id_s1}/annotated_image", timeout=10
                        )
                        if r.status_code == 200:
                            st.session_state[ann_key] = r.content
                    except Exception:
                        pass
                ann_bytes = st.session_state.get(ann_key)
                if ann_bytes:
                    st.image(ann_bytes, use_container_width=True,
                             caption="Numbered boxes = YOLO detections")
                    _render_detection_legend(damage_map)
                else:
                    st.info("Annotated image not available")
            with tab_masked:
                sam_key = f"sam_bytes_{job_id_s1}"
                if not st.session_state.get(sam_key):
                    col_btn, _ = st.columns([1, 3])
                    with col_btn:
                        if st.button("Generate SAM2 masks", key=f"gen_sam_{job_id_s1}"):
                            with st.spinner("Running SAM2 segmentation (30-60s)..."):
                                url = (
                                    f"{API_BASE}/session/{session_id_s1}/masked_image"
                                    if session_id_s1
                                    else f"{API_BASE}/job/{job_id_s1}/masked_image"
                                )
                                try:
                                    r = requests.get(url, timeout=90)
                                    if r.status_code == 200:
                                        st.session_state[sam_key] = r.content
                                        st.rerun()
                                    elif r.status_code == 503:
                                        st.session_state[f"sam_error_{job_id_s1}"] = (
                                            "SAM2 weights missing. "
                                            "Run: python3 scripts/download_sam2_weights.py"
                                        )
                                    elif r.status_code == 404:
                                        st.session_state[f"sam_error_{job_id_s1}"] = (
                                            "No detections to mask."
                                        )
                                    else:
                                        try:
                                            detail = r.json().get("detail", r.text[:100])
                                        except Exception:
                                            detail = r.text[:100]
                                        st.session_state[f"sam_error_{job_id_s1}"] = (
                                            f"Mask generation failed: {detail}"
                                        )
                                except requests.Timeout:
                                    st.session_state[f"sam_error_{job_id_s1}"] = (
                                        "SAM2 timed out (>90s). Try a smaller image."
                                    )
                                except Exception as _e:
                                    st.session_state[f"sam_error_{job_id_s1}"] = str(_e)
                    err = st.session_state.get(f"sam_error_{job_id_s1}")
                    if err:
                        st.warning(err)
                    else:
                        st.info(
                            "Click to generate precise segmentation masks "
                            "over each detected damage region using SAM2."
                        )
                if st.session_state.get(sam_key):
                    st.image(
                        st.session_state[sam_key],
                        use_container_width=True,
                        caption="SAM2 segmentation masks — colour = damage class",
                    )
                    if st.button("Clear masks", key=f"clear_sam_{job_id_s1}"):
                        del st.session_state[sam_key]
                        st.rerun()

        if damage_map:
            severity_icon = {"minor": "🟢", "moderate": "🟡", "severe": "🔴"}
            st.subheader("Pipeline Output")
            st.table({
                "": [severity_icon.get(e["severity"], "⚪") for e in damage_map],
                "Damage": [e["damage"] for e in damage_map],
                "Part": [e["part"] for e in damage_map],
                "Severity": [e["severity"] for e in damage_map],
                "Min ₹": [f"₹{e['cost_min']:,}" for e in damage_map],
                "Max ₹": [f"₹{e['cost_max']:,}" for e in damage_map],
            })
        else:
            st.info("No damage detected by pipeline.")

        col1, col2 = st.columns(2)
        col1.metric("Total Min", f"₹{report['total_min']:,}")
        col2.metric("Total Max", f"₹{report['total_max']:,}")

        c1, c2, c3 = st.columns(3)
        if c1.button("Everything looks correct →", type="primary"):
            st.session_state.correction_actions = [
                {"action": "keep", "original": e, "corrected": e, "reason": None}
                for e in damage_map
            ]
            st.session_state.working_damage_map = list(damage_map)
            st.session_state.correction_step = 4
            st.rerun()
        if c2.button("Make corrections →"):
            st.session_state.working_damage_map = [dict(e) for e in damage_map]
            st.session_state.correction_step = 2
            st.rerun()
        if c3.button("Reject & Re-run"):
            _reset()
            st.rerun()

    # ── Step 2: Draggable canvas + correction rows ────────────────────────────
    elif step == 2:
        from shared.bbox_canvas import render_bbox_canvas

        st.subheader("Correct detections")
        st.caption(
            "Canvas: drag boxes to move · drag corner to resize · ＋ Draw mode for new boxes. "
            "Use the right panel to edit class/part/severity and confirm changes."
        )

        session_id = st.session_state.session_id
        detections = st.session_state.get("detections_with_bbox", [])

        # Fetch image dims from plain_image headers (for canvas scale)
        img_w = st.session_state.get("_canvas_img_w", 1920)
        img_h = st.session_state.get("_canvas_img_h", 1080)
        if "_canvas_img_w" not in st.session_state and session_id:
            try:
                r = requests.head(
                    f"{API_BASE}/session/{session_id}/plain_image",
                    timeout=5,
                )
                if r.status_code == 200:
                    img_w = int(r.headers.get("X-Image-Width", 1920))
                    img_h = int(r.headers.get("X-Image-Height", 1080))
                    st.session_state["_canvas_img_w"] = img_w
                    st.session_state["_canvas_img_h"] = img_h
            except Exception:
                pass

        left_col, right_col = st.columns([65, 35])

        # ── Left: interactive canvas ──────────────────────────────────────────
        with left_col:
            image_url = f"{API_BASE}/session/{session_id}/plain_image"
            render_bbox_canvas(
                image_url=image_url,
                detections=detections,
                img_width=img_w,
                img_height=img_h,
                canvas_height=560,
                key=f"bbox_canvas_{session_id}",
            )
            st.caption(
                "Canvas changes are visual only — use the right panel to register edits."
            )

        # ── Right: native Streamlit correction controls ───────────────────────
        with right_col:
            st.markdown("**Detections**")
            for i, det in enumerate(detections):
                is_removed = st.session_state.get(f"det_removed_{i}", False)
                conf = det.get("confidence", 1.0)
                badge = "🟢" if conf >= 0.7 else "🟡" if conf >= 0.5 else "🔴"

                if is_removed:
                    st.markdown(
                        f"~~{badge} **#{det.get('index')}** {det.get('damage')} · {det.get('part')}~~"
                        f" <small>removed</small>",
                        unsafe_allow_html=True,
                    )
                    if st.button("Undo", key=f"undo_rm_{i}"):
                        st.session_state[f"det_removed_{i}"] = False
                        st.rerun()
                else:
                    st.markdown(f"{badge} **#{det.get('index', i+1)}**")
                    dmg_idx = DAMAGE_CLASSES.index(det.get("damage")) if det.get("damage") in DAMAGE_CLASSES else 0
                    part_idx = PART_LABELS.index(det.get("part")) if det.get("part") in PART_LABELS else 0
                    sev_idx = SEVERITY_LEVELS.index(det.get("severity")) if det.get("severity") in SEVERITY_LEVELS else 1
                    st.selectbox("Class", DAMAGE_CLASSES, index=dmg_idx,
                                 key=f"det_dmg_{i}", label_visibility="collapsed")
                    st.selectbox("Part", PART_LABELS, index=part_idx,
                                 key=f"det_part_{i}", label_visibility="collapsed")
                    st.selectbox("Severity", SEVERITY_LEVELS, index=sev_idx,
                                 key=f"det_sev_{i}", label_visibility="collapsed")
                    src = det.get("source", "yolo")
                    st.caption(f"conf: {conf:.2f}  source: {src}")
                    if st.button("✗ Remove", key=f"rm_det_{i}"):
                        st.session_state[f"det_removed_{i}"] = True
                        st.rerun()
                st.divider()

            with st.expander("+ Add missed damage"):
                add_dmg = st.selectbox("Damage", DAMAGE_CLASSES, key="add2_dmg")
                add_part = st.selectbox("Part", PART_LABELS, key="add2_part")
                add_sev = st.selectbox("Severity", SEVERITY_LEVELS, key="add2_sev")
                c1, c2, c3, c4 = st.columns(4)
                bx1 = c1.number_input("x1", value=0, min_value=0, key="add2_x1")
                by1 = c2.number_input("y1", value=0, min_value=0, key="add2_y1")
                bx2 = c3.number_input("x2", value=100, min_value=0, key="add2_x2")
                by2 = c4.number_input("y2", value=100, min_value=0, key="add2_y2")
                if st.button("Add"):
                    new_dets = list(st.session_state.get("detections_with_bbox", []))
                    new_idx = max((d.get("index", 0) for d in new_dets), default=0) + 1
                    new_dets.append({
                        "index": new_idx,
                        "bbox": [float(bx1), float(by1), float(bx2), float(by2)],
                        "damage": add_dmg, "part": add_part, "severity": add_sev,
                        "confidence": 1.0, "source": "human", "cost_min": 0, "cost_max": 0,
                    })
                    st.session_state["detections_with_bbox"] = new_dets
                    st.rerun()

            with st.expander("Car part reference", expanded=False):
                st.components.v1.html(CAR_REFERENCE_SVG, height=200)

        # ── Apply & refresh annotated image ───────────────────────────────────
        if st.button("Apply corrections & refresh annotated image"):
            corrected = []
            for i, det in enumerate(detections):
                if st.session_state.get(f"det_removed_{i}", False):
                    continue
                corrected.append({
                    "index": det.get("index", i + 1),
                    "bbox": det.get("bbox", [0.0, 0.0, 0.0, 0.0]),
                    "damage": st.session_state.get(f"det_dmg_{i}", det.get("damage")),
                    "part": st.session_state.get(f"det_part_{i}", det.get("part")),
                    "severity": st.session_state.get(f"det_sev_{i}", det.get("severity")),
                    "confidence": det.get("confidence", 0.0),
                    "source": det.get("source", "yolo"),
                    "cost_min": det.get("cost_min", 0),
                    "cost_max": det.get("cost_max", 0),
                })
            try:
                upd = requests.post(
                    f"{API_BASE}/session/{session_id}/update_detections",
                    json={"corrected_detections": corrected},
                    timeout=15,
                )
                if upd.status_code == 200:
                    st.session_state["annotated_image_bytes"] = None
                    st.session_state["detections_with_bbox"] = corrected
                    st.rerun()
                else:
                    st.error(f"Update failed: {upd.text}")
            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to backend.")

        # ── Navigation ────────────────────────────────────────────────────────
        col_next, col_back = st.columns([1, 1])
        if col_next.button("Next →", type="primary"):
            actions = []
            final_map = []
            original_damage_map = report.get("damage_part_map", [])

            for i, det in enumerate(detections):
                is_removed = st.session_state.get(f"det_removed_{i}", False)
                original_entry = original_damage_map[i] if i < len(original_damage_map) else None
                current_entry = {
                    "damage": st.session_state.get(f"det_dmg_{i}", det.get("damage")),
                    "part": st.session_state.get(f"det_part_{i}", det.get("part")),
                    "severity": st.session_state.get(f"det_sev_{i}", det.get("severity")),
                    "cost_min": det.get("cost_min", 0),
                    "cost_max": det.get("cost_max", 0),
                }
                source = det.get("source", "yolo")

                if source == "human":
                    if not is_removed:
                        actions.append({"action": "add", "original": None, "corrected": current_entry, "reason": None})
                        final_map.append(current_entry)
                elif is_removed:
                    actions.append({"action": "remove", "original": original_entry, "corrected": None, "reason": None})
                elif original_entry and (
                    current_entry["damage"] != original_entry.get("damage")
                    or current_entry["part"] != original_entry.get("part")
                    or current_entry["severity"] != original_entry.get("severity")
                ):
                    actions.append({"action": "edit", "original": original_entry, "corrected": current_entry, "reason": None})
                    final_map.append(current_entry)
                else:
                    actions.append({"action": "keep", "original": original_entry, "corrected": original_entry, "reason": None})
                    final_map.append(current_entry)

            st.session_state.correction_actions = actions
            st.session_state.working_damage_map = final_map
            st.session_state.correction_step = 4
            st.rerun()

        if col_back.button("← Back"):
            st.session_state.correction_step = 1
            st.rerun()

    # ── Step 3: Annotate missed damages ───────────────────────────────────────
    elif step == 3:
        st.subheader("Annotate missed damages")
        st.caption("Draw or enter bounding boxes for damages the pipeline missed.")

        try:
            from streamlit_drawable_canvas import st_canvas
            HAS_CANVAS = True
        except ImportError:
            HAS_CANVAS = False
            st.info(
                "Canvas not available — enter approximate pixel coordinates below. "
                "Install `streamlit-drawable-canvas` to enable drawing."
            )

        add_actions = [a for a in st.session_state.correction_actions if a["action"] == "add"]
        bbox_annotations = list(st.session_state.get("bbox_annotations", []))

        for j, action in enumerate(add_actions):
            item = action["corrected"]
            st.markdown(f"**{j + 1}. {item['damage']} on {item['part']}**")
            existing = bbox_annotations[j] if j < len(bbox_annotations) else {}

            if HAS_CANVAS and st.session_state.uploaded_image_bytes:
                from PIL import Image as PILImage
                import io as _io
                bg_img = PILImage.open(_io.BytesIO(st.session_state.uploaded_image_bytes))
                canvas_w = min(bg_img.width, 600)
                canvas_h = min(bg_img.height, 400)
                bg_img.thumbnail((canvas_w, canvas_h))
                result = st_canvas(
                    fill_color="rgba(255, 0, 0, 0.1)",
                    stroke_width=2,
                    stroke_color="#FF0000",
                    background_image=bg_img,
                    drawing_mode="rect",
                    key=f"canvas_{j}",
                    height=canvas_h,
                    width=canvas_w,
                )
                if result.json_data and result.json_data.get("objects"):
                    obj = result.json_data["objects"][0]
                    orig_w, orig_h = st.session_state.get("_img_wh", (canvas_w, canvas_h))
                    scale_x = orig_w / canvas_w
                    scale_y = orig_h / canvas_h
                    ann = {
                        "x1": int(obj.get("left", 0) * scale_x),
                        "y1": int(obj.get("top", 0) * scale_y),
                        "x2": int((obj.get("left", 0) + obj.get("width", 0)) * scale_x),
                        "y2": int((obj.get("top", 0) + obj.get("height", 0)) * scale_y),
                        "damage_class": item["damage"],
                        "part": item["part"],
                        "severity": item["severity"],
                    }
                    while len(bbox_annotations) <= j:
                        bbox_annotations.append({})
                    bbox_annotations[j] = ann
            else:
                c1, c2, c3, c4 = st.columns(4)
                x1 = c1.number_input("x1 (px)", value=int(existing.get("x1", 0)), key=f"bbox_x1_{j}", min_value=0)
                y1 = c2.number_input("y1 (px)", value=int(existing.get("y1", 0)), key=f"bbox_y1_{j}", min_value=0)
                x2 = c3.number_input("x2 (px)", value=int(existing.get("x2", 100)), key=f"bbox_x2_{j}", min_value=0)
                y2 = c4.number_input("y2 (px)", value=int(existing.get("y2", 100)), key=f"bbox_y2_{j}", min_value=0)
                while len(bbox_annotations) <= j:
                    bbox_annotations.append({})
                bbox_annotations[j] = {
                    "x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2),
                    "damage_class": item["damage"],
                    "part": item["part"],
                    "severity": item["severity"],
                }
            st.divider()

        st.session_state["bbox_annotations"] = bbox_annotations

        col_next, col_back = st.columns([1, 1])
        if col_next.button("Next →", type="primary"):
            st.session_state.correction_step = 4
            st.rerun()
        if col_back.button("← Back"):
            st.session_state.correction_step = 2
            st.rerun()

    # ── Step 4: Submit ────────────────────────────────────────────────────────
    elif step == 4:
        st.subheader("Review and submit")

        actions = st.session_state.correction_actions
        bbox_annotations = st.session_state.get("bbox_annotations", [])
        final_map = st.session_state.working_damage_map

        n_kept    = sum(1 for a in actions if a["action"] == "keep")
        n_edited  = sum(1 for a in actions if a["action"] == "edit")
        n_removed = sum(1 for a in actions if a["action"] == "remove")
        n_added   = sum(1 for a in actions if a["action"] == "add")
        n_bboxes  = len(bbox_annotations)

        st.markdown(
            f"✓ Kept: **{n_kept}** | "
            f"✎ Edited: **{n_edited}** | "
            f"✗ Removed: **{n_removed}** | "
            f"+ Added: **{n_added}** | "
            f"📍 Bboxes: **{n_bboxes}**"
        )

        if final_map:
            st.subheader("Final damage map")
            st.table({
                "Damage": [e["damage"] for e in final_map],
                "Part": [e["part"] for e in final_map],
                "Severity": [e["severity"] for e in final_map],
            })

        intern_name = st.text_input("Your name/ID (optional)", key="intern_name_input")
        notes = st.text_area("Notes (optional)", key="submit_notes")

        if st.button("Save Corrections & Approve", type="primary"):
            session_id = st.session_state.session_id
            correction_payload = {
                "correction_actions": actions,
                "bbox_annotations": bbox_annotations,
                "final_damage_map": final_map,
                "annotated_by": intern_name or None,
                "notes": notes or None,
            }

            try:
                save_resp = requests.post(
                    f"{API_BASE}/session/{session_id}/save_correction",
                    json=correction_payload,
                    timeout=15,
                )
                if save_resp.status_code != 200:
                    st.error(f"Save correction failed ({save_resp.status_code}): {save_resp.text}")
                    st.stop()
            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to backend.")
                st.stop()

            try:
                approve_resp = requests.post(
                    f"{API_BASE}/session/{session_id}/approve",
                    json={
                        "damage_part_map": final_map,
                        "correction_notes": notes or None,
                    },
                    timeout=15,
                )
                if approve_resp.status_code == 200:
                    st.session_state.report = approve_resp.json()
                    st.session_state.stage = "done"
                    st.rerun()
                else:
                    st.error(f"Approve failed ({approve_resp.status_code}): {approve_resp.text}")
            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to backend.")

        col_back, col_reject = st.columns([1, 1])
        if col_back.button("← Back"):
            st.session_state.correction_step = 3 if n_added > 0 else 2
            st.rerun()
        if col_reject.button("Reject & Re-run"):
            _reset()
            st.rerun()


def render_done() -> None:
    report = st.session_state.report

    st.title("Vehicle Damage Assessment")
    st.success("✓ Report approved and saved to feedback log.")

    if st.session_state.get("uploaded_image_bytes"):
        st.image(
            st.session_state.uploaded_image_bytes,
            caption="Submitted Image",
            use_container_width=True,
        )

    st.subheader("Approved Damage Summary")

    entries = report.get("damage_part_map", [])
    if entries:
        st.table({
            "Damage":         [e["damage"] for e in entries],
            "Part":           [e["part"] for e in entries],
            "Severity":       [e["severity"] for e in entries],
            "Min Cost (INR)": [f"₹{e['cost_min']:,}" for e in entries],
            "Max Cost (INR)": [f"₹{e['cost_max']:,}" for e in entries],
        })

    col1, col2 = st.columns(2)
    col1.metric("Total Min (INR)", f"₹{report['total_min']:,}")
    col2.metric("Total Max (INR)", f"₹{report['total_max']:,}")

    st.info(f"Decision: {report.get('approval_decision', 'HUMAN_APPROVED')}")

    st.download_button(
        "Download Final Report JSON",
        data=json.dumps(report, indent=2),
        file_name="approved_damage_report.json",
        mime="application/json",
    )

    if st.button("Assess Another Image"):
        _reset()
        st.rerun()


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(
        page_title="Vehicle Damage Assessment",
        page_icon="🚗",
        layout="wide",
    )
    _init_state()

    stage = st.session_state.stage
    if stage == "upload":
        render_upload()
    elif stage == "polling":
        render_polling()
    elif stage == "approved":
        render_approved()
    elif stage == "escalated":
        render_escalated()
    elif stage == "done":
        render_done()
    else:
        _reset()
        st.rerun()


main()
