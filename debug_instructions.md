# Claude Code — Fix Canvas Bbox Coordinates + Smart Detection Merging

## Mode
**Plan mode first.** List every file and function to be changed.
Do not begin editing until plan is confirmed.

---

## Pre-flight reads

Read these files in full:
- `shared/bbox_canvas.py`           — full HTML/JS component, especially toSvg(), updateScale(), onImgLoad()
- `dashboard/app.py`                — _to_canvas_dets(), render_bbox_canvas() call, image_width/height handling
- `pipeline/orchestrator.py`        — _run_yolo_eagerly(), run(), detections_with_bbox building

Also run:
```bash
# Check what img_width and img_height are stored as in session state
grep -n "image_width\|image_height\|img_w\|img_h" dashboard/app.py | head -20
grep -n "image_width\|image_height" pipeline/orchestrator.py | head -20

# Check canvas toSvg and updateScale functions
grep -n "toSvg\|updateScale\|scaleX\|scaleY\|offX\|offY\|getBoundingClientRect" \
  shared/bbox_canvas.py | head -30

# Check what bbox values YOLO actually returns vs image dimensions
python3 -c "
import glob, yaml
samples = glob.glob('data/examples/*.jpg') + glob.glob('data/uploads/*.jpg')
if samples:
    from PIL import Image
    with Image.open(samples[0]) as img:
        print('Image size:', img.size)
    from models.damage_detection import run
    r = run(samples[0], {
        'weights_path': 'models/damage_detection/models/best.pt',
        'confidence_threshold': 0.15,
        'device': 'cpu',
    })
    for d in r.get('detections', []):
        print(d['class'], d['confidence'], d['bbox'])
"
```

Report findings — especially the actual bbox values vs image dimensions,
and what `updateScale()` does with the image position.

---

## Problem 1: Canvas bbox coordinates are wrong

### Root cause

The canvas HTML component has this structure:

```
#canvas-wrap (full width container, e.g. 970px wide)
  └─ #bg-img (the image, rendered smaller, e.g. 510px wide, centered)
  └─ #svg-layer (covers full canvas-wrap, not just the image)
```

`updateScale()` calculates:
```javascript
offX = r.left - wrap.left   // horizontal offset of image within container
offY = r.top  - wrap.top    // vertical offset of image within container
scaleX = r.width  / ORIG_W  // pixels per original pixel, horizontal
scaleY = r.height / ORIG_H  // pixels per original pixel, vertical
```

`toSvg(bx, by)` converts original bbox coords to SVG coords:
```javascript
[bx * scaleX + offX, by * scaleY + offY]
```

**The bug:** `updateScale()` is called in `onImgLoad()` but the image may
not be fully laid out yet when `onImgLoad` fires — `getBoundingClientRect()`
returns zero or incorrect values. The bboxes get drawn at position 0,0 scaled
by wrong offsets.

Additionally, the SVG layer covers the full container but bboxes are drawn
relative to the image position, which only works if `offX/offY` are correct
at draw time.

### Fix: Force re-layout before reading bbox, add MutationObserver

**File:** `shared/bbox_canvas.py`

In the `<script>` section, replace `onImgLoad()` with a more robust version:

```javascript
function onImgLoad() {
  imgEl = document.getElementById('bg-img');
  svgEl = document.getElementById('svg-layer');

  // Force browser to complete layout before reading dimensions
  // Use requestAnimationFrame to ensure paint has occurred
  requestAnimationFrame(function() {
    requestAnimationFrame(function() {
      updateScale();
      populateSelects();
      render();

      // Watch for resize events that would invalidate the scale
      const resizeObs = new ResizeObserver(function() {
        updateScale();
        render();
      });
      resizeObs.observe(document.getElementById('canvas-wrap'));
    });
  });
}

function updateScale() {
  if (!imgEl) return;
  const imgRect  = imgEl.getBoundingClientRect();
  const wrapRect = document.getElementById('canvas-wrap').getBoundingClientRect();

  // If image has zero size, it hasn't rendered yet — skip
  if (imgRect.width === 0 || imgRect.height === 0) {
    setTimeout(updateScale, 100);
    return;
  }

  offX   = imgRect.left  - wrapRect.left;
  offY   = imgRect.top   - wrapRect.top;
  scaleX = imgRect.width  / ORIG_W;
  scaleY = imgRect.height / ORIG_H;

  // Update SVG viewBox to match wrap dimensions
  const wrap = document.getElementById('canvas-wrap');
  svgEl.setAttribute('viewBox',
    `0 0 ${wrap.offsetWidth} ${wrap.offsetHeight}`
  );
}
```

Also update `toSvg()` and `toBbox()` to be explicit about the transform:

```javascript
function toSvg(bx, by) {
  return [
    Math.round(bx * scaleX + offX),
    Math.round(by * scaleY + offY)
  ];
}

function toBbox(sx, sy) {
  return [
    Math.round((sx - offX) / scaleX),
    Math.round((sy - offY) / scaleY)
  ];
}
```

Also add a debug overlay that can be toggled — it shows the current scale
values and image position. Add a small `[debug]` link in the toolbar that
toggles a text overlay showing `scaleX, scaleY, offX, offY`. This helps
diagnose future positioning issues without code changes:

```javascript
// In toolbar HTML:
// <button class="tbtn" onclick="toggleDebug()" style="font-size:10px">dbg</button>

let _debug = false;
function toggleDebug() {
  _debug = !_debug;
  render();
}

// In render(), at the end:
if (_debug) {
  const txt = svgEl.ownerDocument.createElementNS('http://www.w3.org/2000/svg','text');
  txt.setAttribute('x', 8);
  txt.setAttribute('y', 20);
  txt.setAttribute('fill', '#0f0');
  txt.setAttribute('font-size', '11');
  txt.setAttribute('font-family', 'monospace');
  txt.textContent = `scale=${scaleX.toFixed(3)},${scaleY.toFixed(3)} off=${Math.round(offX)},${Math.round(offY)} img=${ORIG_W}x${ORIG_H}`;
  svgEl.appendChild(txt);
}
```

### Fix: Constrain canvas-wrap height and image layout

**File:** `shared/bbox_canvas.py`

The canvas-wrap uses `display:flex; align-items:center; justify-content:center`
which centers the image but makes offset calculation dependent on the image
being fully rendered. Change the image to `object-fit:contain` with explicit
dimensions so the layout is deterministic:

In the `<style>` section, update:

```css
#canvas-wrap {
  position: relative;
  overflow: hidden;
  cursor: default;
  background: #111;
  /* Remove flex centering — use object-fit instead */
}

#bg-img {
  display: block;
  width: 100%;
  height: 100%;
  object-fit: contain;
  object-position: center center;
  user-select: none;
  pointer-events: none;
}

#svg-layer {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  pointer-events: none;
}
```

With `object-fit: contain` and `width:100%; height:100%`, the image fills
the container while maintaining aspect ratio. The image element's
`getBoundingClientRect()` will equal the container's rect, so `offX=0,
offY=0`. The scale becomes simply `containerWidth / ORIG_W`.

Update `updateScale()` to handle this layout:

```javascript
function updateScale() {
  if (!imgEl) return;
  const wrap = document.getElementById('canvas-wrap');
  const wrapW = wrap.offsetWidth;
  const wrapH = wrap.offsetHeight;

  if (wrapW === 0 || wrapH === 0) {
    setTimeout(updateScale, 100);
    return;
  }

  // With object-fit:contain, compute the actual rendered image size
  const imgAspect  = ORIG_W / ORIG_H;
  const wrapAspect = wrapW  / wrapH;

  let rendW, rendH;
  if (imgAspect > wrapAspect) {
    // Image is wider relative to container — constrained by width
    rendW = wrapW;
    rendH = wrapW / imgAspect;
  } else {
    // Image is taller relative to container — constrained by height
    rendH = wrapH;
    rendW = wrapH * imgAspect;
  }

  // Center offset within the wrap
  offX   = (wrapW - rendW) / 2;
  offY   = (wrapH - rendH) / 2;
  scaleX = rendW / ORIG_W;
  scaleY = rendH / ORIG_H;

  svgEl.setAttribute('viewBox', `0 0 ${wrapW} ${wrapH}`);
}
```

This computes the render size mathematically from `ORIG_W / ORIG_H` and the
container dimensions — no reliance on `getBoundingClientRect()` timing.
The result is always correct regardless of when it is called.

---

## Problem 2: Smart merging of overlapping same-class detections

### Root cause

YOLO with `confidence_threshold: 0.15` returns many overlapping boxes for
the same damage region. Standard NMS during YOLO inference uses
`iou_threshold: 0.45` which already suppresses most overlaps, but at very
low confidence thresholds many near-duplicate boxes survive.

The screenshot shows boxes 1, 2, 3, 4 all as `dent` in a small cluster.
These should be merged into one detection representing the full damage area.

### Fix: Add _merge_overlapping_detections() to orchestrator.py

**File:** `pipeline/orchestrator.py`

Add this function. Call it after YOLO runs, before building `detections_with_bbox`:

```python
def _merge_overlapping_detections(
    detections: list,
    iou_threshold: float = 0.30,
    same_class_only: bool = True,
) -> list:
    """
    Merges overlapping detections of the same class using a greedy
    IoU-based algorithm.

    Algorithm:
    1. Sort detections by confidence descending
    2. For each detection, check if it overlaps > iou_threshold with any
       already-accepted detection of the same class
    3. If yes: merge (take union bbox, keep higher confidence, average cost)
    4. If no: accept as new detection

    After merging, re-index so indices are 1..N.

    Args:
        detections: list of dicts with 'bbox', 'class', 'confidence'
        iou_threshold: minimum IoU to trigger merge (0.30 = 30% overlap)
        same_class_only: only merge detections of the same class

    Returns: deduplicated list of detections
    """
    if not detections:
        return []

    def _iou(a: list, b: list) -> float:
        """IoU of two bboxes [x1,y1,x2,y2]."""
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        inter = (ix2 - ix1) * (iy2 - iy1)
        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    def _union_bbox(a: list, b: list) -> list:
        """Bounding box that contains both a and b."""
        return [
            min(a[0], b[0]),
            min(a[1], b[1]),
            max(a[2], b[2]),
            max(a[3], b[3]),
        ]

    # Sort by confidence descending — highest confidence is kept as primary
    sorted_dets = sorted(
        detections,
        key=lambda d: d.get("confidence", 0),
        reverse=True
    )

    merged = []

    for det in sorted_dets:
        det_bbox = det.get("bbox", [0, 0, 0, 0])
        det_cls  = det.get("class", "")
        merged_into = None

        for existing in merged:
            ex_cls  = existing.get("class", "")
            ex_bbox = existing.get("bbox", [0, 0, 0, 0])

            # Only merge same class (unless same_class_only=False)
            if same_class_only and det_cls != ex_cls:
                continue

            iou = _iou(det_bbox, ex_bbox)
            if iou >= iou_threshold:
                # Merge: union bbox, keep higher confidence, note merge
                existing["bbox"]       = _union_bbox(ex_bbox, det_bbox)
                # Confidence stays at the higher value (existing is already higher)
                existing["_merge_count"] = existing.get("_merge_count", 1) + 1
                merged_into = existing
                break

        if merged_into is None:
            # Not merged — add as new detection
            new_det = dict(det)
            new_det["_merge_count"] = 1
            merged.append(new_det)

    # Clean up internal keys and re-index
    result = []
    for det in merged:
        clean = {k: v for k, v in det.items() if not k.startswith("_")}
        result.append(clean)

    logger.info(
        f"Merged {len(detections)} detections → {len(result)} "
        f"(removed {len(detections)-len(result)} overlapping boxes, "
        f"IoU threshold={iou_threshold})"
    )

    return result
```

### Wire _merge_overlapping_detections() into _run_yolo_eagerly()

**File:** `pipeline/orchestrator.py`

In `_run_yolo_eagerly()`, after `detections = result.get("detections", [])`,
add:

```python
# Merge overlapping same-class detections
if len(detections) > 1:
    detections = _merge_overlapping_detections(
        detections,
        iou_threshold=0.30,   # 30% overlap triggers merge
        same_class_only=True,
    )
    total = len(detections)
```

This runs before the annotated image is drawn, so the annotated image will
already show merged boxes.

### Update global_config.yaml with merge threshold

**File:** `configs/global_config.yaml`

Add to the `damage_detection` block:

```yaml
damage_detection:
  weights_path: "models/damage_detection/models/best.pt"
  confidence_threshold: 0.15
  device: "cpu"
  merge_iou_threshold: 0.30    # IoU threshold for merging overlapping boxes
  merge_same_class_only: true  # only merge boxes of the same damage class
```

---

## Verification

```bash
# 1. Test _merge_overlapping_detections with simulated overlapping boxes
python3 -c "
from pipeline.orchestrator import _merge_overlapping_detections

# Simulate the screenshot: 4 overlapping dent boxes + 1 crack + 1 scratch
test_dets = [
    {'class': 'dent',    'confidence': 0.89, 'bbox': [260, 155, 380, 260]},
    {'class': 'dent',    'confidence': 0.73, 'bbox': [275, 170, 395, 270]},
    {'class': 'dent',    'confidence': 0.55, 'bbox': [265, 160, 385, 265]},
    {'class': 'dent',    'confidence': 0.37, 'bbox': [270, 165, 375, 255]},
    {'class': 'crack',   'confidence': 0.33, 'bbox': [300, 200, 370, 240]},
    {'class': 'scratch', 'confidence': 0.61, 'bbox': [500, 300, 700, 450]},
]

merged = _merge_overlapping_detections(test_dets, iou_threshold=0.30)
print(f'Input: {len(test_dets)} detections')
print(f'Output: {len(merged)} after merge')
for d in merged:
    print(f'  {d[\"class\"]} conf={d[\"confidence\"]:.2f} bbox={[int(v) for v in d[\"bbox\"]]}')

# All 4 dents should merge into 1
dent_count = sum(1 for d in merged if d['class'] == 'dent')
assert dent_count == 1, f'Expected 1 merged dent, got {dent_count}'
# Crack should remain (overlaps with dent but different class)
crack_count = sum(1 for d in merged if d['class'] == 'crack')
assert crack_count == 1, f'Expected 1 crack, got {crack_count}'
# Scratch should remain (no overlap)
scratch_count = sum(1 for d in merged if d['class'] == 'scratch')
assert scratch_count == 1, f'Expected 1 scratch, got {scratch_count}'
print('PASS: merge logic correct')
"

# 2. Canvas component syntax check
python3 -c "
from shared.bbox_canvas import render_bbox_canvas
print('PASS: bbox_canvas importable')
"

# 3. Verify updateScale uses object-fit math, not getBoundingClientRect
python3 -c "
src = open('shared/bbox_canvas.py').read()
assert 'imgAspect' in src or 'rendW' in src, 'New updateScale not found'
assert 'object-fit: contain' in src or 'object-fit:contain' in src, \
    'object-fit:contain not in CSS'
assert 'getBoundingClientRect' not in src or \
    src.count('getBoundingClientRect') <= 1, \
    'getBoundingClientRect still used for scaling'
print('PASS: canvas uses deterministic object-fit scaling')
"

# 4. End-to-end: run YOLO on a sample and confirm merge happens
python3 -c "
import glob, yaml
with open('configs/global_config.yaml') as f:
    cfg = yaml.safe_load(f)
samples = glob.glob('data/examples/*.jpg') + glob.glob('data/uploads/*.jpg')
if not samples:
    print('SKIP: no samples')
else:
    from pipeline.orchestrator import _run_yolo_eagerly
    r = _run_yolo_eagerly(samples[0], cfg)
    print(f'Detections after merge: {r[\"total_detections\"]}')
    for d in r.get('detections', []):
        print(f'  {d[\"class\"]} conf={d[\"confidence\"]:.2f}')
    print('PASS if fewer same-class detections than before merge')
"

# 5. Submit a job and verify canvas shows correct positions in dashboard
# (visual test — must be verified manually in browser)
pkill -f 'uvicorn backend' 2>/dev/null; sleep 2
uvicorn backend.app:app --host 0.0.0.0 --port 8000 --log-level warning &
sleep 6
echo "Backend up. Submit an image in Streamlit and verify:"
echo "  1. Bbox positions match the 'Detected Damage' tab image"
echo "  2. No duplicate same-class boxes in the same region"
echo "  3. Toggle [dbg] button — confirm scaleX/scaleY are non-zero"
```

---

## Summary of changes

| File | What changes |
|---|---|
| `shared/bbox_canvas.py` | Replace `updateScale()` with deterministic `object-fit:contain` math; update `onImgLoad()` to use `requestAnimationFrame`; update CSS to `object-fit:contain`; add debug overlay toggle |
| `pipeline/orchestrator.py` | Add `_merge_overlapping_detections()` function; call it in `_run_yolo_eagerly()` after YOLO returns; read merge config from `global_config.yaml` |
| `configs/global_config.yaml` | Add `merge_iou_threshold: 0.30` and `merge_same_class_only: true` to `damage_detection` block |

## Files NOT changed

- `dashboard/app.py` — canvas coordinate issue is entirely in bbox_canvas.py
- `pipeline/schema.py` — no schema changes
- `backend/app/main.py` — no route changes
- `shared/sam_mask.py` — unchanged

---

## Critical checks before approving plan

1. The new `updateScale()` must use `ORIG_W` and `ORIG_H` which are set as
   JavaScript variables from the Python `img_width` and `img_height` params.
   Claude Code must confirm these are correctly passed into the HTML template
   as `{img_width}` and `{img_height}`. If they are hardcoded as 1920/1080,
   the scaling will be wrong for different image sizes.

2. `_merge_overlapping_detections()` must be defined BEFORE
   `_run_yolo_eagerly()` in the file since it is called by it.
   If the plan shows it defined after, reject it.

3. The merge runs on the raw YOLO `detections` list (list of dicts with
   `bbox`, `class`, `confidence`). It must NOT run on `detections_with_bbox`
   (list of `DetectionWithBBox` objects). The merge must happen on the raw
   dict format returned by `models.damage_detection.run()`. Confirm the plan
   applies it to the right variable.

4. IoU threshold 0.30 is the starting value. At this threshold, boxes with
   30% area overlap merge. For the screenshot case (4 heavily overlapping
   dent boxes) this will correctly merge all 4. But it may be too aggressive
   for genuinely separate damages close together (e.g. a scratch on the door
   and a scratch on the fender). `same_class_only=True` mitigates this but
   note it in the plan as a tunable parameter.