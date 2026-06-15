"""
Image annotation helpers used by the correction UI.

  - generate_annotated_image: draw numbered, class-coloured boxes on an upload.
  - write_yolo_labels: persist human-corrected boxes as a YOLO fine-tune sample.
"""

import shutil
from pathlib import Path

import cv2

# BGR colours per damage class (OpenCV uses BGR).
CLASS_COLORS = {
    "dent":          (221, 138,  55),
    "scratch":       (117, 158,  29),
    "crack":         ( 23, 117, 186),
    "glass_shatter": (126,  83, 212),
    "lamp_broken":   ( 48,  90, 216),
    "tire_flat":     (128, 135, 136),
}
DEFAULT_COLOR = (128, 128, 128)

YOLO_CLASS_MAP = {
    "dent": 0, "scratch": 1, "crack": 2,
    "glass_shatter": 3, "lamp_broken": 4, "tire_flat": 5,
}


def generate_annotated_image(
    image_path: str,
    detections: list,
    output_dir: str = "data/uploads/annotated",
) -> str:
    """
    Draw numbered bounding boxes on the image, colour-coded by damage class.
    Low-confidence model boxes are drawn as corner ticks; human boxes are solid.
    Saves a JPEG and returns its path.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    h, w = img.shape[:2]

    for det in detections:
        if hasattr(det, "bbox"):
            bbox, cls, idx, conf, src = (
                det.bbox, det.damage, det.index, det.confidence, det.source,
            )
        else:
            bbox = det.get("bbox", [0, 0, 0, 0])
            cls = det.get("damage", "dent")
            idx = det.get("index", 0)
            conf = det.get("confidence", 0.0)
            src = det.get("source", "yolo")

        if all(v == 0.0 for v in bbox):
            continue

        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        color = CLASS_COLORS.get(cls, DEFAULT_COLOR)
        thickness = 3 if src == "human" else 2

        if conf < 0.5 and src != "human":
            corner_len = min(20, max(1, (x2 - x1) // 4), max(1, (y2 - y1) // 4))
            for cx, cy, dx, dy in [(x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)]:
                cv2.line(img, (cx, cy), (cx + dx * corner_len, cy), color, 3)
                cv2.line(img, (cx, cy), (cx, cy + dy * corner_len), color, 3)
        else:
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

        badge_r = 14
        badge_x = min(x1 + badge_r + 2, w - badge_r - 2)
        badge_y = (y1 - badge_r - 2) if y1 > badge_r + 10 else (y1 + badge_r + 2)
        badge_y = max(badge_r + 2, min(h - badge_r - 2, badge_y))
        cv2.circle(img, (badge_x, badge_y), badge_r, color, -1)
        cv2.circle(img, (badge_x, badge_y), badge_r, (255, 255, 255), 1)
        label = str(idx)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5 if idx < 10 else 0.4
        text_size = cv2.getTextSize(label, font, font_scale, 2)[0]
        cv2.putText(
            img, label,
            (badge_x - text_size[0] // 2, badge_y + text_size[1] // 2),
            font, font_scale, (255, 255, 255), 2,
        )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = str(out_dir / f"{Path(image_path).stem}_annotated.jpg")
    cv2.imwrite(out_path, img)
    return out_path


def write_yolo_labels(
    image_path: str,
    bbox_annotations: list,
    img_w: int,
    img_h: int,
) -> None:
    """Write a YOLO-format label file and copy the image into the fine-tune dataset."""
    src = Path(image_path)
    if not src.exists():
        return
    img_dir = Path("data/yolo_corrections/images")
    lbl_dir = Path("data/yolo_corrections/labels")
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, img_dir / src.name)

    lines = []
    for ann in bbox_annotations:
        if hasattr(ann, "damage_class"):
            cls_name = ann.damage_class
            x1, y1, x2, y2 = ann.x1, ann.y1, ann.x2, ann.y2
        else:
            cls_name = ann.get("damage_class", "dent")
            x1 = ann.get("x1", 0)
            y1 = ann.get("y1", 0)
            x2 = ann.get("x2", 0)
            y2 = ann.get("y2", 0)
        cls_id = YOLO_CLASS_MAP.get(cls_name, 0)
        lines.append(
            f"{cls_id} {((x1 + x2) / 2) / img_w:.6f} {((y1 + y2) / 2) / img_h:.6f} "
            f"{(x2 - x1) / img_w:.6f} {(y2 - y1) / img_h:.6f}"
        )
    (lbl_dir / (src.stem + ".txt")).write_text("\n".join(lines))
