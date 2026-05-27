#!/usr/bin/env python3
"""
Pipeline Context Builder
Compiles coordinate detections into structured prompt context for VLM guidance.
"""

from typing import List, Dict, Any

def build_vlm_guided_context(
    damage_detections: List[Dict[str, Any]],
    part_segmentations: List[Dict[str, Any]],
    plate_info: Dict[str, Any]
) -> str:
    """
    Assembles raw model coordinates and class segmentations into a crisp
    prompt injection block to guide the VLM's visual attention.
    """
    context = []
    context.append("=== PRIMARY MODEL SUGGESTIONS & GROUND TRUTH DETECTIONS ===")
    
    # 1. License Plate
    plate_no = plate_info.get("license_plate", "Unknown")
    plate_conf = plate_info.get("confidence_score", 0.0)
    context.append(f"- Detected License Plate: {plate_no} (Confidence: {plate_conf:.2f})")
    
    # 2. Part Segmentations
    context.append("\n- Identified Panel Segmentations:")
    if not part_segmentations:
        context.append("  None detected.")
    for idx, seg in enumerate(part_segmentations):
        part = seg.get("part", "Unknown")
        conf = seg.get("segment_confidence", 0.0)
        area = seg.get("damage_area_percent", 0.0)
        context.append(f"  {idx+1}. Part: {part} | Confidence: {conf:.2f} | Panel Surface Area: {area:.1f}%")
        
    # 3. Damage Detections
    context.append("\n- Visual Bounding-Box Damage Coordinates:")
    if not damage_detections:
        context.append("  No visual damage bounding boxes detected.")
    for idx, dmg in enumerate(damage_detections):
        part = dmg.get("part", "Unknown")
        dtype = dmg.get("damage_type", "Unknown")
        sev = dmg.get("severity", "Moderate")
        conf = dmg.get("confidence", 0.0)
        box = dmg.get("box", [])
        context.append(f"  {idx+1}. Part: {part} | Damage: {dtype} | Severity: {sev} | Confidence: {conf:.2f} | Box Coordinates: {box}")

    context.append("\n=== TASK DIRECTIONS FOR THE VLM ===")
    context.append("Use the above coordinate suggestions and identified panel regions to guide your visual attention.")
    context.append("Audit the suggested damage items, check for spatial agreement, verify license plate numbers, and output your final assessment strictly conforming to the requested claims JSON schema.")

    return "\n".join(context)
