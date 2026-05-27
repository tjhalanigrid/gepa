#!/usr/bin/env python3
"""
Pydantic Schema Validation Contract
Defines and validates relational models for damages, segmentations, and claims.
"""

from typing import List, Optional
from pydantic import BaseModel, Field

class DamageItem(BaseModel):
    """
    Structured visual damage item returned by models and VLMs.
    """
    part: str = Field(..., description="Affected vehicle part name")
    damage_type: str = Field(..., description="Type of damage detected (e.g. dent, scratch)")
    severity: str = Field("Moderate", description="Severity level: Pristine, Minor, Moderate, Severe")
    supporting_images: List[str] = Field(default_factory=list, description="Associated view filenames")
    confidence: float = Field(0.0, ge=0.0, le=1.0, description="Model prediction confidence score")
    reasoning: Optional[str] = Field(None, description="Visual rationale justifying severity")

class ClaimAnalysisSchema(BaseModel):
    """
    Executive verified vehicle damage insurance claim report.
    """
    claim_id: str = Field(..., description="Unique alphanumeric claims identifier")
    overall_summary: str = Field(..., description="High-level claims summary")
    overall_severity: str = Field(..., description="Pristine | Minor | Moderate | Severe")
    damages: List[DamageItem] = Field(default_factory=list, description="Visual detections list")
    view_consistency_notes: Optional[str] = Field(None, description="Spatial correlation notes")
    cost_estimation: Optional[dict] = Field(None, description="Detailed repair and refinish cost breakdown sheet")
