#!/usr/bin/env python3
"""
FastAPI Claims Assessment Route
Handles claims uploads, executes modular VLM orchestrator pipeline, and outputs verified claims.
"""

import os
import shutil
import uuid
from typing import List
from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel

# Import orchestrator & schema
try:
    from pipeline.orchestrator import ClaimsPipelineOrchestrator
    from pipeline.schema import ClaimAnalysisSchema
except ImportError:
    # Fallback in case of absolute path runs
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
    from pipeline.orchestrator import ClaimsPipelineOrchestrator
    from pipeline.schema import ClaimAnalysisSchema

router = APIRouter(prefix="/assessment", tags=["Claims Assessment"])

# Setup centralized uploads folder
UPLOAD_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../uploads"))
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Initialize global orchestrator
orchestrator = ClaimsPipelineOrchestrator()

@router.post("/evaluate", response_model=ClaimAnalysisSchema)
async def evaluate_vehicle_claim(
    file: UploadFile = File(..., description="Image representing the vehicle damage")
):
    """
    Receives an image representing the vehicle damage, runs our pipeline orchestrator,
    computes estimated repair sheets, and returns the validated Claims Schema structure.
    """
    # 1. Save uploaded file securely
    file_extension = os.path.splitext(file.filename)[1]
    claim_id = f"CLM{uuid.uuid4().hex[:6].upper()}"
    saved_filename = f"{claim_id}_damage{file_extension}"
    saved_path = os.path.join(UPLOAD_DIR, saved_filename)
    
    try:
        with open(saved_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded claims image: {e}")

    # 2. Execute claims orchestrator pipeline
    try:
        logger_info = f"[API Routers] Triggering Orchestrator for claim {claim_id} with file: {saved_filename}"
        print(logger_info)
        
        # Run sequential modules & local Ollama VLM
        claim_analysis = orchestrator.execute(image_paths=[saved_path], claim_id=claim_id)
        
        # Cleanup temporary files (optional, we keep it for database audit logs)
        return claim_analysis
        
    except Exception as e:
        # Gracefully handle VLM or orchestration failure
        print(f"[API Routers Error] Claims pipeline failed: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Claims pipeline processing error: {str(e)}"
        )
