#!/usr/bin/env python3
"""
Pipeline Orchestrator
Coordinates and executes modular ML models sequentially, builder context,
queries VLM reasoning, executes cost sheet estimations, and validates the output.
"""

import os
import sys
import yaml

# Add root folder to sys.path to allow clean absolute package imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from shared.logger import get_logger
from models.damage_detection.infer import DamageDetector
from models.part_segmentation.infer import PartSegmenter
from models.plate_rc_detection.infer import PlateRCDetector
from models.vlm_reasoning.vlm_client import VLMClient
from pipeline.context_builder import build_vlm_guided_context
from pipeline.schema import ClaimAnalysisSchema
from experiments.estimator import VehicleCostEstimator

logger = get_logger("Orchestrator")

class ClaimsPipelineOrchestrator:
    """
    Orchestration manager for the sequential claims assessment flow.
    """
    def __init__(self, config_path: str = None):
        if not config_path:
            config_path = os.path.join(ROOT_DIR, "configs/pipeline_config.yaml")
        
        self.config_path = config_path
        
        # Load yaml configs
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                self.config = yaml.safe_load(f)
            logger.info(f"Loaded pipeline configuration from {config_path}")
        except Exception as e:
            logger.warning(f"Failed to load pipeline yaml config: {e}. Using fallback defaults.")
            self.config = {}

        # 1. Initialize modular ML interfaces
        cfg_models = self.config.get("models", {})
        
        self.damage_detector = DamageDetector(
            config_path=os.path.join(ROOT_DIR, "models/damage_detection/configs/data.yaml"),
            weights_path=os.path.join(ROOT_DIR, cfg_models.get("damage_detection", {}).get("weights_path", ""))
        )
        
        self.part_segmenter = PartSegmenter(
            weights_path=os.path.join(ROOT_DIR, cfg_models.get("part_segmentation", {}).get("weights_path", ""))
        )
        
        self.plate_detector = PlateRCDetector(
            weights_path=os.path.join(ROOT_DIR, cfg_models.get("plate_rc_detection", {}).get("weights_path", ""))
        )

        # 2. Initialize VLM client
        cfg_vlm = cfg_models.get("vlm_reasoning", {})
        self.vlm_client = VLMClient(
            model_name=cfg_vlm.get("model_name", "qwen2.5vl:7b"),
            ollama_url=cfg_vlm.get("ollama_url", "http://localhost:11434")
        )

    def execute(self, image_paths: list, claim_id: str = "CLM001") -> ClaimAnalysisSchema:
        """
        Executes sequential pipeline flow.
        """
        logger.info(f"=== Starting Claims Assessment Pipeline for claim {claim_id} ===")
        
        if not image_paths:
            raise ValueError("No input image paths supplied to the orchestrator.")

        # Take first image as reference image for primary models
        primary_image = image_paths[0]
        logger.info(f"Running primary object detection models on reference image: {os.path.basename(primary_image)}")

        # Step 1: Run damage detector
        logger.info("Step 1: Running Damage Bounding-box Detector...")
        damage_detections = self.damage_detector.predict(primary_image)

        # Step 2: Run part segmenter
        logger.info("Step 2: Running Vehicle Part Segmenter...")
        part_segmentations = self.part_segmenter.predict(primary_image)

        # Step 3: Run license plate detector
        logger.info("Step 3: Running License Plate & Recurrence Detector...")
        plate_info = self.plate_detector.predict(primary_image)

        # Step 4: Assemble prompt coordinate context
        logger.info("Step 4: Building guided visual coordinate context package...")
        guided_context = build_vlm_guided_context(
            damage_detections=damage_detections,
            part_segmentations=part_segmentations,
            plate_info=plate_info
        )

        # Step 5: Query Local VLM client with coordinate guidance injected
        logger.info("Step 5: Invoking local Ollama VLM claims analysis...")
        
        # Inject the compiled context to guide the VLM's attention
        vlm_prompt = f"{guided_context}\n\n"
        self.vlm_client.default_prompt = f"{vlm_prompt}{self.vlm_client.default_prompt}"
        
        vlm_claim_json = self.vlm_client.analyze_claim(image_paths, claim_id=claim_id)

        # Step 6: Compute insurance cost sheet breakdown
        logger.info("Step 6: Running Cost Estimator Engine...")
        if vlm_claim_json and not vlm_claim_json.get("parsing_failed"):
            try:
                costs = VehicleCostEstimator.estimate_claim_costs(vlm_claim_json)
                vlm_claim_json["cost_estimation"] = costs
            except Exception as ce:
                logger.error(f"Cost estimation failed: {ce}")
        
        # Step 7: Enforce validated Pydantic schema contract
        logger.info("Step 7: Validating analysis against Pydantic schema contract...")
        validated_schema = ClaimAnalysisSchema(**vlm_claim_json)
        
        logger.info(f"=== Successfully completed claims pipeline for {claim_id} ===\n")
        return validated_schema

if __name__ == "__main__":
    # Self testing routine
    print("Testing orchestrator initialization...")
    orch = ClaimsPipelineOrchestrator()
    print("Orchestrator initialized successfully.")
