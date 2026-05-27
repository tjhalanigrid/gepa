#!/usr/bin/env python3
"""
VLM Client Interface for Local Ollama Model Queries
Queries Qwen or Gemma vision models locally.
"""

import os
import re
import json
from ollama import chat

# Import shared image optimization helper
try:
    from shared.image_utils import get_optimized_image_paths
except ImportError:
    # Fallback in case of absolute path runs
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
    from shared.image_utils import get_optimized_image_paths

class VLMClient:
    """
    Client wrapper for local Ollama Vision-Language Models.
    """
    def __init__(self, model_name: str = "qwen2.5vl:7b", ollama_url: str = None):
        self.model_name = model_name
        self.ollama_url = ollama_url
        
        # Load prompt template
        current_dir = os.path.dirname(os.path.abspath(__file__))
        prompt_path = os.path.join(current_dir, "prompts/assessment_prompt.txt")
        if os.path.exists(prompt_path):
            with open(prompt_path, "r", encoding="utf-8") as f:
                self.default_prompt = f.read()
        else:
            self.default_prompt = "Perform visual damage inspection and output JSON."

    def clean_json_content(self, content: str) -> str:
        """
        Cleans LLM response to extract pure JSON.
        """
        if not content:
            return ""
        content = content.strip()
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", content)
        if match:
            content = match.group(1).strip()
        if not content.startswith("{") and "{" in content:
            content = content[content.find("{"):]
        if not content.endswith("}") and "}" in content:
            content = content[:content.rfind("}") + 1]
        return content

    def analyze_claim(self, image_paths: list, claim_id: str = "unknown") -> dict:
        """
        Submits optimized images and standard Claims Adjuster prompt to local VLM.
        Returns parsed visual damage JSON dict.
        """
        # Optimize image payload dynamically to prevent Ollama GPU/CPU OOM context crashes
        optimized_paths = get_optimized_image_paths(image_paths)
        print(f"[VLMClient] Optimized image paths from {[os.path.basename(p) for p in image_paths]} to {[os.path.basename(p) for p in optimized_paths]}")

        try:
            response = chat(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": self.default_prompt,
                        "images": optimized_paths
                    }
                ]
            )

            raw_content = response["message"]["content"]
            cleaned_content = self.clean_json_content(raw_content)
            
            try:
                parsed_data = json.loads(cleaned_content)
            except json.JSONDecodeError as je:
                print(f"[VLMClient Error] Failed to parse VLM JSON response: {je}")
                parsed_data = {
                    "claim_id": claim_id,
                    "parsing_failed": True,
                    "parsing_error": str(je),
                    "raw_response": raw_content
                }
            
            parsed_data["claim_id"] = claim_id
            return parsed_data
            
        except Exception as e:
            print(f"[VLMClient Error] Critical failure calling local VLM: {e}")
            return {
                "claim_id": claim_id,
                "parsing_failed": True,
                "parsing_error": str(e),
                "raw_response": ""
            }
