import logging
from typing import Dict, List

logger = logging.getLogger(__name__)


class PartSegmenter:
    def __init__(self, config_path=None, weights_path=None):
        self.config_path = config_path
        self.weights_path = weights_path
        logger.warning(
            "PartSegmenter: real Grounding DINO + SAM2 not yet implemented. "
            "Returning empty segments — VLM will use visual fallback."
        )

    def predict(self, image_path: str) -> List[Dict]:
        return []
