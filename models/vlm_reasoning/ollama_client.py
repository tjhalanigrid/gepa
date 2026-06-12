"""
models/vlm_reasoning/ollama_client.py

Thin stdlib-only wrapper around the Ollama REST API (localhost:11434).
No external dependencies — uses urllib, base64, json only.

Supported endpoints:
  GET  /api/tags  — list available models (health check)
  POST /api/chat  — multimodal chat completion with optional vision

Message format expected by chat():
  [
    {"role": "system",    "content": "..."},
    {"role": "user",      "content": "text...", "images": ["base64str"]},
    {"role": "assistant", "content": "..."},
  ]
The 'images' key is optional and only valid on user messages.
"""

import base64
import json
import logging
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL    = "qwen3.5:9b"


def encode_image(image_path: str) -> str:
    """
    Read an image file and return its base64-encoded string.

    Ollama vision API accepts images as base64 strings in the 'images' field
    of a user message. JPEG and PNG are supported.

    Raises FileNotFoundError if the path does not exist.
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def check_health(
    base_url: str = DEFAULT_BASE_URL,
    model: str    = DEFAULT_MODEL,
) -> bool:
    """
    Return True if Ollama is reachable and the specified model is available.

    Hits GET /api/tags and checks the returned model name list.
    Times out in 5s to avoid blocking startup on slow machines.
    """
    try:
        req = urllib.request.Request(
            f"{base_url}/api/tags",
            headers={"Accept": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())

        available = [m["name"] for m in data.get("models", [])]
        if model in available:
            logger.info(f"Ollama healthy. Model '{model}' available at {base_url}.")
            return True

        logger.warning(
            f"Ollama is running but model '{model}' not found. "
            f"Available models: {available}. "
            f"Pull it with: ollama pull {model}"
        )
        return False

    except urllib.error.URLError as e:
        logger.error(f"Ollama unreachable at {base_url}: {e}. Start with: ollama serve")
        return False
    except Exception as e:
        logger.error(f"Ollama health check error: {e}")
        return False


def chat(
    messages: List[dict],
    model: str    = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    temperature: float = 0.7,
    num_predict: int   = 512,
    think: Optional[bool] = False,
    top_p: Optional[float] = 0.8,
    top_k: Optional[int] = 20,
    presence_penalty: Optional[float] = 1.5,
    num_ctx: Optional[int] = 8192,
) -> str:
    """
    POST /api/chat and return the assistant content string.

    Args:
        messages:    Conversation history in Ollama format (see module docstring).
        model:       Ollama model tag, e.g. "qwen3.5:9b".
        base_url:    Ollama server URL.
        temperature: Sampling temperature.
        num_predict: Maximum tokens to generate.
        think:       Thinking mode. DEFAULT False — Qwen3.5 thinking mode burns the
                     whole num_predict budget on <think> and never emits the JSON
                     action, so we disable it for this structured CodeAct task.
        top_p, top_k, presence_penalty: Qwen3.5 instruct (non-thinking) sampling
                     defaults (model card "Best Practices"). presence_penalty curbs
                     the endless-repetition failure mode.

    Returns:
        The assistant content string (thinking content excluded when think=False).

    Raises:
        RuntimeError: On HTTP errors or connection failures.
    """
    options: dict = {"temperature": temperature, "num_predict": num_predict}
    if num_ctx is not None:
        # Ollama default context is small (~4096); images cost ~1700 tokens each, so
        # a multi-image agent run evicts the system prompt. Widen it.
        options["num_ctx"] = num_ctx
    if top_p is not None:
        options["top_p"] = top_p
    if top_k is not None:
        options["top_k"] = top_k
    if presence_penalty is not None:
        options["presence_penalty"] = presence_penalty

    payload: dict = {
        "model":    model,
        "messages": messages,
        "stream":   False,
        "options":  options,
    }
    if think is not None:
        payload["think"] = think    # Ollama thinking-model toggle

    body = json.dumps(payload).encode("utf-8")
    req  = urllib.request.Request(
        f"{base_url}/api/chat",
        data    = body,
        headers = {"Content-Type": "application/json"},
        method  = "POST",
    )

    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            result = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        raise RuntimeError(
            f"Ollama HTTP {e.code} at {base_url}/api/chat. "
            f"Body: {e.read()[:200]}"
        ) from e
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"Ollama connection failed at {base_url}: {e}. "
            "Is Ollama running? Try: ollama serve"
        ) from e
    except Exception as e:
        raise RuntimeError(f"Ollama chat error: {e}") from e

    elapsed    = round(time.time() - t0, 2)
    content    = result.get("message", {}).get("content", "")
    n_eval     = result.get("eval_count", 0)
    n_prompt   = result.get("prompt_eval_count", 0)
    total_dur  = result.get("total_duration", 0) / 1e9   # ns → s

    logger.info(
        f"Ollama chat | model={model} | prompt_tok={n_prompt} "
        f"| new_tok={n_eval} | wall={elapsed}s | model_time={total_dur:.1f}s"
    )
    logger.debug(f"Ollama raw response (first 200): {content[:200]}")
    return content
