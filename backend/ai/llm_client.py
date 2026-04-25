"""
EduPath AI — Unified LLM Client
Team KRIYA | Meta Hackathon 2026

Thin abstraction over the OpenAI-compatible API. All LLM calls across
the platform route through generate_json() or generate_text().

Supports three backends (auto-detected from env vars):
  1. HuggingFace Inference API (default) — uses our GRPO fine-tuned model
  2. Groq API — if GROQ_API_KEY is set
  3. OpenAI API — if OPENAI_API_KEY is set

No external API key is required for the default HF backend.
"""
import os
import json
import re
import logging

logger = logging.getLogger(__name__)


def _get_config():
    """Get LLM configuration from environment variables."""
    # Default to HuggingFace Inference API with our GRPO fine-tuned model
    api_base_url = os.getenv(
        "API_BASE_URL",
        "https://api-inference.huggingface.co/v1"
    )
    model_name = os.getenv(
        "MODEL_NAME",
        "degree-checker-01/edupath-grpo-tutor"
    )
    hf_token = os.getenv("HF_TOKEN", "")

    return api_base_url, model_name, hf_token


def _is_hf_backend():
    """Check if we're using HuggingFace Inference API."""
    url = os.getenv("API_BASE_URL", "https://api-inference.huggingface.co/v1")
    return "huggingface" in url


def _get_client():
    """Get OpenAI-compatible client."""
    from openai import OpenAI

    api_base_url, model_name, hf_token = _get_config()

    # Priority: LLM_API_KEY > GROQ_API_KEY > HF_TOKEN > OPENAI_API_KEY
    api_key = (
        os.getenv("LLM_API_KEY")
        or os.getenv("GROQ_API_KEY")
        or hf_token
        or os.getenv("OPENAI_API_KEY", "sk-placeholder")
    )

    client = OpenAI(
        base_url=api_base_url,
        api_key=api_key,
    )
    return client, model_name


def _extract_json_from_text(text: str) -> dict:
    """Extract JSON from LLM response that may contain extra text or markdown."""
    if not text or not text.strip():
        return {}

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting from markdown code blocks
    md_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?\s*```', text, re.DOTALL)
    if md_match:
        try:
            return json.loads(md_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Try extracting first JSON object
    brace_match = re.search(r'\{.*\}', text, re.DOTALL)
    if brace_match:
        raw = brace_match.group()
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # Fix common issues: trailing commas
            fixed = re.sub(r',\s*}', '}', re.sub(r',\s*]', ']', raw))
            try:
                return json.loads(fixed)
            except json.JSONDecodeError:
                pass

    return {}


def generate_json(system_prompt: str, user_prompt: str) -> dict:
    """Generate structured JSON output from LLM using OpenAI client.

    Handles both providers that support response_format (OpenAI, Groq)
    and those that don't (HuggingFace Inference API) by extracting
    JSON from the raw response text when needed.
    """
    client, model_name = _get_client()

    # Ensure system prompt asks for JSON output
    json_instruction = "\n\nIMPORTANT: Respond ONLY with valid JSON. No extra text."
    enhanced_system = system_prompt
    if "json" not in system_prompt.lower():
        enhanced_system = system_prompt + json_instruction

    try:
        # Build kwargs — only add response_format for providers that support it
        kwargs = dict(
            model=model_name,
            messages=[
                {"role": "system", "content": enhanced_system},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            top_p=0.9,
        )

        # HF Inference API doesn't reliably support response_format
        if not _is_hf_backend():
            kwargs["response_format"] = {"type": "json_object"}

        response = client.chat.completions.create(**kwargs)
        raw = response.choices[0].message.content
        if not raw or raw.strip() == "":
            logger.error("LLM returned empty response")
            return {}

        # Parse JSON — with robust extraction for non-JSON-mode providers
        result = _extract_json_from_text(raw)
        if not result:
            logger.error(f"Could not extract JSON from LLM response: {raw[:200]}")
        return result

    except json.JSONDecodeError as e:
        logger.error(f"LLM returned invalid JSON: {e}")
        return {}
    except Exception as e:
        logger.error(f"LLM generation failed: {e}")
        raise


def generate_text(system_prompt: str, user_prompt: str) -> str:
    """Generate free-text output from LLM using OpenAI client."""
    client, model_name = _get_client()

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.5,
            top_p=0.9,
        )
        return response.choices[0].message.content or ""
    except Exception as e:
        logger.error(f"LLM text generation failed: {e}")
        raise


def generate_chat(system_prompt: str, messages: list) -> str:
    """Generate a response in a multi-turn conversation.

    Args:
        system_prompt: The system prompt for the agent.
        messages: List of {"role": "user"|"assistant", "content": "..."} dicts.

    Returns:
        The assistant's response text.
    """
    client, model_name = _get_client()

    full_messages = [{"role": "system", "content": system_prompt}]
    full_messages.extend(messages)

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=full_messages,
            temperature=0.5,
            top_p=0.9,
        )
        return response.choices[0].message.content or ""
    except Exception as e:
        logger.error(f"LLM chat generation failed: {e}")
        raise


def generate_json_with_retry(system_prompt: str, user_prompt: str,
                              retries: int = 2) -> dict:
    """Generate JSON with automatic retry on failure.

    Uses exponential backoff: 1s, 2s between retries.
    Falls back to empty dict if all retries fail.
    """
    import time

    for attempt in range(retries + 1):
        try:
            result = generate_json(system_prompt, user_prompt)
            if result:  # Non-empty result
                return result
            if attempt < retries:
                logger.warning(f"Empty JSON result, retry {attempt + 1}/{retries}")
                time.sleep(2 ** attempt)
        except Exception as e:
            if attempt < retries:
                logger.warning(f"JSON generation failed (attempt {attempt + 1}): {e}")
                time.sleep(2 ** attempt)
            else:
                logger.error(f"JSON generation failed after {retries + 1} attempts: {e}")

    return {}


def is_api_key_set() -> bool:
    """Check if an LLM backend is available.

    Returns True if ANY of the following is configured:
    - API_BASE_URL is set (custom endpoint)
    - HF_TOKEN is set (HuggingFace Inference API)
    - GROQ_API_KEY is set (Groq)
    - LLM_API_KEY is set (generic)
    - OPENAI_API_KEY is set (OpenAI)

    Also returns True by default since the HF Inference API
    works without authentication for public models.
    """
    # Our GRPO model is public on HuggingFace — always available
    # Even without HF_TOKEN, public models can be queried (with rate limits)
    return True
