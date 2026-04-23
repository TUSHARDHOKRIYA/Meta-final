"""
EduPath AI — Unified LLM Client
Team KRIYA | Meta Hackathon 2026

Thin abstraction over the OpenAI-compatible API. All LLM calls across
the platform route through generate_json() or generate_text().
Reads API_BASE_URL, MODEL_NAME, and HF_TOKEN from environment variables.
"""
import os
import json
import logging

logger = logging.getLogger(__name__)


def _get_config():
    """Get LLM configuration from environment variables."""
    api_base_url = os.getenv("API_BASE_URL")
    model_name = os.getenv("MODEL_NAME", "gpt-4o-mini")
    hf_token = os.getenv("HF_TOKEN", "")

    if not api_base_url:
        raise ValueError(
            "API_BASE_URL environment variable not set. "
            "Set it to your LLM endpoint (e.g. https://api.openai.com/v1)"
        )
    return api_base_url, model_name, hf_token


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


def generate_json(system_prompt: str, user_prompt: str) -> dict:
    """Generate structured JSON output from LLM using OpenAI client."""
    client, model_name = _get_client()

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            top_p=0.9,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content
        if not raw or raw.strip() == "":
            logger.error("LLM returned empty response")
            return {}
        return json.loads(raw)
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
    """Check if the required LLM environment variables are configured."""
    return bool(os.getenv("API_BASE_URL"))
