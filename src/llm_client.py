from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

from openai import APIStatusError, OpenAI, RateLimitError

from .config import (
    DEEPSEEK_BASE_URL,
    DEEPSEEK_REASONER_MAX_TOKENS,
    DEFAULT_DEEPSEEK_MODEL,
    JSON_MODE_PROVIDERS,
    LLM_INSUFFICIENT_BALANCE,
    LLM_LIMIT_ERROR,
    LLM_MAX_COMPLETION_TOKENS,
    LLM_REQUEST_TOO_LARGE,
    OLLAMA_BASE_URL,
    OLLAMA_HOST,
)

Provider = str

DEEPSEEK_REASONER_MODELS = frozenset({"deepseek-reasoner", "deepseek-r1"})


def _is_deepseek_reasoner(model_id: str) -> bool:
    lowered = model_id.lower()
    return lowered in DEEPSEEK_REASONER_MODELS or "reasoner" in lowered


HF_CREDITS_ERROR = "HF_CREDITS_DEPLETED"
HF_MODEL_UNSUPPORTED = "HF_MODEL_UNSUPPORTED"
HF_ERROR = "HF_ERROR"


DEEPSEEK_BALANCE_URL = "https://api.deepseek.com/user/balance"


def deepseek_balance_status(api_key: str | None) -> dict:
    if not api_key:
        return {"ok": False, "error": "no_key"}

    try:
        request = urllib.request.Request(
            DEEPSEEK_BALANCE_URL,
            headers={"Authorization": f"Bearer {api_key}"},
        )
        with urllib.request.urlopen(request, timeout=15) as response:
            parsed = json.loads(response.read().decode())
    except urllib.error.HTTPError as exc:
        return {"ok": False, "error": f"http_{exc.code}"}
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        return {"ok": False, "error": str(exc)}

    balance_infos = parsed.get("balance_infos") or []
    usd = next(
        (item for item in balance_infos if str(item.get("currency", "")).upper() == "USD"),
        balance_infos[0] if balance_infos else {},
    )
    total = str(usd.get("total_balance", "")).strip()
    topped_up = str(usd.get("topped_up_balance", "")).strip()
    currency = str(usd.get("currency", "USD")).upper() or "USD"

    return {
        "ok": True,
        "is_available": bool(parsed.get("is_available")),
        "currency": currency,
        "total_balance": total,
        "topped_up_balance": topped_up,
    }


def create_llm_client(provider: Provider, api_key: str | None = None) -> OpenAI:
    if provider == "ollama":
        return OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")
    if provider == "deepseek":
        return OpenAI(
            base_url=DEEPSEEK_BASE_URL,
            api_key=api_key or os.getenv("DEEPSEEK_API_KEY", ""),
        )
    return OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY", ""))


def _openai_compatible_chat(
    provider: Provider,
    model: str,
    messages: list[dict[str, str]],
    *,
    api_key: str | None,
    temperature: float,
    max_tokens: int | None,
    json_mode: bool,
) -> str:
    reasoner = provider == "deepseek" and _is_deepseek_reasoner(model)
    request_kwargs: dict = {
        "model": model,
        "messages": messages,
    }
    if not reasoner:
        request_kwargs["temperature"] = temperature
    if max_tokens is not None:
        request_kwargs["max_tokens"] = max_tokens
    elif reasoner:
        request_kwargs["max_tokens"] = DEEPSEEK_REASONER_MAX_TOKENS
    if json_mode and not reasoner:
        request_kwargs["response_format"] = {"type": "json_object"}

    client = create_llm_client(provider, api_key)
    try:
        response = client.chat.completions.create(**request_kwargs)
    except RateLimitError as exc:
        raise RuntimeError(LLM_LIMIT_ERROR) from exc
    except APIStatusError as exc:
        if exc.status_code == 413:
            raise RuntimeError(LLM_REQUEST_TOO_LARGE) from exc
        if exc.status_code == 429:
            raise RuntimeError(LLM_LIMIT_ERROR) from exc
        if exc.status_code == 402 and provider == "deepseek":
            raise RuntimeError(LLM_INSUFFICIENT_BALANCE) from exc
        raise
    return response.choices[0].message.content or "{}"


def ollama_status() -> tuple[bool, list[str]]:
    try:
        with urllib.request.urlopen(f"{OLLAMA_HOST}/api/tags", timeout=3) as response:
            data = json.loads(response.read().decode())
        models = [item["name"] for item in data.get("models", [])]
        return True, models
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError):
        return False, []


def chat_complete(
    provider: Provider,
    model: str,
    messages: list[dict[str, str]],
    *,
    api_key: str | None = None,
    temperature: float = 0.4,
    max_tokens: int | None = None,
    json_mode: bool = False,
) -> str:
    if provider == "huggingface":
        from huggingface_hub import InferenceClient
        from huggingface_hub.errors import HfHubHTTPError

        token = api_key or os.getenv("HF_TOKEN")
        client = InferenceClient(model=model, token=token)
        try:
            response = client.chat_completion(
                messages=messages,
                max_tokens=4096,
                temperature=temperature,
            )
        except HfHubHTTPError as exc:
            status = getattr(exc.response, "status_code", None)
            body = (getattr(exc.response, "text", None) or str(exc)).lower()
            if status == 402:
                raise RuntimeError(HF_CREDITS_ERROR) from exc
            if status == 400 and (
                "model_not_supported" in body or "not supported by any provider" in body
            ):
                raise RuntimeError(HF_MODEL_UNSUPPORTED) from exc
            raise RuntimeError(HF_ERROR) from exc
        return response.choices[0].message.content or "{}"

    if provider in {"openai", "deepseek", "ollama"}:
        resolved_model = model
        if provider == "openai" and not model:
            from .config import DEFAULT_MODEL

            resolved_model = DEFAULT_MODEL
        if provider == "deepseek" and not model:
            from .config import DEFAULT_DEEPSEEK_MODEL

            resolved_model = DEFAULT_DEEPSEEK_MODEL
        use_json_mode = json_mode and provider in JSON_MODE_PROVIDERS
        if provider == "deepseek" and _is_deepseek_reasoner(resolved_model):
            use_json_mode = False
        return _openai_compatible_chat(
            provider,
            resolved_model,
            messages,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens or (LLM_MAX_COMPLETION_TOKENS if json_mode else None),
            json_mode=use_json_mode,
        )

    request_kwargs: dict = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    client = create_llm_client(provider, api_key)
    response = client.chat.completions.create(**request_kwargs)
    return response.choices[0].message.content or "{}"
