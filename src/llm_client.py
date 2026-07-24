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
    GROQ_BASE_URL,
    GROQ_MAX_COMPLETION_TOKENS,
    GROQ_MODEL_CHAIN,
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


GROQ_MODEL_LABELS = {
    "qwen/qwen3.6-27b": "Qwen 3.6 27B",
    "llama-3.1-8b-instant": "Llama 3.1 8B",
}


def _groq_completion_kwargs(
    model_id: str,
    messages: list[dict[str, str]],
    *,
    temperature: float,
    max_tokens: int | None = None,
    json_mode: bool = False,
) -> dict:
    kwargs: dict = {
        "model": model_id,
        "messages": messages,
        "temperature": temperature,
    }
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    if model_id.startswith("qwen/"):
        kwargs["reasoning_effort"] = "none"
    return kwargs

HF_CREDITS_ERROR = "HF_CREDITS_DEPLETED"
HF_MODEL_UNSUPPORTED = "HF_MODEL_UNSUPPORTED"
HF_ERROR = "HF_ERROR"


def _parse_groq_rate_headers(headers: dict) -> tuple[int | None, int | None]:
    remaining = headers.get("x-ratelimit-remaining-requests")
    limit = headers.get("x-ratelimit-limit-requests")
    return (
        int(remaining) if remaining is not None else None,
        int(limit) if limit is not None else None,
    )


def _check_groq_model_quota(client: OpenAI, model_id: str) -> dict:
    label = GROQ_MODEL_LABELS.get(model_id, model_id)
    entry: dict = {
        "model_id": model_id,
        "label": label,
        "available": False,
        "remaining_requests": None,
        "limit_requests": None,
        "error": None,
    }
    try:
        response = client.chat.completions.with_raw_response.create(
            **_groq_completion_kwargs(
                model_id,
                [{"role": "user", "content": "OK"}],
                temperature=0.1,
                max_tokens=1,
            )
        )
        remaining, limit = _parse_groq_rate_headers(dict(response.headers))
        entry["remaining_requests"] = remaining
        entry["limit_requests"] = limit
        entry["available"] = remaining is None or remaining > 0
    except RateLimitError as exc:
        entry["error"] = "limit"
        response = getattr(exc, "response", None)
        if response is not None:
            remaining, limit = _parse_groq_rate_headers(dict(response.headers))
            entry["remaining_requests"] = remaining
            entry["limit_requests"] = limit
    except APIStatusError as exc:
        if exc.status_code == 429:
            entry["error"] = "limit"
            response = getattr(exc, "response", None)
            if response is not None:
                remaining, limit = _parse_groq_rate_headers(dict(response.headers))
                entry["remaining_requests"] = remaining
                entry["limit_requests"] = limit
        else:
            entry["error"] = str(exc)
    except Exception as exc:
        entry["error"] = str(exc)
    return entry


def groq_quota_status(api_key: str | None) -> dict:
    if not api_key:
        return {"ok": False, "error": "no_key", "models": []}

    client = OpenAI(base_url=GROQ_BASE_URL, api_key=api_key)
    models = [_check_groq_model_quota(client, model_id) for model_id in GROQ_MODEL_CHAIN]
    any_available = any(model["available"] for model in models)
    all_limit = bool(models) and all(model.get("error") == "limit" for model in models)

    return {
        "ok": True,
        "models": models,
        "any_available": any_available,
        "all_limit": all_limit and not any_available,
    }


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


def _groq_chat(
    messages: list[dict[str, str]],
    api_key: str | None,
    temperature: float,
    *,
    max_tokens: int | None = None,
    json_mode: bool = False,
) -> str:
    client = OpenAI(
        base_url=GROQ_BASE_URL,
        api_key=api_key or os.getenv("GROQ_API_KEY", ""),
    )
    last_error: Exception | None = None
    for groq_model in GROQ_MODEL_CHAIN:
        try:
            response = client.chat.completions.create(
                **_groq_completion_kwargs(
                    groq_model,
                    messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    json_mode=json_mode,
                )
            )
            return response.choices[0].message.content or "{}"
        except RateLimitError as exc:
            last_error = exc
        except APIStatusError as exc:
            if exc.status_code in (413, 429):
                last_error = exc
            else:
                raise
    if isinstance(last_error, APIStatusError) and last_error.status_code == 413:
        raise RuntimeError(LLM_REQUEST_TOO_LARGE) from last_error
    raise RuntimeError(LLM_LIMIT_ERROR) from last_error


def create_llm_client(provider: Provider, api_key: str | None = None) -> OpenAI:
    if provider == "ollama":
        return OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")
    if provider == "groq":
        return OpenAI(
            base_url=GROQ_BASE_URL,
            api_key=api_key or os.getenv("GROQ_API_KEY", ""),
        )
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

    if provider == "groq":
        return _groq_chat(
            messages,
            api_key,
            temperature,
            max_tokens=max_tokens or (GROQ_MAX_COMPLETION_TOKENS if json_mode else None),
            json_mode=json_mode,
        )

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
