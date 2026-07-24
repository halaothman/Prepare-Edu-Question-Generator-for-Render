"""Universal document-to-questions pipeline.

Strategy:
1. Extract text from the uploaded file.
2. Split into logical segments (headings / topics), capped at MAX_LOGICAL_SEGMENTS.
3. Generate a fixed total question budget distributed across segments.
4. Merge, dedupe, validate, then cap to TARGET_QUESTIONS_TOTAL with type diversity.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any

from .chunking import build_logical_segments
from .config import (
    AUTO_FALLBACK_TO_GROQ,
    DEFAULT_DEEPSEEK_MODEL,
    DEFAULT_GROQ_MODEL,
    DEFAULT_HF_MODEL,
    DEFAULT_OLLAMA_MODEL,
    LLM_INSUFFICIENT_BALANCE,
    LLM_LIMIT_ERROR,
    LLM_REQUEST_TOO_LARGE,
    LOGICAL_SEGMENT_MAX_CHARS,
    MAX_LOGICAL_SEGMENTS,
    PIPELINE_ALL_SEGMENTS_FAILED,
    TARGET_COMPUTATION_MIN,
    TARGET_ANALYSIS_APPLICATION_MIN,
    TARGET_QUESTIONS_TOTAL,
)
from .generator import Difficulty, Lang, QuestionType, generate_questions
from .question_selection import cap_payload, distribute_question_counts
from .validator import filter_payload

PipelineProgressCallback = Callable[[str, dict[str, Any]], None]


def _resolve_model_id(provider: str, model: str) -> str:
    if model:
        return model
    defaults = {
        "deepseek": DEFAULT_DEEPSEEK_MODEL,
        "groq": DEFAULT_GROQ_MODEL,
        "ollama": DEFAULT_OLLAMA_MODEL,
        "huggingface": DEFAULT_HF_MODEL,
    }
    return defaults.get(provider, DEFAULT_HF_MODEL)


def _emit_progress(
    callback: PipelineProgressCallback | None,
    stage: str,
    **payload: Any,
) -> None:
    if callback is not None:
        callback(stage, payload)


def sample_segments(segments: list[str], max_segments: int) -> list[str]:
    if len(segments) <= max_segments:
        return segments
    if max_segments <= 0:
        return []
    if max_segments == 1:
        return [segments[len(segments) // 2]]

    last_index = len(segments) - 1
    picked: list[str] = []
    seen: set[int] = set()
    for i in range(max_segments):
        index = round(i * last_index / (max_segments - 1))
        if index in seen:
            continue
        seen.add(index)
        picked.append(segments[index])
    return picked


def _question_key(item: dict) -> str:
    question = re.sub(r"\s+", " ", str(item.get("q", "")).strip().lower())
    options = "|".join(
        re.sub(r"\s+", " ", str(option).strip().lower())
        for option in item.get("options", [])
    )
    return f"{question}::{options}"


def dedupe_payload(payload: dict) -> dict:
    deduped = {"mcq": [], "tf": [], "short": []}
    seen_mcq: set[str] = set()

    for item in payload.get("mcq", []):
        key = _question_key(item)
        if not key or key in seen_mcq:
            continue
        seen_mcq.add(key)
        deduped["mcq"].append(item)

    for key in ("tf", "short"):
        seen: set[str] = set()
        for item in payload.get(key, []):
            item_key = re.sub(r"\s+", " ", str(item.get("q", "")).strip().lower())
            if not item_key or item_key in seen:
                continue
            seen.add(item_key)
            deduped[key].append(item)

    return deduped


def merge_payloads(payloads: list[dict]) -> dict:
    merged = {"mcq": [], "tf": [], "short": []}
    for payload in payloads:
        for key in merged:
            merged[key].extend(payload.get(key, []))
    return dedupe_payload(merged)


_PROVIDER_LIMIT_ERRORS = frozenset({LLM_LIMIT_ERROR, LLM_INSUFFICIENT_BALANCE})


def _generate_segment_payload(
    segment: str,
    lang: Lang,
    difficulty: Difficulty,
    types: list[QuestionType],
    num_questions: int | None,
    model: str,
    provider: str,
    api_key: str | None,
    math_focus: bool,
    dl_focus: bool,
) -> dict | None:
    if num_questions == 0:
        return {"mcq": [], "tf": [], "short": []}
    try:
        return generate_questions(
            segment,
            lang,
            difficulty,
            types,
            num_questions,
            model,
            provider,
            api_key,
            math_focus,
            dl_focus,
        )
    except json.JSONDecodeError:
        return None
    except RuntimeError as exc:
        message = str(exc)
        if message == LLM_REQUEST_TOO_LARGE:
            return None
        if message in _PROVIDER_LIMIT_ERRORS:
            raise
        return None


def _generate_segment_with_fallback(
    segment: str,
    lang: Lang,
    difficulty: Difficulty,
    types: list[QuestionType],
    num_questions: int | None,
    model: str,
    provider: str,
    api_key: str | None,
    math_focus: bool,
    dl_focus: bool,
    *,
    fallback_provider: str | None,
    fallback_api_key: str | None,
    active: dict[str, object],
) -> dict | None:
    try:
        return _generate_segment_payload(
            segment,
            lang,
            difficulty,
            types,
            num_questions,
            model,
            str(active["provider"]),
            str(active["api_key"]) if active["api_key"] else None,
            math_focus,
            dl_focus,
        )
    except RuntimeError as exc:
        message = str(exc)
        if message not in _PROVIDER_LIMIT_ERRORS:
            raise
        if message == LLM_INSUFFICIENT_BALANCE and active["provider"] == provider:
            active["primary_balance_exhausted"] = True
        if (
            not fallback_provider
            or not fallback_api_key
            or active["provider"] == fallback_provider
        ):
            raise
        active["provider"] = fallback_provider
        active["api_key"] = fallback_api_key
        active["fallback_used"] = True
        return _generate_segment_payload(
            segment,
            lang,
            difficulty,
            types,
            num_questions,
            model,
            fallback_provider,
            fallback_api_key,
            math_focus,
            dl_focus,
        )


def generate_from_document(
    text: str,
    lang: Lang,
    difficulty: Difficulty,
    types: list[QuestionType],
    num_questions: int | None = None,
    model: str = DEFAULT_HF_MODEL,
    provider: str = "huggingface",
    api_key: str | None = None,
    math_focus: bool = False,
    dl_focus: bool = False,
    *,
    fallback_provider: str | None = None,
    fallback_api_key: str | None = None,
    progress_callback: PipelineProgressCallback | None = None,
    target_questions: int | None = None,
    target_computation_min: int | None = None,
) -> tuple[dict, dict]:
    question_budget = (
        target_questions
        if target_questions is not None
        else (num_questions if num_questions is not None else TARGET_QUESTIONS_TOTAL)
    )
    computation_goal = (
        target_computation_min
        if target_computation_min is not None
        else TARGET_COMPUTATION_MIN
    )

    segments = build_logical_segments(
        text,
        max_segments=MAX_LOGICAL_SEGMENTS,
        max_segment_chars=LOGICAL_SEGMENT_MAX_CHARS,
    )

    if not segments:
        return {"mcq": [], "tf": [], "short": []}, {
            "text_chars": len(text),
            "segments_total": 0,
            "segments_used": 0,
            "segments_skipped": 0,
            "segment_max_chars": LOGICAL_SEGMENT_MAX_CHARS,
            "target_questions": question_budget,
            "segmentation_mode": "logical",
        }

    segments_total = len(segments)
    segments_sampled = False

    _emit_progress(
        progress_callback,
        "chunking",
        text_chars=len(text),
        segments_total=segments_total,
        segments_used=len(segments),
        segments_sampled=segments_sampled,
        segment_max_chars=LOGICAL_SEGMENT_MAX_CHARS,
        target_questions=question_budget,
    )

    payloads: list[dict] = []
    segments_skipped = 0
    active: dict[str, object] = {
        "provider": provider,
        "api_key": api_key,
        "fallback_used": False,
        "primary_balance_exhausted": False,
    }

    per_segment_counts = distribute_question_counts(question_budget, len(segments))
    segment_jobs = list(zip(segments, per_segment_counts))

    use_fallback = (
        AUTO_FALLBACK_TO_GROQ
        and bool(fallback_provider)
        and bool(fallback_api_key)
    )

    total_jobs = len(segment_jobs)

    for job_index, (segment, segment_count) in enumerate(segment_jobs, start=1):
        if segment_count <= 0:
            continue
        active_provider = str(active["provider"])
        model_id = _resolve_model_id(active_provider, model)
        _emit_progress(
            progress_callback,
            "segment_llm_start",
            index=job_index,
            total=total_jobs,
            provider=active_provider,
            model=model_id,
            segment_chars=len(segment),
            segment_questions=segment_count,
        )
        if use_fallback:
            payload = _generate_segment_with_fallback(
                segment,
                lang,
                difficulty,
                types,
                segment_count,
                model,
                provider,
                api_key,
                math_focus,
                dl_focus,
                fallback_provider=fallback_provider,
                fallback_api_key=fallback_api_key,
                active=active,
            )
        else:
            payload = _generate_segment_payload(
                segment,
                lang,
                difficulty,
                types,
                segment_count,
                model,
                str(active["provider"]),
                str(active["api_key"]) if active["api_key"] else None,
                math_focus,
                dl_focus,
            )
        if payload is not None:
            payloads.append(payload)
            _emit_progress(
                progress_callback,
                "segment_llm_done",
                index=job_index,
                total=total_jobs,
                provider=str(active["provider"]),
                model=_resolve_model_id(str(active["provider"]), model),
                mcq_count=len(payload.get("mcq", [])),
            )
        else:
            segments_skipped += 1
            _emit_progress(
                progress_callback,
                "segment_skip",
                index=job_index,
                total=total_jobs,
                provider=str(active["provider"]),
            )

    if not payloads:
        if active.get("primary_balance_exhausted"):
            raise RuntimeError(LLM_INSUFFICIENT_BALANCE)
        raise RuntimeError(PIPELINE_ALL_SEGMENTS_FAILED)

    mcq_before_merge = sum(len(p.get("mcq", [])) for p in payloads)
    _emit_progress(
        progress_callback,
        "merge",
        mcq_raw=mcq_before_merge,
        segment_payloads=len(payloads),
    )
    merged = merge_payloads(payloads)
    mcq_after_dedupe = len(merged.get("mcq", []))
    _emit_progress(
        progress_callback,
        "merge_done",
        mcq_after_dedupe=mcq_after_dedupe,
    )
    _emit_progress(progress_callback, "validate", mcq_before_filter=mcq_after_dedupe)
    provider_used = str(active["provider"])
    meta = {
        "text_chars": len(text),
        "segments_total": segments_total,
        "segments_used": len(segments),
        "segments_skipped": segments_skipped,
        "segment_max_chars": LOGICAL_SEGMENT_MAX_CHARS,
        "segmentation_mode": "logical",
        "target_questions": question_budget,
        "target_computation_min": computation_goal,
        "target_analysis_application_min": TARGET_ANALYSIS_APPLICATION_MIN,
        "provider_used": provider_used,
        "primary_provider": provider,
        "fallback_provider": fallback_provider,
        "fallback_used": bool(active["fallback_used"]),
        "primary_balance_exhausted": bool(active.get("primary_balance_exhausted")),
    }
    active_api_key = active["api_key"]
    filtered = filter_payload(
        merged,
        text,
        lang,
        provider_used,
        str(active_api_key) if active_api_key else None,
        model,
    )
    mcq_before_cap = len(filtered.get("mcq", []))
    _emit_progress(
        progress_callback,
        "cap",
        mcq_before_cap=mcq_before_cap,
        target_questions=question_budget,
    )
    capped = cap_payload(
        filtered,
        max_total=question_budget,
        min_computation=computation_goal,
        source=text,
    )
    mcq_final = len(capped.get("mcq", []))
    meta["mcq_before_cap"] = mcq_before_cap
    meta["mcq_final"] = mcq_final
    _emit_progress(
        progress_callback,
        "done",
        mcq_final=mcq_final,
        provider_used=provider_used,
        segments_skipped=segments_skipped,
        fallback_used=bool(active["fallback_used"]),
    )
    return capped, meta
