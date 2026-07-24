from __future__ import annotations

import json
import re
from typing import Literal

from langdetect import detect

from .config import DEFAULT_HF_MODEL, JSON_MODE_PROVIDERS
from .llm_client import chat_complete
from .prompts import build_prompt, build_system_message

Lang = Literal["ar", "en"]
Difficulty = Literal["Easy", "Medium", "Hard"]
QuestionType = Literal["mcq", "tf", "short"]

LANGUAGE_RULES_AR = """
- استخدم العربية الفصحى مع مصطلحات تقنية إنجليزية شائعة فقط (مثل CNN, Attention, FLOPs)
- ممنوع تماماً: الأحرف الصينية أو اليابانية أو الكورية أو أي رموز غريبة
- مسموح فقط: العربية، الإنجليزية، الأرقام، وعلامات رياضية شائعة (+ - × ÷ = ^ / ( ) [ ] %)
"""

LANGUAGE_RULES_EN = """
- Use clear English with standard technical terms only
- Never use Chinese, Japanese, Korean, or unrelated scripts/symbols
- Allowed only: English, Arabic if needed for terms, digits, and common math symbols (+ - × ÷ = ^ / ( ) [ ] %)
"""

FORBIDDEN_CHARS = re.compile(
    "["
    "\u4e00-\u9fff"  # CJK Unified Ideographs
    "\u3400-\u4dbf"  # CJK Extension A
    "\u3040-\u30ff"  # Hiragana + Katakana
    "\u31f0-\u31ff"  # Katakana extensions
    "\uac00-\ud7af"  # Hangul syllables
    "\u1100-\u11ff"  # Hangul jamo
    "\u0400-\u04ff"  # Cyrillic
    "\u0900-\u097f"  # Devanagari
    "\u0980-\u09ff"  # Bengali
    "\u0e00-\u0e7f"  # Thai
    "]+"
)


def detect_lang(text: str) -> Lang:
    try:
        return "ar" if detect(text) == "ar" else "en"
    except Exception:
        return "en"


def sanitize_text(text: str) -> str:
    if not text:
        return text
    cleaned = FORBIDDEN_CHARS.sub("", str(text))
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    return cleaned.strip()


def sanitize_payload(payload: dict) -> dict:
    for key in ("mcq", "tf", "short"):
        for item in payload.get(key, []):
            item["q"] = sanitize_text(item.get("q", ""))
            item["solution"] = sanitize_text(item.get("solution") or item.get("explanation", ""))
            item["explanation"] = item["solution"]
            if item.get("question_kind"):
                item["question_kind"] = sanitize_text(str(item["question_kind"]))
            if key == "mcq":
                item["options"] = [sanitize_text(option) for option in item.get("options", [])]
                if not isinstance(item.get("answer"), bool):
                    item["answer"] = sanitize_text(str(item.get("answer", "")))
            elif key == "tf":
                if not isinstance(item.get("answer"), bool):
                    item["answer"] = sanitize_text(str(item.get("answer", "")))
            else:
                item["answer"] = sanitize_text(str(item.get("answer", "")))
    return payload


def _mcq_item_from_raw(item: dict, default_difficulty: str) -> dict:
    options = _resolve_mcq_options(item.get("options", []))
    answer = _resolve_mcq_answer(options, item.get("correct_answer", item.get("answer", "")))
    question_kind = item.get("type", item.get("question_kind", ""))
    if question_kind in {"mcq", "true_false", "tf", "short"}:
        question_kind = item.get("question_kind", "")
    return {
        "q": item.get("question") or item.get("q", ""),
        "options": options,
        "answer": answer,
        "solution": item.get("solution") or item.get("explanation", ""),
        "question_kind": question_kind,
        "difficulty": item.get("difficulty", default_difficulty),
    }


def _strip_model_artifacts(text: str) -> str:
    cleaned = str(text or "")
    think_open = "<" + "think" + ">"
    think_close = "</" + "think" + ">"
    cleaned = re.sub(
        rf"{re.escape(think_open)}.*?{re.escape(think_close)}",
        "",
        cleaned,
        flags=re.DOTALL | re.IGNORECASE,
    )
    cleaned = re.sub(
        r"<think>.*?</think>",
        "",
        cleaned,
        flags=re.DOTALL | re.IGNORECASE,
    )
    cleaned = cleaned.replace("\u201c", '"').replace("\u201d", '"')
    cleaned = cleaned.replace("\u2018", "'").replace("\u2019", "'")
    return cleaned.strip()


def _extract_json_object(text: str) -> str:
    text = _strip_model_artifacts(text)
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return text[start : end + 1]
    return text


def _fix_invalid_json_escapes(text: str) -> str:
    return re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", text)


def _remove_trailing_commas(text: str) -> str:
    return re.sub(r",(\s*[}\]])", r"\1", text)


def safe_json(raw: str) -> dict:
    text = _extract_json_object(raw)
    candidates = [
        text,
        _fix_invalid_json_escapes(text),
        _remove_trailing_commas(text),
        _remove_trailing_commas(_fix_invalid_json_escapes(text)),
    ]
    last_error: json.JSONDecodeError | None = None
    seen: set[str] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    raise json.JSONDecodeError("No JSON object found", raw, 0)


def _resolve_mcq_options(options: object) -> list[str]:
    if isinstance(options, dict):
        return [str(options.get(key, "")) for key in ("A", "B", "C", "D")]
    if isinstance(options, list):
        return [str(option) for option in options[:4]]
    return []


def _resolve_mcq_answer(options: list[str], answer: object) -> str:
    if isinstance(answer, str) and len(answer) == 1 and answer.upper() in "ABCD":
        index = "ABCD".index(answer.upper())
        if index < len(options):
            return options[index]
    return str(answer) if answer is not None else ""


def normalize_payload(raw: dict, default_difficulty: str = "Hard") -> dict:
    normalized: dict[str, list[dict]] = {"mcq": [], "tf": [], "short": []}

    if isinstance(raw.get("mcq"), list):
        for item in raw["mcq"]:
            normalized["mcq"].append(_mcq_item_from_raw(item, default_difficulty))
        return normalized

    for item in raw.get("questions", []):
        question_type = str(item.get("type", "")).lower().replace("-", "_").replace(" ", "_")
        if question_type in {"analytical", "computational"}:
            normalized["mcq"].append(_mcq_item_from_raw(item, default_difficulty))
            continue

        question = item.get("question") or item.get("q", "")
        solution = item.get("solution") or item.get("explanation", "")
        item_difficulty = item.get("difficulty", default_difficulty)
        question_kind = item.get("question_kind", "")

        if question_type == "mcq":
            normalized["mcq"].append(_mcq_item_from_raw(item, default_difficulty))
        elif question_type in {"tf", "true_false", "true/false"}:
            normalized["tf"].append(
                {
                    "q": question,
                    "answer": item.get("correct_answer", item.get("answer")),
                    "solution": solution,
                    "question_kind": question_kind,
                    "difficulty": item_difficulty,
                }
            )
        elif question_type == "short":
            normalized["short"].append(
                {
                    "q": question,
                    "answer": item.get("correct_answer", item.get("answer", "")),
                    "solution": solution,
                    "question_kind": question_kind,
                    "difficulty": item_difficulty,
                }
            )

    return normalized


def generate_questions(
    context: str,
    lang: Lang,
    difficulty: Difficulty,
    types: list[QuestionType],
    num_questions: int | None = None,
    model: str = DEFAULT_HF_MODEL,
    provider: str = "huggingface",
    api_key: str | None = None,
    math_focus: bool = False,
    dl_focus: bool = False,
) -> dict:
    prompt = build_prompt(
        context,
        lang,
        difficulty,
        types,
        num_questions,
        provider,
        math_focus=math_focus,
        dl_focus=dl_focus,
    )
    system = build_system_message(lang, provider)
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]
    last_error: json.JSONDecodeError | None = None
    for attempt, temp in enumerate((0.25, 0.1)):
        try:
            content = chat_complete(
                provider,
                model,
                messages,
                api_key=api_key,
                temperature=temp,
                json_mode=provider in JSON_MODE_PROVIDERS,
            )
            return sanitize_payload(
                normalize_payload(safe_json(content), default_difficulty=difficulty)
            )
        except json.JSONDecodeError as exc:
            last_error = exc
            if attempt == 0:
                continue
            raise
    if last_error is not None:
        raise last_error
    raise json.JSONDecodeError("No JSON object found", "", 0)
