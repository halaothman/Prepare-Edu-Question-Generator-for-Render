from __future__ import annotations

import re

from .config import TARGET_ANALYSIS_APPLICATION_MIN, TARGET_COMPUTATION_MIN
from .numeric_recall import is_numeric_recall_from_source

_COMPUTATION_TYPES = frozenset(
    {"computation", "computational", "حساب", "حسابي", "calculation", "numeric"}
)

_ANALYSIS_APPLICATION_TYPES = frozenset(
    {"analysis", "application", "تحليل", "تطبيق", "analytical", "applied"}
)

_MECHANICAL_COMPUTE = re.compile(
    r"("
    r"كم\s+(?:كلمة|word).*(?:تجاه|ignore|ignored|تُتجاه|يتم\s+تجاه)"
    r"|(?:تجاهل|ignored).*(?:طرح|−|-|\−)"
    r"|^\s*[\d،,\.]+\s*[-−]\s*[\d،,\.]+\s*$"
    r"|vocab(?:ulary)?\s*[-−]\s*max"
    r"|max_words?\s*=.*كم"
    r")",
    re.IGNORECASE,
)

_UNCERTAIN_SOLUTION = re.compile(
    r"(لكن\s+عملي[اأ]|غير\s+واثق|قد\s+يحدث\s+خطأ|ربما\s+ي|من\s+المحتمل|"
    r"but\s+in\s+practice|might\s+be\s+wrong|uncertain)",
    re.IGNORECASE,
)

_TRIVIAL_LABEL = re.compile(
    r"(label\s*=\s*[01]|التصنيف\s*[01]).*(إيجاب|سلب|positive|negative)",
    re.IGNORECASE,
)


def distribute_question_counts(num_questions: int, segment_count: int) -> list[int]:
    if segment_count <= 0:
        return []
    if num_questions <= 0:
        return [0] * segment_count
    if num_questions <= segment_count:
        return [1 if index < num_questions else 0 for index in range(segment_count)]
    base = num_questions // segment_count
    remainder = num_questions % segment_count
    return [base + (1 if index < remainder else 0) for index in range(segment_count)]


def _question_type(item: dict) -> str:
    raw = str(item.get("question_kind") or item.get("type") or "").strip().lower()
    return raw.replace("-", "_").replace(" ", "_")


def _is_computation(item: dict) -> bool:
    kind = _question_type(item)
    if kind in _COMPUTATION_TYPES:
        return True
    text = str(item.get("q", ""))
    return bool(re.search(r"[=\+\-\×÷\^]|احسب|calculate|compute|\d+\s*[\*/]", text, re.I))


def _is_analysis_or_application(item: dict) -> bool:
    kind = _question_type(item)
    if kind in _ANALYSIS_APPLICATION_TYPES:
        return True
    text = str(item.get("q", ""))
    return bool(
        re.search(
            r"لماذا|ماذا\s+(?:يحدث|لو|إذا)|أثر|compare|why|what\s+if|overfit|dropout",
            text,
            re.I,
        )
    )


def _concept_key(item: dict) -> str:
    q = re.sub(r"\s+", " ", str(item.get("q", "")).strip().lower())
    if re.search(r"embedding|تضمين", q) and re.search(
        r"parameter|معامل|params", q
    ):
        return "concept:embedding_params"
    if re.search(r"max_words?|max_word|أقصى\s+كلم", q):
        return "concept:max_words"
    if re.search(r"padding|حشو|pad", q):
        return "concept:padding"
    if re.search(r"lstm|gru|rnn", q):
        if re.search(r"parameter|معامل", q):
            return "concept:rnn_params"
    return f"q:{q[:100]}"


def _is_weak_item(item: dict, source: str = "") -> bool:
    q = str(item.get("q", ""))
    sol = str(item.get("solution") or "")
    if _MECHANICAL_COMPUTE.search(q):
        return True
    if _TRIVIAL_LABEL.search(q):
        return True
    if _UNCERTAIN_SOLUTION.search(sol):
        return True
    if len(sol) > 450:
        return True
    if source and is_numeric_recall_from_source(item, source):
        return True
    return False


def _difficulty_rank(item: dict) -> int:
    diff = str(item.get("difficulty", "")).strip().lower()
    if diff == "hard":
        return 2
    if diff in {"medium", "med"}:
        return 1
    return 0


def _quality_score(item: dict, source: str = "") -> float:
    if _is_weak_item(item, source):
        return -50.0

    score = float(_difficulty_rank(item) * 3)

    if _is_analysis_or_application(item):
        score += 5.0
    elif _is_computation(item):
        score += 2.0
        sol = str(item.get("solution") or "")
        if len(re.findall(r"[\.؛;]\s|\n|ثم|then|→", sol)) >= 1:
            score += 2.0
    elif _question_type(item) == "understanding":
        score += 0.5

    question = str(item.get("q", ""))
    score += min(len(question) / 120.0, 2.0)

    options = item.get("options") or []
    if isinstance(options, list) and len(options) >= 4:
        score += 0.5

    solution = str(item.get("solution") or "")
    if 40 <= len(solution) <= 280:
        score += 1.0
    return score


def cap_and_diversify_mcq(
    items: list[dict],
    *,
    max_total: int,
    min_computation: int,
    min_analysis_application: int,
    source: str = "",
) -> list[dict]:
    if max_total <= 0 or not items:
        return []

    ranked = sorted(items, key=lambda item: _quality_score(item, source), reverse=True)
    selected: list[dict] = []
    seen_questions: set[str] = set()
    seen_concepts: set[str] = set()

    def try_add(item: dict) -> bool:
        if len(selected) >= max_total:
            return False
        if _is_weak_item(item, source):
            return False
        qkey = re.sub(r"\s+", " ", str(item.get("q", "")).strip().lower())
        if not qkey or qkey in seen_questions:
            return False
        ckey = _concept_key(item)
        if ckey.startswith("concept:") and ckey in seen_concepts:
            return False
        seen_questions.add(qkey)
        if ckey.startswith("concept:"):
            seen_concepts.add(ckey)
        selected.append(item)
        return True

    for item in ranked:
        if sum(1 for x in selected if _is_analysis_or_application(x)) >= min_analysis_application:
            break
        if _is_analysis_or_application(item):
            try_add(item)

    for item in ranked:
        if sum(1 for x in selected if _is_computation(x)) >= min_computation:
            break
        if _is_computation(item):
            try_add(item)

    for item in ranked:
        try_add(item)

    return selected[:max_total]


def cap_payload(
    payload: dict,
    *,
    max_total: int,
    min_computation: int | None = None,
    min_analysis_application: int | None = None,
    source: str = "",
) -> dict:
    capped = {
        "mcq": [],
        "tf": list(payload.get("tf", [])),
        "short": list(payload.get("short", [])),
    }
    capped["mcq"] = cap_and_diversify_mcq(
        list(payload.get("mcq", [])),
        max_total=max_total,
        min_computation=min_computation if min_computation is not None else TARGET_COMPUTATION_MIN,
        min_analysis_application=(
            min_analysis_application
            if min_analysis_application is not None
            else TARGET_ANALYSIS_APPLICATION_MIN
        ),
        source=source,
    )
    return capped
