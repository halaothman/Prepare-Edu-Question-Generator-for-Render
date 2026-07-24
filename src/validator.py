from __future__ import annotations

import json
import re

from .config import JSON_MODE_PROVIDERS
from .generator import Lang, safe_json
from .llm_client import chat_complete
from .numeric_recall import is_numeric_recall_from_source

VALIDATION_CONTEXT_LIMIT = 14_000

FORBIDDEN_VAGUE_WORDS = re.compile(
    r"\b(best|expected|usually|generally|might|could|would)\b",
    re.IGNORECASE,
)

TRIVIAL_NAME_PATTERNS = re.compile(
    r"^(what is|what does|define|ما هو|ما هي|عرّف|عرّف)\s+[\w\.]+(\s+do|\s+does)?\s*\?",
    re.IGNORECASE,
)

FORBIDDEN_TRIVIAL_TERMS = re.compile(
    r"(ما هو|ما هي|what is|what does)\s+"
    r"(tensorflow|sgd|adam|dropout|labelencoder|tf-?idf|batchnormalization|batch normalization|"
    r"standardscaler|minmaxscaler|onehotencoder|keras|numpy|pandas)\s*\?",
    re.IGNORECASE,
)

ONE_LINE_RECALL_PATTERNS = re.compile(
    r"(what is the (value|name|filename|optimizer|learning rate|batch size|epoch)|"
    r"ما (هو|هي|قيمة)|"
    r"how many elements|"
    r"كم عدد العناصر|"
    r"initial value of|"
    r"القيمة الابتدائية)",
    re.IGNORECASE,
)

FORBIDDEN_ASSUMPTION_PATTERNS = re.compile(
    r"(إذا افترضنا|لنفترض|إذا كانت البيانات|إذا كان عدد العينات|إذا كان الإدخال|"
    r"إذا احتوى النموذج|إذا كان حجم البيانات|إذا كانت طبقة الإدخال|إذا كانت الصورة|"
    r"if we assume|let us assume|assuming (the|that)|"
    r"if the (data|input|model|batch|image|dataset))",
    re.IGNORECASE,
)

MECHANICAL_SUBTRACTION = re.compile(
    r"(كم\s+(?:كلمة|word).*(?:تجاه|ignore|ignored|تُتجاه|يتم\s+تجاه)|"
    r"max_words?\s*=.*(?:كم|how\s+many))",
    re.IGNORECASE,
)

TRIVIAL_SENTIMENT_LABEL = re.compile(
    r"(label\s*=\s*[01]|y\s*=\s*[01]|التصنيف\s*[01]).*(إيجاب|سلب|positive|negative|sentiment)",
    re.IGNORECASE,
)

UNCERTAIN_SOLUTION = re.compile(
    r"(لكن\s+عملي[اأ]|غير\s+واثق|قد\s+يحدث\s+خطأ|ربما\s|من\s+المحتمل|but\s+in\s+practice)",
    re.IGNORECASE,
)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip()).lower()


def _contains_forbidden_vague_words(text: str, source: str) -> bool:
    source_lower = source.lower()
    for match in FORBIDDEN_VAGUE_WORDS.finditer(text):
        if match.group(0).lower() not in source_lower:
            return True
    return False


def _is_trivial_name_question(question: str) -> bool:
    return bool(TRIVIAL_NAME_PATTERNS.match(question.strip()))


def _passes_rule_checks(item: dict, source: str = "") -> bool:
    question = str(item.get("q", "")).strip()
    options = [str(option).strip() for option in item.get("options", []) if str(option).strip()]
    answer = str(item.get("answer", "")).strip()

    if len(question) < 10:
        return False
    if len(options) != 4:
        return False
    if len({_normalize(option) for option in options}) < 4:
        return False
    if not answer:
        return False

    normalized_options = [_normalize(option) for option in options]
    if _normalize(answer) not in normalized_options:
        return False

    if _is_trivial_name_question(question):
        return False

    if FORBIDDEN_TRIVIAL_TERMS.search(question):
        return False

    if ONE_LINE_RECALL_PATTERNS.search(question):
        return False

    if FORBIDDEN_ASSUMPTION_PATTERNS.search(question):
        return False

    if MECHANICAL_SUBTRACTION.search(question):
        return False

    if TRIVIAL_SENTIMENT_LABEL.search(question):
        return False

    solution = str(item.get("solution") or item.get("explanation") or "")
    if UNCERTAIN_SOLUTION.search(solution):
        return False
    if len(solution) > 500:
        return False

    if source and is_numeric_recall_from_source(item, source):
        return False

    combined_text = " ".join([question, answer, *options])
    if _contains_forbidden_vague_words(combined_text, source):
        return False

    forbidden_option_fragments = (
        "reducing images",
        "deep storage",
        "improving storage",
        "تقليل الصور",
        "تخزين عميق",
        "تحسين التخزين",
    )
    lowered_options = [option.lower() for option in options]
    if any(fragment in option for fragment in forbidden_option_fragments for option in lowered_options):
        return False

    return True


def _build_validation_prompt(source: str, questions: list[dict], lang: Lang) -> str:
    compact_questions = [
        {
            "id": index + 1,
            "type": item.get("question_kind", item.get("type", "")),
            "q": item.get("q", ""),
            "options": item.get("options", []),
            "answer": item.get("answer", ""),
            "solution": item.get("solution", item.get("explanation", "")),
        }
        for index, item in enumerate(questions)
    ]

    language_note = (
        "Respond with JSON only. Reasons may be in Arabic."
        if lang == "ar"
        else "Respond with JSON only. Reasons in English."
    )

    return f"""You are a strict university professor validating Hard Arabic MCQs.

SOURCE DOCUMENT (only allowed evidence):
{source[:VALIDATION_CONTEXT_LIMIT]}

QUESTIONS TO VALIDATE:
{json.dumps(compact_questions, ensure_ascii=False)}

Reject a question if ANY condition holds:

SOURCE GROUNDING:
1. Not fully answerable using ONLY the source document.
2. Correct answer cannot be verified from the source alone.
3. Relies on hidden assumptions, missing numbers, or unstated facts.
4. Uses external knowledge not in the source.
5. Invents equations, numbers, datasets, tensor dimensions, filenames, architectures, hyperparameters, optimizers, or assumptions.

QUESTION QUALITY:
6. Simple recall/memorization instead of reasoning, analysis, or calculation.
7. Answerable by reading one line of code or one line from the document.
8. Asks directly about variable names, filenames, function names, imports, library names, initial values, element counts, learning rates, epochs, or batch sizes without genuine reasoning.
9. Not difficult enough for a university exam.

COMPUTATIONAL / ANALYTICAL:
10. Computational: answer cannot be calculated from values in the question or source.
11. Analytical: simple definition, vague, or subjective question.

FORBIDDEN EXAMPLES:
12. "What is TensorFlow/SGD/Adam/Dropout/LabelEncoder/BatchNormalization?" or Arabic equivalents.

OPTIONS / EXPLANATION:
13. Illogical, nonsense, or obviously wrong distractors.
14. More than one option could reasonably be correct.
15. Solution merely repeats the correct option instead of explaining why.

When unsure, reject.

{language_note}

Return JSON only:
{{
  "keep": [1, 2],
  "rejected": [{{"id": 3, "reason": "..."}}]
}}

"keep" must list ONLY the ids of valid questions."""


def _llm_filter_ids(
    source: str,
    questions: list[dict],
    lang: Lang,
    provider: str,
    api_key: str | None,
    model: str,
) -> set[int]:
    if not questions:
        return set()

    prompt = _build_validation_prompt(source, questions, lang)
    system = (
        "You validate MCQs against a source document. Be strict. JSON only."
        if lang == "en"
        else "تحقق من أسئلة MCQ مقابل المستند. كن صارماً. JSON فقط."
    )
    content = chat_complete(
        provider,
        model,
        [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        api_key=api_key,
        temperature=0.1,
        max_tokens=2048,
        json_mode=provider in JSON_MODE_PROVIDERS,
    )
    parsed = safe_json(content)
    keep_ids: set[int] = set()
    for item in parsed.get("keep", []):
        try:
            keep_ids.add(int(item))
        except (TypeError, ValueError):
            continue
    return keep_ids


def filter_mcq_payload(
    payload: dict,
    source: str,
    lang: Lang,
    provider: str,
    api_key: str | None,
    model: str = "",
) -> dict:
    mcq_items = payload.get("mcq", [])
    if not mcq_items:
        return payload

    rule_passed = [item for item in mcq_items if _passes_rule_checks(item, source)]
    if not rule_passed:
        payload["mcq"] = []
        return payload

    try:
        keep_ids = _llm_filter_ids(source, rule_passed, lang, provider, api_key, model)
    except Exception:
        payload["mcq"] = rule_passed
        return payload

    payload["mcq"] = [
        item for index, item in enumerate(rule_passed, start=1) if index in keep_ids
    ]
    return payload


def filter_payload(
    payload: dict,
    source: str,
    lang: Lang,
    provider: str,
    api_key: str | None,
    model: str = "",
) -> dict:
    return filter_mcq_payload(payload, source, lang, provider, api_key, model)
