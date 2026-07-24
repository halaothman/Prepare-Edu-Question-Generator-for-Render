from __future__ import annotations

from ._shared import Difficulty
from .deepseek import build_deepseek_prompt
from .openai import build_openai_prompt
from .qwen import build_qwen_prompt

PromptFamily = str

PROMPT_PROVIDER_MAP: dict[str, PromptFamily] = {
    "groq": "qwen",
    "deepseek": "deepseek",
    "openai": "openai",
    "huggingface": "qwen",
    "ollama": "qwen",
}

PROMPT_BUILDERS = {
    "qwen": build_qwen_prompt,
    "deepseek": build_deepseek_prompt,
    "openai": build_openai_prompt,
}

SYSTEM_MESSAGES: dict[PromptFamily, dict[str, str]] = {
    "qwen": {
        "ar": (
            "أنت خبير في تصميم الاختبارات الجامعية باللغة العربية. "
            "ولّد من 5 إلى 10 أسئلة MCQ عالية الجودة من المستند فقط. "
            "الجودة أهم من الكمية. JSON صالح فقط."
        ),
        "en": (
            "You design university exams in Arabic. "
            "Generate 5–10 high-quality MCQs from the document chunk only. "
            "Quality over quantity. Valid JSON only."
        ),
    },
    "deepseek": {
        "en": (
            "University final exam MCQs in Arabic: mix analysis/application (why, what-if) "
            "with multi-step computation (understand then calculate). "
            "No mechanical subtraction-only items; confident short solutions. JSON only."
        ),
        "ar": (
            "أنت عضو هيئة تدريس تعد امتحاناً نهائياً. "
            "مزيج: تحليل/تطبيق (لماذا، ماذا لو) + حساب multi-step (فهم ثم حساب) — "
            "لا طرح ميكانيكي ولا solution متردد. JSON فقط."
        ),
    },
    "openai": {
        "en": (
            "You are an expert university professor (OpenAI / ChatGPT prompt). "
            "Generate Hard Arabic MCQs from the document chunk only. Valid JSON only."
        ),
        "ar": (
            "أنت أستاذ جامعي خبير (برومبت ChatGPT). "
            "ولّد أسئلة MCQ عربية صعبة من جزء المستند فقط. JSON صالح فقط."
        ),
    },
}


def resolve_prompt_family(provider: str) -> PromptFamily:
    return PROMPT_PROVIDER_MAP.get(provider, "qwen")


def build_prompt(
    context: str,
    lang: str,
    difficulty: Difficulty,
    types: list[str],
    num_questions: int | None,
    provider: str,
    *,
    math_focus: bool = False,
    dl_focus: bool = False,
) -> str:
    del lang, types, math_focus, dl_focus
    family = resolve_prompt_family(provider)
    builder = PROMPT_BUILDERS[family]
    return builder(context, difficulty, num_questions)


def build_system_message(lang: str, provider: str) -> str:
    family = resolve_prompt_family(provider)
    messages = SYSTEM_MESSAGES[family]
    return messages["ar" if lang == "ar" else "en"]
