import os
from typing import Optional

IS_HF_SPACE = bool(os.getenv("SPACE_ID"))


def _detect_default_provider() -> str:
    if os.getenv("LLM_PROVIDER"):
        return os.getenv("LLM_PROVIDER", "ollama")
    if IS_HF_SPACE:
        if os.getenv("GROQ_API_KEY"):
            return "groq"
        if os.getenv("HF_TOKEN"):
            return "huggingface"
        return "groq"
    if os.getenv("GROQ_API_KEY"):
        return "groq"
    return "ollama"


DEFAULT_PROVIDER = _detect_default_provider()

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", f"{OLLAMA_HOST}/v1")
DEFAULT_OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")

OLLAMA_QWEN_MODELS = [
    "qwen2.5:7b",
    "qwen2.5:14b",
    "qwen3",
]

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
# Qwen 2.5 72B is not hosted on Groq; qwen/qwen3.6-27b is the best Qwen available there.
DEFAULT_GROQ_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3.6-27b")
# Primary first; on 429/daily limit silently try the next model.
GROQ_MODEL_CHAIN = [
    "qwen/qwen3.6-27b",
    "llama-3.1-8b-instant",
]
GROQ_MODELS = [
    ("Qwen 3.6 27B ⭐ (أفضل Qwen على Groq)", "qwen/qwen3.6-27b"),
    ("Llama 3.1 8B (احتياطي)", "llama-3.1-8b-instant"),
]

DEFAULT_HF_MODEL = os.getenv("HF_MODEL", "Qwen/Qwen2.5-7B-Instruct")
HF_QWEN_MODELS = [
    ("Qwen2.5-7B (الوحيد المدعوم)", "Qwen/Qwen2.5-7B-Instruct"),
]

DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
OPENAI_MODELS = ["gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-4.1-mini"]

DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
# deepseek-reasoner = DeepSeek R1 | deepseek-chat = V3 (أسرع، بدون chain-of-thought)
DEFAULT_DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
DEEPSEEK_REASONER_MAX_TOKENS = int(os.getenv("DEEPSEEK_REASONER_MAX_TOKENS", "8192"))


def deepseek_model_display_name(model_id: Optional[str] = None) -> str:
    model = model_id or DEFAULT_DEEPSEEK_MODEL
    if "reasoner" in model.lower():
        return "DeepSeek R1"
    if "chat" in model.lower():
        return "DeepSeek Chat (V3)"
    return model

PRIMARY_PROVIDER = os.getenv("PRIMARY_PROVIDER", "deepseek")
FALLBACK_PROVIDER = os.getenv("FALLBACK_PROVIDER", "groq")

# مبدئياً: لا تبديل تلقائي إلى Groq/Qwen/Llama عند نفاد DeepSeek (المفاتيح تبقى في secrets).
AUTO_FALLBACK_TO_GROQ = os.getenv("AUTO_FALLBACK_TO_GROQ", "false").lower() in {
    "1",
    "true",
    "yes",
}
# إظهار Qwen (Groq) في الواجهة — false = DeepSeek فقط
SHOW_ALTERNATE_LLM_PROVIDERS = os.getenv(
    "SHOW_ALTERNATE_LLM_PROVIDERS", "false"
).lower() in {"1", "true", "yes"}

# Every provider uses the same chunked pipeline in src/pipeline.py — do not bypass it.
CHUNKED_PIPELINE_PROVIDERS = ("groq", "openai", "deepseek", "huggingface", "ollama")
JSON_MODE_PROVIDERS = frozenset({"groq", "openai", "deepseek"})
LLM_MAX_COMPLETION_TOKENS = int(os.getenv("LLM_MAX_COMPLETION_TOKENS", "8192"))

CHUNK_SIZE = 650
CHUNK_OVERLAP = 100

# Universal document pipeline (all providers/models):
# 1) extract text  2) chunk to ~2500-3500 chars  3) prompt+chunk per request
# 4) merge questions  5) dedupe  6) validate against full source
SEGMENT_MIN_CHARS = 2500
SEGMENT_MAX_CHARS_LIMIT = 3500
SEGMENT_MAX_CHARS = max(
    SEGMENT_MIN_CHARS,
    min(int(os.getenv("SEGMENT_MAX_CHARS", "3000")), SEGMENT_MAX_CHARS_LIMIT),
)
MAX_SEGMENTS_PER_RUN = int(os.getenv("MAX_SEGMENTS_PER_RUN", "20"))

# Exam pipeline defaults (large documents: few logical sections, fixed question budget)
TARGET_QUESTIONS_TOTAL = int(os.getenv("TARGET_QUESTIONS_TOTAL", "20"))
TARGET_COMPUTATION_MIN = int(os.getenv("TARGET_COMPUTATION_MIN", "6"))
TARGET_ANALYSIS_APPLICATION_MIN = int(os.getenv("TARGET_ANALYSIS_APPLICATION_MIN", "8"))
MAX_LOGICAL_SEGMENTS = int(os.getenv("MAX_LOGICAL_SEGMENTS", "10"))
LOGICAL_SEGMENT_MAX_CHARS = int(os.getenv("LOGICAL_SEGMENT_MAX_CHARS", "14000"))

LLM_REQUEST_TOO_LARGE = "LLM_REQUEST_TOO_LARGE"
LLM_LIMIT_ERROR = "LLM_LIMIT_ERROR"
LLM_INSUFFICIENT_BALANCE = "LLM_INSUFFICIENT_BALANCE"
PIPELINE_ALL_SEGMENTS_FAILED = "PIPELINE_ALL_SEGMENTS_FAILED"

# Provider-specific aliases kept for existing call sites
MAX_CHUNK_GROUPS = MAX_SEGMENTS_PER_RUN
GROQ_MAX_SEGMENT_CHARS = SEGMENT_MAX_CHARS
GROQ_MAX_SEGMENTS = MAX_SEGMENTS_PER_RUN
GROQ_LIMIT_ERROR = LLM_LIMIT_ERROR
GROQ_REQUEST_TOO_LARGE = LLM_REQUEST_TOO_LARGE
GROQ_MAX_COMPLETION_TOKENS = int(os.getenv("GROQ_MAX_COMPLETION_TOKENS", "4096"))
