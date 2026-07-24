from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import JSON_MODE_PROVIDERS
from src.llm_client import chat_complete
from src.prompts import build_prompt, build_system_message

secrets = Path(".streamlit/secrets.toml").read_text(encoding="utf-8")
groq_key = re.search(r'GROQ_API_KEY = "([^"]*)"', secrets).group(1)
sample = ("CNN Conv2D batch_size=64. " * 20)[:500]
prompt = build_prompt(sample, "ar", "Hard", ["mcq"], None, "groq")
system = build_system_message("ar", "groq")
messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
total = len(system) + len(prompt)
print("total_chars", total)

try:
    content = chat_complete(
        "groq",
        "",
        messages,
        api_key=groq_key,
        temperature=0.25,
        json_mode="groq" in JSON_MODE_PROVIDERS,
    )
    print("OK", len(content))
except Exception as exc:
    print("ERR", exc)
