from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.generator import generate_questions
from src.prompts import build_prompt, build_system_message
from src.llm_client import chat_complete

secrets = Path(".streamlit/secrets.toml").read_text(encoding="utf-8")
groq_key = re.search(r'GROQ_API_KEY = "([^"]*)"', secrets).group(1)
sample = ("CNN Conv2D batch_size=64 epochs=10 lr=0.001. " * 30)[:2000]
prompt = build_prompt(sample, "ar", "Hard", ["mcq"], None, "groq")
system = build_system_message("ar", "groq")
messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
print("prompt_len", len(prompt))

for max_tokens in (8192, 4096, 2048, 1024):
    try:
        content = chat_complete(
            "groq",
            "",
            messages,
            api_key=groq_key,
            max_tokens=max_tokens,
            json_mode=True,
            temperature=0.2,
        )
        print(max_tokens, "OK", len(content))
    except Exception as exc:
        print(max_tokens, type(exc).__name__, exc)
