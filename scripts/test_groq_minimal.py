from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.llm_client import chat_complete, groq_quota_status

secrets = Path(".streamlit/secrets.toml").read_text(encoding="utf-8")
groq_key = re.search(r'GROQ_API_KEY = "([^"]*)"', secrets).group(1)

print("quota", groq_quota_status(groq_key))

try:
    c = chat_complete(
        "groq",
        "",
        [{"role": "user", "content": "Say OK"}],
        api_key=groq_key,
        max_tokens=10,
        json_mode=False,
    )
    print("minimal OK", c[:50])
except Exception as exc:
    print("minimal ERR", type(exc).__name__, exc)
