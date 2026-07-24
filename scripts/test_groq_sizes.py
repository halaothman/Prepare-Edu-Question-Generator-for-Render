from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.generator import generate_questions

secrets = Path(".streamlit/secrets.toml").read_text(encoding="utf-8")
groq_match = re.search(r'GROQ_API_KEY = "([^"]*)"', secrets)
groq_key = groq_match.group(1) if groq_match else ""

for size in (500, 1000, 2000, 3000):
    sample = ("CNN Conv2D batch_size=64 epochs=10. " * 20)[:size]
    try:
        payload = generate_questions(
            sample,
            "ar",
            "Hard",
            ["mcq"],
            provider="groq",
            api_key=groq_key,
            model="",
        )
        print(size, "OK", len(payload.get("mcq", [])))
    except Exception as exc:
        print(size, type(exc).__name__, exc)
