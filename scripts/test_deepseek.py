from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openai import OpenAI

from src.generator import safe_json
from src.prompts import build_prompt, build_system_message

secrets = Path(".streamlit/secrets.toml").read_text(encoding="utf-8")
match = re.search(r'DEEPSEEK_API_KEY = "([^"]*)"', secrets)
api_key = match.group(1) if match else ""
print("key_len", len(api_key))

sample = (
    "Convolutional Neural Networks use Conv2D layers. "
    "If input shape is (32, 32, 3) and batch_size=64, "
    "the model has 1,234,567 parameters after training for 10 epochs "
    "with learning rate 0.001."
)

for base in ("https://api.deepseek.com", "https://api.deepseek.com/v1"):
    print("\n=== base:", base, "===")
    try:
        client = OpenAI(base_url=base, api_key=api_key)
        prompt = build_prompt(sample, "ar", "Hard", ["mcq"], None, "deepseek")
        system = build_system_message("ar", "deepseek")
        r = client.chat.completions.create(
            model="deepseek-reasoner",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            max_tokens=8192,
            temperature=0.2,
        )
        content = r.choices[0].message.content or ""
        print("raw_len", len(content))
        print("raw_head", content[:300])
        parsed = safe_json(content)
        print("mcq_count", len(parsed.get("mcq", [])))
    except Exception as exc:
        print("ERR", type(exc).__name__, exc)
