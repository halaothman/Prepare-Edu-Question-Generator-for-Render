from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.generator import generate_questions, safe_json
from src.pipeline import generate_from_document

secrets = Path(".streamlit/secrets.toml").read_text(encoding="utf-8")
groq_match = re.search(r'GROQ_API_KEY = "([^"]*)"', secrets)
groq_key = groq_match.group(1) if groq_match else ""

sample = (
    "Convolutional Neural Networks use Conv2D layers. "
    "If input shape is (32, 32, 3) and batch_size=64, "
    "the model has 1,234,567 parameters after training for 10 epochs "
    "with learning rate 0.001. " * 80
)

print("Testing groq directly...")
try:
    payload = generate_questions(
        sample[:3000],
        "ar",
        "Hard",
        ["mcq"],
        provider="groq",
        api_key=groq_key,
        model="",
    )
    print("groq mcq", len(payload.get("mcq", [])))
except Exception as exc:
    print("groq ERR", type(exc).__name__, exc)

print("\nTesting pipeline with deepseek->groq fallback...")
deepseek_match = re.search(r'DEEPSEEK_API_KEY = "([^"]*)"', secrets)
deepseek_key = deepseek_match.group(1) if deepseek_match else ""
try:
    payload, meta = generate_from_document(
        text=sample,
        lang="ar",
        difficulty="Hard",
        types=["mcq"],
        provider="deepseek",
        api_key=deepseek_key,
        fallback_provider="groq",
        fallback_api_key=groq_key,
    )
    print("pipeline mcq", len(payload.get("mcq", [])))
    print("meta", meta)
except json.JSONDecodeError as exc:
    print("JSONDecodeError", exc)
except Exception as exc:
    print("ERR", type(exc).__name__, exc)
