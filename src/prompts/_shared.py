from __future__ import annotations

from typing import Literal

Difficulty = Literal["Easy", "Medium", "Hard"]

LANGUAGE_RULES_AR = """
- استخدم العربية الفصحى مع مصطلحات تقنية إنجليزية شائعة فقط (مثل CNN, Attention, FLOPs)
- ممنوع تماماً: الأحرف الصينية أو اليابانية أو الكورية أو أي رموز غريبة
- مسموح فقط: العربية، الإنجليزية، الأرقام، وعلامات رياضية شائعة (+ - × ÷ = ^ / ( ) [ ] %)
"""


def distribution_blocks(num_questions: int | None) -> tuple[str, str]:
    if num_questions is None:
        target_line = (
            "Generate as many valid Hard MCQs as the uploaded document fully supports. "
            "There is no target count."
        )
        distribution_block = """Generate as many valid questions as the document supports.

Aim for roughly 50% computational and 50% analytical when both types are supported.

Do NOT invent questions to reach a target count or to balance the ratio.

If only computational or only analytical questions are supported, return only that type.

Return fewer questions, or even zero, rather than weak or unsupported questions."""
    else:
        computational_count = num_questions // 2
        analytical_count = num_questions - computational_count
        target_line = f"Target: up to {num_questions} questions."
        distribution_block = f"""Generate:

50% Computational questions ({computational_count} questions)

50% Analytical questions ({analytical_count} questions)

If the document cannot support the full count for either type, return fewer valid questions of that type."""
    return target_line, distribution_block


def output_format_section(context: str) -> str:
    return f"""Return ONLY valid JSON.

Return every valid question the document supports.
An empty "mcq" array is acceptable if no valid question can be generated.
Do NOT pad the list with weak questions.

Use exactly this schema:

{{
  "mcq": [
    {{
      "q": "",
      "options": [
        "",
        "",
        "",
        ""
      ],
      "answer": "",
      "solution": "",
      "type": "computational | analytical",
      "difficulty": "hard"
    }}
  ]
}}

For "type" use only: "computational" or "analytical".

UPLOADED DOCUMENT:
{context}"""
