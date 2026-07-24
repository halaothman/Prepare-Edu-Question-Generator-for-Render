from __future__ import annotations

from ._shared import LANGUAGE_RULES_AR, Difficulty, distribution_blocks, output_format_section


def build_openai_prompt(
    context: str,
    difficulty: Difficulty,
    num_questions: int | None = None,
) -> str:
    """OpenAI / ChatGPT-specific generation prompt — customize here when enabling GPT-4."""
    target_line, distribution_block = distribution_blocks(num_questions)

    return f"""You are an expert university professor generating Arabic MCQs from the uploaded document only.

Customize this prompt in src/prompts/openai.py for ChatGPT / GPT-4-specific behaviour.

{LANGUAGE_RULES_AR}

{target_line}
Difficulty: {difficulty.lower()} only.

Rules:
- Use ONLY the uploaded document. No external knowledge. No hallucination.
- Require reasoning or calculation; forbid one-line recall.
- Exactly four options, one correct answer, plausible distractors.
- {distribution_block.strip()}

{output_format_section(context)}"""
