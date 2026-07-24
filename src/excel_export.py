from __future__ import annotations

import io

import pandas as pd


def questions_to_dataframe(payload: dict, default_difficulty: str = "Medium") -> pd.DataFrame:
    rows: list[dict] = []
    counter = 1

    for item in payload.get("mcq", []):
        options = item.get("options", [])
        rows.append(
            {
                "#": counter,
                "Type": "MCQ",
                "Question Kind": item.get("question_kind", ""),
                "Question": item.get("q", ""),
                "Option A": options[0] if len(options) > 0 else "",
                "Option B": options[1] if len(options) > 1 else "",
                "Option C": options[2] if len(options) > 2 else "",
                "Option D": options[3] if len(options) > 3 else "",
                "Answer": item.get("answer", ""),
                "Solution": item.get("solution") or item.get("explanation", ""),
            }
        )
        counter += 1

    for item in payload.get("tf", []):
        answer = item.get("answer")
        if isinstance(answer, bool):
            answer_text = "True" if answer else "False"
        else:
            answer_text = str(answer)
        rows.append(
            {
                "#": counter,
                "Type": "True/False",
                "Question Kind": item.get("question_kind", ""),
                "Question": item.get("q", ""),
                "Option A": "True",
                "Option B": "False",
                "Option C": "",
                "Option D": "",
                "Answer": answer_text,
                "Solution": item.get("solution") or item.get("explanation", ""),
            }
        )
        counter += 1

    for item in payload.get("short", []):
        rows.append(
            {
                "#": counter,
                "Type": "Short Answer",
                "Question Kind": item.get("question_kind", ""),
                "Question": item.get("q", ""),
                "Option A": "",
                "Option B": "",
                "Option C": "",
                "Option D": "",
                "Answer": item.get("answer", ""),
                "Solution": item.get("solution") or item.get("explanation", ""),
            }
        )
        counter += 1

    columns = [
        "#",
        "Type",
        "Question Kind",
        "Question",
        "Option A",
        "Option B",
        "Option C",
        "Option D",
        "Answer",
        "Solution",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows)[columns]


def dataframe_to_excel(df: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Questions")
    return buffer.getvalue()
