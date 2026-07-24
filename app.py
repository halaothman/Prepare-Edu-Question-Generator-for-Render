from __future__ import annotations

import html
import json
import os
import tempfile

import streamlit as st

from src.config import (
    GROQ_LIMIT_ERROR,
    GROQ_REQUEST_TOO_LARGE,
    LLM_INSUFFICIENT_BALANCE,
    LLM_LIMIT_ERROR,
    LLM_REQUEST_TOO_LARGE,
    PIPELINE_ALL_SEGMENTS_FAILED,
    SHOW_ALTERNATE_LLM_PROVIDERS,
    TARGET_QUESTIONS_TOTAL,
)
from src.excel_export import dataframe_to_excel, questions_to_dataframe
from src.generator import QuestionType, detect_lang
from src.loaders import load_text
from src.pipeline import generate_from_document

ProviderChoice = str

PROVIDER_OPTIONS: dict[ProviderChoice, str] = {
    "deepseek": "الافتراضي",
    "groq": "احتياطي",
}
DEFAULT_PROVIDER: ProviderChoice = "deepseek"
DIFFICULTY = "Hard"
QUESTION_TYPES: list[QuestionType] = ["mcq"]
MATH_FOCUS = True
DL_FOCUS = True

st.set_page_config(
    page_title="توليد أسئلة تعليمية باستخدام نماذج اللغات الكبيرة",
    page_icon="📝",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    .main-title { font-size: 2rem; font-weight: 700; margin-bottom: 0.25rem; text-align: center; }
    .sub-title { color: #64748b; margin-bottom: 1.5rem; text-align: center; }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #2563eb, #1d4ed8);
        color: white;
        border: 0;
        padding: 0.85rem 1rem;
        font-weight: 600;
        font-size: 1.05rem;
    }
    [data-testid="stSidebar"] { display: none; }
    .rtl-block {
        direction: rtl;
        text-align: right;
    }
    .questions-header {
        direction: rtl;
        text-align: right;
    }
    .model-status {
        direction: rtl;
        text-align: right;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 0.5rem;
        padding: 0.75rem 1rem;
        margin-bottom: 1rem;
        font-size: 0.95rem;
        color: #334155;
    }
    div[data-testid="stRadio"] {
        direction: rtl;
        text-align: right;
        margin-bottom: 0.75rem;
    }
    div[data-testid="stRadio"] label p {
        font-size: 0.95rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def _read_secret(name: str) -> str | None:
    try:
        value = st.secrets[name]
        if value is None:
            return None
        text = str(value).strip()
        return text or None
    except (KeyError, st.errors.StreamlitSecretNotFoundError, AttributeError):
        pass
    env_value = os.getenv(name)
    if env_value is None:
        return None
    text = env_value.strip()
    return text or None


def get_deepseek_api_key() -> str | None:
    return _read_secret("DEEPSEEK_API_KEY")


def get_groq_api_key() -> str | None:
    return _read_secret("GROQ_API_KEY")


def init_state() -> None:
    defaults = {
        "questions_df": None,
        "last_filename": None,
        "segment_count": 0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


_PIPELINE_PHASES: list[tuple[str, str]] = [
    ("extract", "استخراج النص من الملف"),
    ("chunk", "تقسيم المستند"),
    ("generate", "توليد الأسئلة"),
    ("merge", "دمج النتائج"),
    ("validate", "التحقق من الأسئلة"),
    ("select", "اختيار الأسئلة المناسبة"),
    ("done", "اكتمل التوليد"),
]


def _stage_phase_key(stage: str) -> str:
    if stage == "extract_done":
        return "extract"
    if stage == "chunking":
        return "chunk"
    if stage in {"segment_llm_start", "segment_llm_done", "segment_skip"}:
        return "generate"
    if stage in {"merge", "merge_done"}:
        return "merge"
    if stage == "validate":
        return "validate"
    if stage == "cap":
        return "select"
    if stage == "done":
        return "done"
    return "generate"


def pipeline_status_headline(stage: str, data: dict) -> str:
    del data
    labels = dict(_PIPELINE_PHASES)
    phase = _stage_phase_key(stage)
    label = labels.get(phase, "جاري التوليد")
    if phase == "done":
        return label
    return f"{label}…"


def _render_phase_checklist(completed: set[str], current: str) -> str:
    rows: list[str] = []
    for key, label in _PIPELINE_PHASES:
        if key in completed:
            rows.append(f"✓ {html.escape(label)}")
        elif key == current:
            rows.append(f"▸ {html.escape(label)}…")
        else:
            rows.append(f"○ {html.escape(label)}")
    return (
        "<div class='rtl-block' style='font-size:0.95rem;line-height:1.9;color:#334155'>"
        + "<br>".join(rows)
        + "</div>"
    )


def make_pipeline_progress_ui():
    completed_phases: set[str] = set()
    current_phase = "extract"
    log_box = st.empty()
    progress_bar = st.progress(0.0)
    order = [p[0] for p in _PIPELINE_PHASES]

    log_box.markdown(_render_phase_checklist(completed_phases, current_phase), unsafe_allow_html=True)

    def callback(stage: str, data: dict) -> None:
        nonlocal current_phase
        phase = _stage_phase_key(stage)
        if phase in order:
            idx = order.index(phase)
            for earlier in order[:idx]:
                completed_phases.add(earlier)
            current_phase = phase
        log_box.markdown(
            _render_phase_checklist(completed_phases, current_phase),
            unsafe_allow_html=True,
        )
        if stage == "chunking":
            progress_bar.progress(0.12)
        elif stage in {"segment_llm_start", "segment_llm_done", "segment_skip"}:
            index = int(data.get("index") or 1)
            total = max(1, int(data.get("total") or 1))
            progress_bar.progress(0.12 + 0.58 * index / total)
        elif stage in {"merge", "merge_done"}:
            progress_bar.progress(0.78)
        elif stage == "validate":
            progress_bar.progress(0.88)
        elif stage == "cap":
            progress_bar.progress(0.94)
        elif stage == "done":
            completed_phases.update(order)
            current_phase = "done"
            log_box.markdown(
                _render_phase_checklist(completed_phases, "done"),
                unsafe_allow_html=True,
            )
            progress_bar.progress(1.0)

    return callback, progress_bar

def render_provider_status(selected_provider: ProviderChoice) -> None:
    key_ok = (
        get_deepseek_api_key()
        if selected_provider == "deepseek"
        else get_groq_api_key()
    )
    if key_ok:
        return
    st.markdown(
        '<div class="model-status">⚠️ خدمة التوليد غير مهيّأة. '
        "تواصل مع مسؤول النظام.</div>",
        unsafe_allow_html=True,
    )


def rtl_markdown(content: str) -> None:
    st.markdown(
        f'<div class="rtl-block">{content}</div>',
        unsafe_allow_html=True,
    )


def render_questions(df) -> None:
    type_labels = {
        "MCQ": "اختيار من متعدد",
        "True/False": "صح / خطأ",
        "Short Answer": "إجابة قصيرة",
    }

    for _, row in df.iterrows():
        q_type = row["Type"]
        label = type_labels.get(q_type, q_type)
        with st.container(border=True):
            kind = row.get("Question Kind", "")
            kind_line = f" · {html.escape(str(kind))}" if kind and str(kind).strip() else ""
            rtl_markdown(f"<h3>س{row['#']} — {label}{kind_line}</h3>")
            rtl_markdown(f"<p><strong>{html.escape(str(row['Question']))}</strong></p>")

            if q_type == "MCQ":
                options_html = ""
                for letter in ("A", "B", "C", "D"):
                    option = row.get(f"Option {letter}", "")
                    if option:
                        options_html += (
                            f"<p><strong>{letter}.</strong> {html.escape(str(option))}</p>"
                        )
                rtl_markdown(options_html)
            elif q_type == "True/False":
                rtl_markdown("<p>○ صح &nbsp;&nbsp; ○ خطأ</p>")

            rtl_markdown(
                f"<p>✅ <strong>الإجابة:</strong> {html.escape(str(row['Answer']))}</p>"
            )

            solution = row.get("Solution", row.get("Explanation", ""))
            if solution and str(solution).strip():
                with st.expander("💡 الحل"):
                    rtl_markdown(f"<p>{html.escape(str(solution))}</p>")


init_state()

st.markdown(
    '<div class="main-title">📝 توليد أسئلة تعليمية باستخدام نماذج اللغات الكبيرة</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="sub-title">ارفع ملف المحاضرة واضغط توليد الأسئلة</div>',
    unsafe_allow_html=True,
)

selected_provider: ProviderChoice = DEFAULT_PROVIDER
if SHOW_ALTERNATE_LLM_PROVIDERS:
    selected_provider = st.radio(
        "مصدر التوليد",
        options=list(PROVIDER_OPTIONS.keys()),
        index=list(PROVIDER_OPTIONS.keys()).index(DEFAULT_PROVIDER),
        format_func=lambda key: PROVIDER_OPTIONS[key],
        horizontal=True,
    )

render_provider_status(selected_provider)

uploaded = st.file_uploader("PDF / DOCX / TXT", type=["pdf", "docx", "txt"], label_visibility="collapsed")

if st.button("توليد الأسئلة", type="primary", use_container_width=True):
    deepseek_key = get_deepseek_api_key()
    groq_key = get_groq_api_key()

    if selected_provider == "deepseek":
        api_key = deepseek_key
        if not api_key:
            st.error("خدمة التوليد غير مهيّأة. تواصل مع مسؤول النظام.")
            st.stop()
    else:
        api_key = groq_key
        if not api_key:
            st.error("خدمة التوليد غير مهيّأة. تواصل مع مسؤول النظام.")
            st.stop()

    if not uploaded:
        st.error("يرجى رفع ملف PDF أو DOCX أو TXT.")
        st.stop()

    with st.status("جاري توليد الأسئلة…", expanded=True) as run_status:
        progress_cb, _progress_bar = make_pipeline_progress_ui()
        suffix = os.path.splitext(uploaded.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.getbuffer())
            tmp_path = tmp.name

        def on_pipeline_progress(stage: str, data: dict) -> None:
            progress_cb(stage, data)
            run_status.update(label=pipeline_status_headline(stage, data))

        try:
            text = load_text(tmp_path).strip()
            if len(text) < 100:
                run_status.update(label="فشل — نص قصير", state="error")
                st.error("الملف قصير جداً أو لم يُستخرج منه نص.")
                st.stop()

            lang = detect_lang(text)
            on_pipeline_progress(
                "extract_done",
                {"text_chars": len(text), "lang": "العربية" if lang == "ar" else "English"},
            )

            try:
                payload, run_meta = generate_from_document(
                    text=text,
                    lang=lang,
                    difficulty=DIFFICULTY,
                    types=QUESTION_TYPES,
                    model="",
                    provider=selected_provider,
                    api_key=api_key,
                    math_focus=MATH_FOCUS,
                    dl_focus=DL_FOCUS,
                    progress_callback=on_pipeline_progress,
                )
            except RuntimeError as exc:
                run_status.update(label="فشل التوليد", state="error")
                message = str(exc)
                if message == LLM_INSUFFICIENT_BALANCE:
                    st.error(
                        "تعذّر إكمال التوليد (انتهى الرصيد أو الحد المتاح). "
                        "حاول لاحقاً أو تواصل مع مسؤول النظام."
                    )
                elif message == PIPELINE_ALL_SEGMENTS_FAILED:
                    st.error(
                        "تعذّر توليد أسئلة من جميع أجزاء المستند. "
                        "جرّب ملفاً أصغر أو حاول لاحقاً."
                    )
                elif message in {GROQ_LIMIT_ERROR, LLM_LIMIT_ERROR}:
                    st.error("تم استنفاد الحد المتاح. حاول لاحقاً.")
                elif message in {GROQ_REQUEST_TOO_LARGE, LLM_REQUEST_TOO_LARGE}:
                    st.error(
                        "تعذّر معالجة بعض أجزاء المستند. "
                        "جرّب مرة أخرى — النظام يقسّم المستند تلقائياً."
                    )
                else:
                    st.error("تعذّر توليد الأسئلة. حاول لاحقاً.")
                st.stop()
            except json.JSONDecodeError:
                run_status.update(label="فشل — JSON", state="error")
                st.error(
                    "تعذّر قراءة نتيجة التوليد. "
                    "جرّب مرة أخرى. حجم الملف بالميغا ليس المشكلة — المهم حجم النص المستخرج."
                )
                st.stop()

            df = questions_to_dataframe(payload, default_difficulty=DIFFICULTY)
            st.session_state["questions_df"] = df
            st.session_state["last_filename"] = os.path.splitext(uploaded.name)[0]
            st.session_state["run_meta"] = run_meta
            run_status.update(label=pipeline_status_headline("done", {"mcq_final": len(df)}), state="complete")
        finally:
            os.unlink(tmp_path)

    count = len(st.session_state["questions_df"]) if st.session_state["questions_df"] is not None else 0
    meta = st.session_state.get("run_meta") or {}
    if meta:
        skipped = meta.get("segments_skipped", 0)
        skipped_line = f" — تخطّي {skipped} جزء." if skipped else ""
        st.caption(
            f"النص: {meta.get('text_chars', 0):,} حرف — "
            f"{meta.get('segments_used', 0)} جزء — "
            f"هدف {meta.get('target_questions', TARGET_QUESTIONS_TOTAL)} سؤال "
            f"(ظهر {count}).{skipped_line}"
        )
    if count == 0:
        st.warning("لم يُولَّد أي سؤال صالح من هذا المستند.")
    else:
        st.success(f"تم توليد {count} سؤالاً صالحاً")

if st.session_state["questions_df"] is not None and not st.session_state["questions_df"].empty:
    st.markdown('<h2 class="questions-header">الأسئلة</h2>', unsafe_allow_html=True)
    render_questions(st.session_state["questions_df"])

    excel_bytes = dataframe_to_excel(st.session_state["questions_df"])
    base_name = st.session_state.get("last_filename") or "questions"
    st.download_button(
        label="📥 تحميل Excel",
        data=excel_bytes,
        file_name=f"{base_name}_questions.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )
