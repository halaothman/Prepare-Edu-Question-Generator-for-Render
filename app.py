from __future__ import annotations

import html
import json
import os
import tempfile

import streamlit as st

from src.config import (
    DEFAULT_DEEPSEEK_MODEL,
    DEFAULT_GROQ_MODEL,
    GROQ_LIMIT_ERROR,
    GROQ_REQUEST_TOO_LARGE,
    LLM_INSUFFICIENT_BALANCE,
    LLM_LIMIT_ERROR,
    LLM_REQUEST_TOO_LARGE,
    PIPELINE_ALL_SEGMENTS_FAILED,
    SHOW_ALTERNATE_LLM_PROVIDERS,
    TARGET_QUESTIONS_TOTAL,
    deepseek_model_display_name,
)
from src.excel_export import dataframe_to_excel, questions_to_dataframe
from src.generator import QuestionType, detect_lang
from src.llm_client import GROQ_MODEL_LABELS, deepseek_balance_status, groq_quota_status
from src.loaders import load_text
from src.pipeline import generate_from_document

ProviderChoice = str

PROVIDER_OPTIONS: dict[ProviderChoice, str] = {
    "deepseek": f"{deepseek_model_display_name()} — {DEFAULT_DEEPSEEK_MODEL}",
    "groq": f"Qwen (Groq) — {GROQ_MODEL_LABELS.get(DEFAULT_GROQ_MODEL, DEFAULT_GROQ_MODEL)}",
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
        "last_progress_log": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


@st.cache_data(ttl=120, show_spinner=False)
def fetch_groq_quota(api_key: str) -> dict:
    return groq_quota_status(api_key)


@st.cache_data(ttl=120, show_spinner=False)
def fetch_deepseek_balance(api_key: str) -> dict:
    return deepseek_balance_status(api_key)


def format_deepseek_balance_line(status: dict) -> str:
    if not status.get("ok"):
        return "تعذّر جلب الرصيد"
    total = status.get("total_balance")
    currency = status.get("currency", "USD")
    if not total:
        return "غير متاح"
    symbol = "$" if currency == "USD" else ""
    suffix = currency if currency != "USD" else "USD"
    if status.get("is_available") is False:
        return f"{symbol}{total} {suffix} — غير كافٍ"
    return f"{symbol}{total} {suffix} متبقٍ"


def format_quota_line(model: dict) -> str:
    remaining = model.get("remaining_requests")
    limit = model.get("limit_requests")
    if remaining is not None and limit is not None:
        return f"{remaining}/{limit} طلب متبقٍ"
    if model.get("available"):
        return "متاح"
    return "غير متاح"


def provider_model_label(provider: str) -> str:
    if provider == "deepseek":
        return f"{deepseek_model_display_name()} ({DEFAULT_DEEPSEEK_MODEL})"
    if provider == "groq":
        return f"{GROQ_MODEL_LABELS.get(DEFAULT_GROQ_MODEL, DEFAULT_GROQ_MODEL)} (Groq)"
    return provider


def format_pipeline_progress_line(stage: str, data: dict) -> str:
    if stage == "extract_done":
        lang = data.get("lang", "?")
        return f"① استخراج النص: {data.get('text_chars', 0):,} حرف — اللغة: {lang}"
    if stage == "chunking":
        sampled = " (عيّنة)" if data.get("segments_sampled") else ""
        target = data.get("target_questions", TARGET_QUESTIONS_TOTAL)
        return (
            f"② تقسيم **منطقي** للمستند: {data.get('segments_total', 0)} جزء — "
            f"يُرسَل {data.get('segments_used', 0)} جزء للنموذج{sampled} "
            f"(هدف {target} سؤالاً إجمالاً)"
        )
    if stage == "segment_llm_start":
        seg_q = data.get("segment_questions")
        per_part = f" — {seg_q} سؤال/جزء" if seg_q else ""
        return (
            f"③ [{data.get('index')}/{data.get('total')}] "
            f"النموذج يولّد أسئلة — {provider_model_label(str(data.get('provider', '')))}"
            f"{per_part} — {data.get('segment_chars', 0):,} حرف…"
        )
    if stage == "segment_llm_done":
        return (
            f"   ✓ [{data.get('index')}/{data.get('total')}] "
            f"ردّ النموذج: {data.get('mcq_count', 0)} سؤال MCQ"
        )
    if stage == "segment_skip":
        return f"   ⚠ [{data.get('index')}/{data.get('total')}] تخطّي الجزء (JSON أو خطأ مزود)"
    if stage == "merge":
        return (
            f"④ دمج النتائج: {data.get('mcq_raw', 0)} سؤال من "
            f"{data.get('segment_payloads', 0)} جزء"
        )
    if stage == "merge_done":
        return f"   إزالة التكرار → {data.get('mcq_after_dedupe', 0)} سؤال"
    if stage == "validate":
        return f"⑤ التحقق من الأسئلة مقابل المستند ({data.get('mcq_before_filter', 0)} قبل الفلترة)…"
    if stage == "cap":
        return (
            f"⑥ اختيار أفضل {data.get('target_questions', TARGET_QUESTIONS_TOTAL)} سؤالاً "
            f"مع تنوع (حساب/تحليل) من {data.get('mcq_before_cap', 0)} مرشّح"
        )
    if stage == "done":
        return (
            f"⑦ اكتمل — {data.get('mcq_final', 0)} سؤال صالح "
            f"({provider_model_label(str(data.get('provider_used', '')))})"
        )
    return f"{stage}: {data}"


def pipeline_status_headline(stage: str, data: dict) -> str:
    if stage == "extract_done":
        return "استخراج النص من الملف…"
    if stage == "chunking":
        return f"تقسيم المستند إلى {data.get('segments_used', 0)} جزء…"
    if stage == "segment_llm_start":
        return (
            f"النموذج يعمل — الجزء {data.get('index')} من {data.get('total')} "
            f"({provider_model_label(str(data.get('provider', '')))})"
        )
    if stage in {"segment_llm_done", "segment_skip"}:
        return f"الجزء {data.get('index')} من {data.get('total')} — متابعة…"
    if stage in {"merge", "merge_done"}:
        return "دمج الأسئلة وإزالة التكرار…"
    if stage == "validate":
        return "التحقق من جودة الأسئلة…"
    if stage == "cap":
        return f"اختيار أفضل {TARGET_QUESTIONS_TOTAL} سؤالاً…"
    if stage == "done":
        return "اكتمل التوليد"
    return "جاري التوليد…"


def make_pipeline_progress_ui():
    lines: list[str] = []
    log_box = st.empty()
    progress_bar = st.progress(0.0)

    def callback(stage: str, data: dict) -> None:
        lines.append(format_pipeline_progress_line(stage, data))
        log_box.markdown(
            "<div class='rtl-block' style='font-size:0.88rem;line-height:1.6;color:#334155'>"
            + "<br>".join(html.escape(line) for line in lines)
            + "</div>",
            unsafe_allow_html=True,
        )
        if stage == "chunking":
            progress_bar.progress(0.08)
        elif stage == "segment_llm_start":
            index = int(data.get("index") or 1)
            total = max(1, int(data.get("total") or 1))
            # Most time is spent waiting on the model per segment.
            progress_bar.progress(0.08 + 0.75 * (index - 1) / total)
        elif stage == "segment_llm_done":
            index = int(data.get("index") or 1)
            total = max(1, int(data.get("total") or 1))
            progress_bar.progress(0.08 + 0.75 * index / total)
        elif stage == "merge":
            progress_bar.progress(0.88)
        elif stage == "validate":
            progress_bar.progress(0.94)
        elif stage == "cap":
            progress_bar.progress(0.97)
        elif stage == "done":
            progress_bar.progress(1.0)

    return callback, lines, progress_bar


def render_provider_status(selected_provider: ProviderChoice) -> None:
    if selected_provider == "deepseek":
        deepseek_key = get_deepseek_api_key()
        primary_label = html.escape(provider_model_label("deepseek"))

        if not deepseek_key:
            st.markdown(
                '<div class="model-status">⚠️ <strong>DeepSeek:</strong> مفتاح API غير مهيأ أو فارغ — '
                "افتح <code>.streamlit/secrets.toml</code>، ضع المفتاح، "
                "<strong>احفظ الملف (Ctrl+S)</strong>، ثم أعد تشغيل التطبيق.</div>",
                unsafe_allow_html=True,
            )
            return

        balance = fetch_deepseek_balance(deepseek_key)
        balance_line = html.escape(format_deepseek_balance_line(balance))

        st.markdown(
            (
                f'<div class="model-status">'
                f"📌 <strong>النموذج المختار:</strong> {primary_label}<br>"
                f"💰 <strong>رصيد DeepSeek:</strong> {balance_line}"
                f"</div>"
            ),
            unsafe_allow_html=True,
        )
        return

    if selected_provider != "groq":
        return

    api_key = get_groq_api_key()
    if not api_key:
        st.markdown(
            '<div class="model-status">⚠️ <strong>Groq:</strong> مفتاح API غير مهيأ</div>',
            unsafe_allow_html=True,
        )
        return

    status = fetch_groq_quota(api_key)
    if not status.get("ok") or not status.get("models"):
        st.markdown(
            '<div class="model-status">⚠️ <strong>Groq:</strong> تعذّر التحقق من حالة النموذج</div>',
            unsafe_allow_html=True,
        )
        return

    primary = status["models"][0]
    primary_label = html.escape(primary["label"])
    primary_quota = html.escape(format_quota_line(primary))

    if primary.get("available"):
        limit_text = "لم يصل للحد بعد"
        if primary.get("remaining_requests") == 0:
            limit_text = "وصل الحد اليومي"
        st.markdown(
            (
                f'<div class="model-status">'
                f"🤖 <strong>النموذج:</strong> {primary_label} (Groq)<br>"
                f"📊 <strong>الحد اليومي:</strong> {primary_quota} — {limit_text}"
                f"</div>"
            ),
            unsafe_allow_html=True,
        )
        return

    fallback = next((model for model in status["models"][1:] if model.get("available")), None)
    if fallback:
        fallback_label = html.escape(fallback["label"])
        fallback_quota = html.escape(format_quota_line(fallback))
        st.markdown(
            (
                f'<div class="model-status">'
                f"🤖 <strong>النموذج الأساسي:</strong> {primary_label} — وصل الحد اليومي<br>"
                f"🔄 <strong>الاحتياطي:</strong> {fallback_label} — {fallback_quota} — لم يصل للحد بعد"
                f"</div>"
            ),
            unsafe_allow_html=True,
        )
        return

    st.markdown(
        (
            f'<div class="model-status">'
            f"🤖 <strong>النموذج:</strong> {primary_label} (Groq)<br>"
            f"⛔ <strong>الحد اليومي:</strong> تم استنفاد الحد لجميع النماذج — حاول لاحقاً"
            f"</div>"
        ),
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
        "اختر النموذج",
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
            st.error("مفتاح DeepSeek غير مهيأ. أضف DEEPSEEK_API_KEY في secrets.toml.")
            st.stop()
    else:
        api_key = groq_key
        if not api_key:
            st.error("مفتاح Groq غير مهيأ. أضف GROQ_API_KEY في secrets.toml.")
            st.stop()

    if not uploaded:
        st.error("يرجى رفع ملف PDF أو DOCX أو TXT.")
        st.stop()

    with st.status("جاري توليد الأسئلة…", expanded=True) as run_status:
        progress_cb, progress_lines, _progress_bar = make_pipeline_progress_ui()
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
                        "انتهى رصيد DeepSeek. يرجى شحن الرصيد من platform.deepseek.com "
                        "ثم المحاولة مرة أخرى. (لا يتم التبديل تلقائياً إلى Qwen أو Llama.)"
                    )
                elif message == PIPELINE_ALL_SEGMENTS_FAILED:
                    st.error(
                        "تعذّر توليد أسئلة من جميع أجزاء المستند. "
                        "تحقق من رصيد/حد المزود المختار أو جرّب ملفاً أصغر."
                    )
                elif message in {GROQ_LIMIT_ERROR, LLM_LIMIT_ERROR}:
                    st.error("تم استنفاد حد المزود. حاول لاحقاً.")
                elif message in {GROQ_REQUEST_TOO_LARGE, LLM_REQUEST_TOO_LARGE}:
                    st.error(
                        "تعذّر معالجة بعض أجزاء المستند ضمن حد المزود. "
                        "جرّب مرة أخرى — النظام يقسّم المستند تلقائياً."
                    )
                else:
                    st.error("تعذّر توليد الأسئلة. حاول لاحقاً.")
                st.stop()
            except json.JSONDecodeError:
                run_status.update(label="فشل — JSON", state="error")
                st.error(
                    "تعذّر قراءة نتيجة النموذج (JSON). "
                    "جرّب مرة أخرى. حجم الملف بالميغا ليس المشكلة — المهم حجم النص المستخرج."
                )
                st.stop()

            df = questions_to_dataframe(payload, default_difficulty=DIFFICULTY)
            st.session_state["questions_df"] = df
            st.session_state["last_filename"] = os.path.splitext(uploaded.name)[0]
            st.session_state["run_meta"] = run_meta
            st.session_state["last_progress_log"] = list(progress_lines)
            run_status.update(label=pipeline_status_headline("done", {"mcq_final": len(df)}), state="complete")
        finally:
            os.unlink(tmp_path)

    if st.session_state.get("last_progress_log"):
        with st.expander("سجل التوليد (Debug)", expanded=False):
            st.markdown(
                "<div class='rtl-block' style='font-size:0.88rem;line-height:1.6'>"
                + "<br>".join(html.escape(line) for line in st.session_state["last_progress_log"])
                + "</div>",
                unsafe_allow_html=True,
            )

    count = len(st.session_state["questions_df"]) if st.session_state["questions_df"] is not None else 0
    meta = st.session_state.get("run_meta") or {}
    model_used = provider_model_label(str(meta.get("provider_used", selected_provider)))
    if meta:
        skipped = meta.get("segments_skipped", 0)
        skipped_line = f" — تخطّي {skipped} جزء." if skipped else ""
        st.caption(
            f"🤖 النموذج: {model_used} — "
            f"النص: {meta.get('text_chars', 0):,} حرف — "
            f"{meta.get('segments_used', 0)} جزء منطقي — "
            f"هدف {meta.get('target_questions', TARGET_QUESTIONS_TOTAL)} سؤال "
            f"(ظهر {count}).{skipped_line}"
        )
    if count == 0:
        st.warning("لم يُولَّد أي سؤال صالح من هذا المستند.")
    else:
        st.success(f"تم توليد {count} سؤالاً صالحاً — النموذج: {model_used}")

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
