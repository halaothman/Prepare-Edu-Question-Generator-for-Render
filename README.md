---
title: Edu Question Generator
emoji: 📝
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: "1.32.0"
app_file: app.py
pinned: false
---

# Edu Question Generator

تطبيق Streamlit لتوليد أسئلة امتحانية (MCQ) من PDF/DOCX/TXT باستخدام **DeepSeek** (والإعدادات الاختيارية الأخرى).

1. افتح الرابط
2. ارفع **PDF / DOCX / TXT**
3. **توليد الأسئلة**
4. راجع الجدول وحمّل **Excel**

## Features (current)

- Pipeline: تقسيم منطقي → ~20 سؤالاً → فلترة وجودة
- DeepSeek Chat (V3) عبر `DEEPSEEK_API_KEY`
- عرض رصيد DeepSeek في الواجهة
- تصدير Excel

## Local run

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
# أنشئ .streamlit/secrets.toml من secrets.toml.example
streamlit run app.py
```

## Deploy on Render

See **[docs/RENDER.md](docs/RENDER.md)** (Arabic step-by-step).

Quick: connect GitHub repo → **Blueprint** with `render.yaml` → set **DEEPSEEK_API_KEY** in Environment.

## Deploy on Hugging Face Spaces (alternative)

Best for sharing with a supervisor — no local setup for users.

1. Create account at [huggingface.co](https://huggingface.co)
2. **New Space** → SDK: **Streamlit** → upload this repo
3. **Settings → Secrets** → add:
   ```
   HF_TOKEN = hf_...
   LLM_PROVIDER = huggingface
   ```
4. Wait for the Space to build (~2 min)
5. Share the URL: `https://huggingface.co/spaces/YOUR_USERNAME/edu-question-generator`

The professor opens the link, uploads a file, and generates questions — no API key needed on their side.

### Qwen models (cloud)

| Model | Notes |
|-------|-------|
| Qwen2.5-7B | Free tier, recommended |
| Qwen2.5-14B | Better quality |
| Qwen2.5-72B | Strongest — runs on HF servers, not your PC |

## Notes

- Large PDFs are handled by splitting text into chunks (no embeddings or FAISS)
- Each segment is sent directly to the model to generate questions
- Hugging Face Spaces is accessible from most regions including Syria
