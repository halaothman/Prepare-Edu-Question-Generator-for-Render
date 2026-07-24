# استضافة على Render (تحديث تلقائي من GitHub)

دليل نشر تطبيق **Streamlit** (`app.py`) على [Render](https://render.com) مع **نشر تلقائي** عند كل `git push` إلى `main`.

**المستودع:** [halaothman/Prepare-Edu-Question-Generator-for-Render](https://github.com/halaothman/Prepare-Edu-Question-Generator-for-Render)

## أمان المفاتيح

| محلي (جهازك) | على Render |
|--------------|------------|
| `.streamlit/secrets.toml` | **Environment Variables** في لوحة Render |
| مُستثنى في `.gitignore` — **لا يُرفع إلى GitHub** | `DEEPSEEK_API_KEY` = Secret (قيمة `sk-...`) |

- انسخ `.streamlit/secrets.toml.example` إلى `secrets.toml` للتطوير المحلي فقط.
- **لا** تضع المفتاح في `render.yaml` ولا في أي ملف داخل المستودع.
- التطبيق يقرأ المفتاح من `st.secrets` محلياً، ومن **متغير البيئة** `DEEPSEEK_API_KEY` على Render (انظر `app.py` → `_read_secret`).

## 1) تأكد أن GitHub جاهز

الكود على `main` بدون `secrets.toml`. بعد أي تعديل:

```bash
git add .
git commit -m "your message"
git push origin main
```

## 2) أول نشر — Blueprint (موصى به)

1. [Render Dashboard](https://dashboard.render.com) → **New** → **Blueprint**
2. اربط حساب **GitHub** واختر المستودع `Prepare-Edu-Question-Generator-for-Render`
3. Render يقرأ `render.yaml` من الجذر وينشئ **Web Service** اسمه `edu-question-generator`
4. قبل أو بعد أول Build، افتح الخدمة → **Environment**:
   - **Key:** `DEEPSEEK_API_KEY`
   - **Value:** مفتاح DeepSeek (`sk-...`)
   - فعّل **Secret** (إخفاء القيمة)
5. **Save Changes** — Render يعيد النشر إن لزم
6. انتظر **Build** ثم **Live** (~3–5 دقائق)
7. الرابط: `https://edu-question-generator-xxxx.onrender.com`

في `render.yaml` مضبوط:

- `branch: main`
- `autoDeployTrigger: commit` → **كل push إلى main يطلق build + deploy تلقائياً**

### التحقق من التحديث التلقائي

1. الخدمة → **Settings** → **Build & Deploy**
2. **Auto-Deploy** يجب أن يكون **On Commit** (أو ما يعادله)
3. **Branch:** `main`

## 3) نشر يدوي (بدون Blueprint)

1. **New** → **Web Service** → نفس المستودع، فرع `main`
2. **Language:** Python  
3. **Build Command:**  
   `pip install --upgrade pip && pip install -r requirements.txt`
4. **Start Command:**  
   `streamlit run app.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true --browser.gatherUsageStats=false`
5. **Health Check Path:** `/_stcore/health`
6. **Auto-Deploy:** On Commit
7. **Environment Variables:**

| Key | Value |
|-----|--------|
| `PYTHON_VERSION` | `3.11.9` |
| `LLM_PROVIDER` | `deepseek` |
| `DEEPSEEK_MODEL` | `deepseek-chat` |
| `DEEPSEEK_API_KEY` | *(Secret — من لوحة Render فقط)* |
| `AUTO_FALLBACK_TO_GROQ` | `false` |
| `SHOW_ALTERNATE_LLM_PROVIDERS` | `false` |
| `TARGET_QUESTIONS_TOTAL` | `20` |

(اختياري) `GROQ_API_KEY` إذا فعّلت مزوداً احتياطياً لاحقاً.

## 4) مشاركة الرابط

- على **Free**: الخدمة **تنام** بعد ~15 دقيقة بدون زيارات — أول فتح قد يستغرق 30–60 ثانية.
- توليد ~20 سؤالاً من ملف كبير قد يستغرق **عدة دقائق**؛ إن انقطع الاتصال فكّر في خطة **Starter** أو ملف أصغر.

## 5) استكشاف الأخطاء

| المشكلة | الحل |
|---------|------|
| «خدمة التوليد غير مهيّأة» | أضف `DEEPSEEK_API_KEY` في Environment → Save → انتظر Deploy |
| لا ينشر بعد push | تحقق من Auto-Deploy وفرع `main` وصلاحيات تطبيق Render على GitHub |
| WebSocket / الصفحة لا تفتح | `enableXsrfProtection = false` في `.streamlit/config.toml` (موجود) |
| Build فشل | `requirements.txt` و `runtime.txt` / `PYTHON_VERSION=3.11.9` |
| 502 بعد انتظار طويل | Free tier + مهلة الطلب؛ جرّب ملفاً أصغر |

## 6) سير العمل المعتاد

1. عدّل الكود محلياً (المفاتيح في `secrets.toml` فقط على جهازك)
2. `git push origin main`
3. Render يبني وينشر تلقائياً — راقب **Events** في Dashboard
4. افتح الرابط وجرّب رفع ملف و«توليد الأسئلة»
