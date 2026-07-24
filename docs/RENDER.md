# استضافة على Render

دليل نشر تطبيق **Streamlit** (`app.py`) على [Render](https://render.com).

## المتطلبات

- حساب على Render (GitHub/GitLab للربط بالمستودع)
- المستودع مرفوع على GitHub **بدون** `.streamlit/secrets.toml` (مُستثنى في `.gitignore`)
- مفتاح **DeepSeek**: `DEEPSEEK_API_KEY`

## 1) رفع الكود

```bash
git add .
git commit -m "Prepare Render deployment"
git push origin main
```

## 2) النشر عبر Blueprint (موصى به)

1. [Render Dashboard](https://dashboard.render.com) → **New** → **Blueprint**
2. اربط مستودع `edu-question-generator`
3. Render يقرأ `render.yaml` وينشئ خدمة **Web Service**
4. عند أول نشر، أضف متغيراً سرياً:
   - **Environment** → **DEEPSEEK_API_KEY** = `sk-...`
5. انتظر **Build** ثم **Deploy** (~3–5 دقائق)

## 3) النشر اليدوي (بدون Blueprint)

1. **New** → **Web Service** → نفس المستودع
2. **Language**: Python  
3. **Build Command**:  
   `pip install --upgrade pip && pip install -r requirements.txt`
4. **Start Command**:  
   `streamlit run app.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true --browser.gatherUsageStats=false`
5. **Health Check Path**: `/_stcore/health`
6. **Environment Variables** (مثال):

| Key | Value |
|-----|--------|
| `PYTHON_VERSION` | `3.11.9` |
| `LLM_PROVIDER` | `deepseek` |
| `DEEPSEEK_MODEL` | `deepseek-chat` |
| `DEEPSEEK_API_KEY` | *(Secret)* |
| `AUTO_FALLBACK_TO_GROQ` | `false` |
| `SHOW_ALTERNATE_LLM_PROVIDERS` | `false` |
| `TARGET_QUESTIONS_TOTAL` | `20` |

## 4) مشاركة الرابط مع الدكتور

بعد النشر: `https://edu-question-generator-xxxx.onrender.com`

- على **Free**: الخدمة **تنام** بعد ~15 دقيقة بدون زيارات — أول فتح قد يستغرق 30–60 ثانية.
- توليد أسئلة من ملف كبير (46 صفحة) قد يستغرق **عدة دقائق**؛ إن انقطع الاتصال فكّري في **Starter** ($7/شهر) أو Hugging Face Spaces.

## 5) استكشاف الأخطاء

| المشكلة | الحل |
|---------|------|
| «مفتاح DeepSeek غير مهيأ» | أضف `DEEPSEEK_API_KEY` في Environment وأعد Deploy |
| الصفحة لا تفتح / WebSocket | تأكد من `enableXsrfProtection = false` في `.streamlit/config.toml` |
| Build فشل | تحقق من `requirements.txt` و `PYTHON_VERSION=3.11.9` |
| 502 بعد انتظار طويل | Free tier + مهلة؛ جرّب ملفاً أصغر أو خطة مدفوعة |

## ملاحظة أمان

لا تضع المفتاح في `render.yaml` — استخدم **Environment** فقط (`sync: false` في Blueprint).
