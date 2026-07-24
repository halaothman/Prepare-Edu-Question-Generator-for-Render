# Colab — Edu Question Generator

## خطوات البدء

### 1) حساب OpenAI
- سجّلي على https://platform.openai.com
- أنشئي API Key من: https://platform.openai.com/api-keys
- أضيفي رصيد (Billing) — التكلفة تقريباً $0.05–0.30 لكل تجربة

### 2) فتح Notebook
- اذهبي إلى https://colab.research.google.com
- **File → Upload notebook**
- ارفعي `Edu_Question_Generator.ipynb`

### 3) التشغيل
- **Runtime → Run all**
- أدخلي API Key عندما يُطلب
- ارفعي ملف PDF / Word / TXT
- انتظري التوليد
- حمّلي Excel من آخر خلية

### 4) الإعدادات (الخلية الثانية)
```python
MODEL = "gpt-4.1"       # أو gpt-4o | o3
DIFFICULTY = "Hard"     # Easy | Medium | Hard
NUM_QUESTIONS = 10
```

## ملاحظات
- للتعلم العميق + حسابات: استخدمي `Hard` + `gpt-4.1` أو `o3`
- Colab للتطوير والتجربة فقط — **رابط الدكتور: استضافة Render** (انظري `docs/RENDER.md`)
