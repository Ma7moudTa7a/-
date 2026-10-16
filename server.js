require("dotenv").config();
const express = require("express");
const multer = require("multer");
const path = require("path");
const fs = require("fs");
const OpenAI = require("openai");
const Anthropic = require("@anthropic-ai/sdk");

const app = express();

// إعداد مجلد الرفع
const uploadsDir = path.join(__dirname, "uploads");
fs.mkdirSync(uploadsDir, { recursive: true });

// إعداد تخزين الملفات مع الحفاظ على صيغ الصوت المعتمدة
const storage = multer.diskStorage({
  destination: (req, file, cb) => cb(null, uploadsDir),
  filename: (req, file, cb) => {
    const ext = path.extname(file.originalname).toLowerCase();
    const validExts = [".mp3", ".wav", ".m4a", ".ogg", ".webm", ".flac", ".mp4", ".aac"];
    const chosenExt = validExts.includes(ext) ? (ext === ".aac" ? ".m4a" : ext) : ".webm";
    cb(null, `${Date.now()}-${Math.round(Math.random() * 1e5)}${chosenExt}`);
  },
});

const upload = multer({
  storage,
  limits: { fileSize: 25 * 1024 * 1024 }, // 25MB أقصى حد مسموح به في Whisper
});

app.use(express.json());
app.use(express.static(path.join(__dirname, "public")));

// دالة للحصول على عميل الذكاء الاصطناعي (Groq المجاني أو OpenAI)
function getAIClient(customKey) {
  // المفتاح إما مرسل من الواجهة أو من المتغيرات البيئية
  const key = customKey || process.env.GROQ_API_KEY || process.env.OPENAI_API_KEY;
  if (!key) {
    throw new Error("مفتاح الذكاء الاصطناعي غير متوفر. يرجى إضافة مفتاح Groq المجاني (يبدأ بـ gsk_) أو مفتاح OpenAI في ملف .env أو من خلال الإعدادات في الصفحة.");
  }

  // التعرف على المزود: Groq يبدأ بـ gsk_ أو إذا تم ضبط GROQ_API_KEY في .env
  const isGroq = key.startsWith("gsk_") || (Boolean(process.env.GROQ_API_KEY) && !key.startsWith("sk-"));

  if (isGroq) {
    const groqKey = key.startsWith("gsk_") ? key : process.env.GROQ_API_KEY;
    return {
      provider: "groq",
      client: new OpenAI({
        apiKey: groqKey,
        baseURL: "https://api.groq.com/openai/v1",
      }),
      audioModel: "whisper-large-v3-turbo",
      chatModel: "llama-3.3-70b-versatile",
    };
  }

  // الافتراضي هو OpenAI
  return {
    provider: "openai",
    client: new OpenAI({ apiKey: key }),
    audioModel: "whisper-1",
    chatModel: "gpt-4o-mini",
  };
}

// دالة للحصول على عميل Anthropic (اختياري)
function getAnthropicClient(customKey) {
  const key = customKey || process.env.ANTHROPIC_API_KEY;
  if (!key) {
    throw new Error("مفتاح Anthropic غير متوفر.");
  }
  return new Anthropic({ apiKey: key });
}

// إزالة تكرار الكلمات المتتالية (هلوسة Whisper في حال وجود صمت أو ضوضاء)
function collapseRepeats(text) {
  if (!text) return "";
  let out = text;
  for (let i = 0; i < 3; i++) {
    out = out.replace(/(\S+(?:\s+\S+){0,4}?)(?:\s+\1){2,}/gu, "$1");
  }
  return out.replace(/\s{2,}/g, " ").trim();
}

// فحص حالة المفاتيح
app.get("/api/config", (req, res) => {
  const groqSet = Boolean(process.env.GROQ_API_KEY && process.env.GROQ_API_KEY.trim());
  const openaiSet = Boolean(process.env.OPENAI_API_KEY && process.env.OPENAI_API_KEY.trim());
  const anthropicSet = Boolean(process.env.ANTHROPIC_API_KEY && process.env.ANTHROPIC_API_KEY.trim());
  res.json({
    groqConfigured: groqSet,
    openaiConfigured: openaiSet,
    anthropicConfigured: anthropicSet,
    configured: groqSet || openaiSet,
  });
});

// 1) تفريغ الصوت إلى نص (مع خيار الترجمة الفورية)
app.post("/transcribe", upload.single("audio"), async (req, res) => {
  let filePath = req.file?.path;
  try {
    if (!req.file || req.file.size < 500) {
      return res.status(400).json({ error: "الملف الصوتي قصير جدًا أو فارغ، يرجى المحاولة مرة أخرى." });
    }

    const customKey = req.headers["x-api-key"] || req.headers["x-openai-key"] || req.body.apiKey;
    const { client, audioModel, provider } = getAIClient(customKey);

    let audioLang = (req.body.audioLanguage || "").trim(); // e.g. "ar_dialect", "ar_msa", "en", or "auto"
    const audioLanguageAliases = { dialect: "ar_dialect", msa: "ar_msa" };
    audioLang = audioLanguageAliases[audioLang.toLowerCase()] || audioLang;
    const targetLang = (req.body.targetLanguage || "").trim(); // e.g. "en", "ar_msa", "none"

    const transcriptionParams = {
      file: fs.createReadStream(filePath),
      model: audioModel,
      temperature: 0,
    };

    if (audioLang && audioLang !== "auto") {
      if (audioLang.startsWith("ar")) {
        // Whisper يقبل فقط "ar" كرمز لغة — نحوّل ar_dialect و ar_msa إلى ar
        transcriptionParams.language = "ar";
        if (audioLang === "ar_msa") {
          transcriptionParams.prompt = "تسجيل باللغة العربية الفصحى. نصوص بليغة وكتابة فصيحة مضبوطة إملائياً ونحوياً.";
        } else {
          transcriptionParams.prompt = "تسجيل صوتي بالعامية المصرية والعربية الدارجة. كلام عفوي وتلقائي بالعامية.";
        }
      } else {
        transcriptionParams.language = audioLang;
      }
    }

    console.log("Transcription params:", { language: transcriptionParams.language, model: transcriptionParams.model, audioLang });

    // استدعاء النموذج للتفريغ الصوتي (Whisper على Groq أو OpenAI)
    const transcription = await client.audio.transcriptions.create(transcriptionParams);
    const rawText = transcription.text || "";
    const cleanText = collapseRepeats(rawText);

    let translation = "";
    // إذا طُلب ترجمة نصية للغة معينة
    if (targetLang && targetLang !== "none" && cleanText.trim()) {
      try {
        translation = await translateWithAI(cleanText, targetLang, audioLang, customKey);
      } catch (transErr) {
        console.warn("Translation error:", transErr.message);
        translation = `(تعذر إتمام الترجمة تلقائيًا: ${transErr.message})`;
      }
    }

    res.json({
      text: cleanText,
      translation,
      cleaned: cleanText.length < rawText.length * 0.8,
      provider,
    });
  } catch (err) {
    console.error("Transcribe error:", err);
    res.status(500).json({
      error: err.message || "فشل تفريغ الصوت، يرجى التأكد من مفتاح الـ API وصلاحية الملف.",
    });
  } finally {
    if (filePath && fs.existsSync(filePath)) {
      fs.unlink(filePath, () => {});
    }
  }
});

// دالة مساعدة لترجمة النص باستخدام الذكاء الاصطناعي
async function translateWithAI(text, targetLang, sourceLang = "auto", customKey = null) {
  const targetNames = {
    en: "English",
    ar: "Modern Standard Arabic (اللغة العربية الفصحى)",
    ar_msa: "Modern Standard Arabic (اللغة العربية الفصحى السليمة الخالية من أي ألفاظ عامية)",
    ar_dialect: "Egyptian Arabic Dialect (اللهجة المصرية العامية الدارجة اليومية الطبيعية)",
    fr: "French (Français)",
    de: "German (Deutsch)",
    es: "Spanish (Español)",
    tr: "Turkish (Türkçe)",
    it: "Italian (Italiano)",
    ru: "Russian (Русский)",
    zh: "Simplified Chinese",
  };

  const targetName = targetNames[targetLang] || targetLang;
  const prompt = `Translate or convert the following text accurately into ${targetName}.
Source language context: ${sourceLang === "auto" ? "Detect language automatically" : sourceLang}.
Guidelines:
- If translating from another language to Arabic, or converting between colloquial dialect and Modern Standard Arabic (الفصحى):
  - For Modern Standard Arabic (ar_msa): Use clean, grammatically correct, eloquent Arabic without slang.
  - For Egyptian Colloquial (ar_dialect): Use smooth, natural, everyday Egyptian dialect as spoken in Egypt.
- Maintain original meaning, tone, and context faithfully.
- Return ONLY the translation or converted text, with no explanation, intro, quotes, or markdown wrappers.

Text:
${text}`;

  try {
    const { client, chatModel } = getAIClient(customKey);
    const resp = await client.chat.completions.create({
      model: chatModel,
      messages: [{ role: "user", content: prompt }],
      temperature: 0.3,
    });
    return resp.choices[0]?.message?.content?.trim() || "";
  } catch (err) {
    // محاولة بديلة مع Anthropic إذا كان المفتاح متوفراً
    if (process.env.ANTHROPIC_API_KEY && !customKey) {
      const anthropic = getAnthropicClient();
      const msg = await anthropic.messages.create({
        model: "claude-3-5-haiku-20241022",
        max_tokens: 1500,
        messages: [{ role: "user", content: prompt }],
      });
      return msg.content[0]?.text?.trim() || "";
    }
    throw err;
  }
}

// 2) ترجمة نص مباشر (دون إعادة رفع الصوت)
app.post("/translate", async (req, res) => {
  const { text, targetLang, sourceLang } = req.body;
  const customKey = req.headers["x-api-key"] || req.headers["x-openai-key"] || req.body.apiKey;

  if (!text || !text.trim()) {
    return res.status(400).json({ error: "النص المطلوب ترجمته فارغ." });
  }

  try {
    const translation = await translateWithAI(text, targetLang || "en", sourceLang || "auto", customKey);
    res.json({ translation });
  } catch (err) {
    console.error("Translate error:", err);
    res.status(500).json({ error: err.message || "فشلت عملية الترجمة." });
  }
});

// 3) تحويل النص لبوست سوشيال ميديا
app.post("/rewrite", async (req, res) => {
  const { text, platform } = req.body;
  const customKey = req.headers["x-api-key"] || req.headers["x-openai-key"] || req.body.apiKey;

  if (!text || !text.trim()) {
    return res.status(400).json({ error: "النص فارغ." });
  }

  const prompt = `حوّل الكلام التالي لبوست جذاب ومناسب لمنصة ${platform || "فيسبوك"}.
حافظ على المعنى الأصلي والنبرة، وحسّن الصياغة واستخدم تنسيق مناسب للمنصة (مثل الهاشتاجات أو النقاط لو ملائم).
اكتب نص البوست فقط بدون أي مقدمات أو شروحات:

${text}`;

  try {
    const { client, chatModel } = getAIClient(customKey);
    const resp = await client.chat.completions.create({
      model: chatModel,
      messages: [{ role: "user", content: prompt }],
      temperature: 0.7,
    });
    return res.json({ post: resp.choices[0]?.message?.content?.trim() || "" });
  } catch (err) {
    if (process.env.ANTHROPIC_API_KEY && !customKey) {
      try {
        const anthropic = getAnthropicClient();
        const msg = await anthropic.messages.create({
          model: "claude-3-5-haiku-20241022",
          max_tokens: 1200,
          messages: [{ role: "user", content: prompt }],
        });
        return res.json({ post: msg.content[0]?.text?.trim() || "" });
      } catch (anthropicErr) {
        console.error("Anthropic rewrite error:", anthropicErr);
      }
    }
    console.error("Rewrite error:", err);
    res.status(500).json({ error: err.message || "فشل توليد البوست." });
  }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`Server running on http://localhost:${PORT}`);
});
