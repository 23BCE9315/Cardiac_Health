"""
rag/rag_pipeline.py — RAG Pipeline (Grok-powered)
===================================================
• Retrieves relevant medical context using FAISS or TF-IDF
• Passes context + query to Grok API for high-quality answers
• Falls back to extractive method if no API key
"""

import os, re, logging, warnings
warnings.filterwarnings("ignore")

log = logging.getLogger(__name__)

RAG_DIR   = os.path.dirname(os.path.abspath(__file__))
DOC_DIR   = os.path.join(RAG_DIR, "documents")
IDX_PATH  = os.path.join(RAG_DIR, "faiss.index")
META_PATH = os.path.join(RAG_DIR, "chunks.pkl")

BUILTIN_MEDICAL_KB = [
    "Heart disease refers to several types of heart conditions. The most common type is coronary artery disease (CAD), which can cause heart attack. Symptoms include chest pain or discomfort, shortness of breath, pain or discomfort in arms or shoulder, and lightheadedness or nausea.",
    "Risk factors for heart disease include high blood pressure, high blood cholesterol, smoking, diabetes, obesity or overweight, unhealthy diet, physical inactivity, and excessive alcohol use. Age and family history also play a role.",
    "The 13 features used in heart disease prediction are: age, sex, chest pain type (cp), resting blood pressure (trestbps), serum cholesterol (chol), fasting blood sugar (fbs), resting ECG results (restecg), max heart rate (thalach), exercise-induced angina (exang), ST depression (oldpeak), slope of peak exercise ST segment, number of major vessels (ca), and thalassemia type (thal).",
    "Prevention of heart disease includes eating a healthy diet low in saturated fats and sodium, exercising regularly at least 150 minutes per week, not smoking, maintaining a healthy weight, managing stress, and getting regular health screenings.",
    "Treatment for heart disease can include lifestyle changes, medications such as statins, beta-blockers, ACE inhibitors, medical procedures like angioplasty or bypass surgery, and cardiac rehabilitation programs.",
    "Cholesterol is a waxy substance found in blood. LDL bad cholesterol builds up in artery walls. HDL good cholesterol carries cholesterol to the liver. Normal total cholesterol is below 200 mg/dL.",
    "Normal cholesterol: Total below 200 mg/dL desirable. LDL below 100 mg/dL optimal. HDL above 60 mg/dL protective. Triglycerides below 150 mg/dL normal.",
    "Blood pressure normal is less than 120/80 mmHg. Elevated is 120-129 systolic. Hypertension stage 1 is 130-139/80-89. Stage 2 is 140 or higher over 90 or higher.",
    "Hypertension is often called the silent killer because it has no symptoms. It forces the heart to work harder, increasing risk of heart disease, stroke, and kidney failure.",
    "Diabetes is a chronic condition where the body cannot process blood glucose properly. Type 2 diabetes is most common and related to insulin resistance. Fasting blood sugar above 126 mg/dL indicates diabetes.",
    "People with diabetes are 2 to 4 times more likely to develop heart disease. Managing blood sugar, blood pressure, and cholesterol is critical for diabetics.",
    "An ECG records electrical activity of the heart. ST-segment changes during exercise testing are important indicators of coronary artery disease.",
    "Exercise-induced angina is chest pain during exercise — a significant warning sign. Maximum heart rate and ST depression are key diagnostic indicators.",
    "Thalassemia is a blood disorder causing abnormal hemoglobin, leading to anemia and cardiovascular stress.",
    "BMI under 18.5 is underweight, 18.5 to 24.9 is normal, 25 to 29.9 is overweight, 30 or above is obese. Obesity raises risk of heart disease, diabetes, and hypertension.",
    "The Mediterranean diet reduces heart disease risk: vegetables, fruits, whole grains, legumes, fish, and olive oil while limiting red meat and processed foods.",
    "Chronic stress raises cortisol and adrenaline, increasing heart rate and blood pressure. Meditation, yoga, and deep breathing help manage stress.",
    "Smoking damages blood vessels, reduces HDL cholesterol, increases blood pressure, and causes blood clots. Quitting smoking significantly reduces cardiovascular risk.",
    "Regular physical activity strengthens the heart, lowers blood pressure, raises HDL cholesterol, and reduces stress. Aim for 150 minutes of moderate aerobic activity per week.",
    "Aspirin therapy prevents blood platelets from clumping into clots. Recommended for high-risk heart patients only — always consult your doctor first.",
]

EMOJI_MAP = {
    "heart": "❤️", "blood pressure": "🩸", "cholesterol": "🧪",
    "diet": "🥗", "eat": "🥗", "food": "🥗", "exercise": "🏃",
    "walk": "🚶", "run": "🏃", "smoking": "🚭", "smoke": "🚭",
    "diabetes": "💉", "insulin": "💉", "stress": "🧘",
    "medication": "💊", "medicine": "💊", "drug": "💊",
    "doctor": "🩺", "hospital": "🏥", "symptom": "🔍",
    "pain": "🔍", "prevent": "🛡️", "weight": "⚖️",
    "sleep": "😴", "water": "💧", "fruit": "🍎",
}


class RAGPipeline:

    def __init__(self):
        self.chunks    = []
        self.index     = None
        self.embedder  = None
        self.use_faiss = False
        self.tfidf_mat = None
        self.tfidf_vec = None
        self._init_embedder()
        self._load_or_build_index()

    def _init_embedder(self):
        try:
            from sentence_transformers import SentenceTransformer
            self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
            self.use_faiss = True
            log.info("Embedder: all-MiniLM-L6-v2")
        except Exception as e:
            log.warning("sentence-transformers unavailable — TF-IDF fallback: %s", e)

    def _load_chunks(self):
        chunks = list(BUILTIN_MEDICAL_KB)
        os.makedirs(DOC_DIR, exist_ok=True)
        for fname in [f for f in os.listdir(DOC_DIR) if f.lower().endswith(".pdf")]:
            try:
                from pypdf import PdfReader
                reader = PdfReader(os.path.join(DOC_DIR, fname))
                text   = " ".join(p.extract_text() or "" for p in reader.pages)
                chunks.extend(self._split_text(text))
                log.info("Loaded PDF: %s", fname)
            except Exception as ex:
                log.warning("Could not read %s: %s", fname, ex)
        log.info("Total chunks: %d", len(chunks))
        return chunks

    @staticmethod
    def _split_text(text, chunk_size=300, overlap=50):
        words, result, i = text.split(), [], 0
        while i < len(words):
            chunk = " ".join(words[i: i + chunk_size])
            if len(chunk.strip()) > 20:
                result.append(chunk)
            i += chunk_size - overlap
        return result

    def _load_or_build_index(self):
        import pickle
        self.chunks = self._load_chunks()
        if self.use_faiss:
            try:
                import faiss, numpy as np
                if os.path.exists(IDX_PATH) and os.path.exists(META_PATH):
                    self.index = faiss.read_index(IDX_PATH)
                    with open(META_PATH, "rb") as f:
                        saved = pickle.load(f)
                    if len(saved) == self.index.ntotal:
                        self.chunks = saved
                        log.info("Loaded FAISS index (%d vectors)", self.index.ntotal)
                        return
                log.info("Building FAISS index...")
                emb = self.embedder.encode(self.chunks, show_progress_bar=False, convert_to_numpy=True)
                self.index = faiss.IndexFlatL2(emb.shape[1])
                self.index.add(emb.astype(np.float32))
                faiss.write_index(self.index, IDX_PATH)
                with open(META_PATH, "wb") as f:
                    pickle.dump(self.chunks, f)
                log.info("FAISS index built and saved")
                return
            except Exception as e:
                log.warning("FAISS unavailable — TF-IDF: %s", e)
                self.use_faiss = False
        self._build_tfidf()

    def _build_tfidf(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.tfidf_vec = TfidfVectorizer(stop_words="english", max_features=5000)
        self.tfidf_mat = self.tfidf_vec.fit_transform(self.chunks)
        log.info("TF-IDF index built (%d chunks)", len(self.chunks))

    def retrieve(self, query, top_k=2):
        if self.use_faiss and self.index is not None:
            import numpy as np
            q_emb = self.embedder.encode([query], convert_to_numpy=True)
            _, idx = self.index.search(q_emb.astype(np.float32), top_k)
            return [self.chunks[i] for i in idx[0] if i < len(self.chunks)]
        else:
            import numpy as np
            from sklearn.metrics.pairwise import cosine_similarity
            q_vec = self.tfidf_vec.transform([query])
            sims  = cosine_similarity(q_vec, self.tfidf_mat).flatten()
            return [self.chunks[i] for i in np.argsort(sims)[::-1][:top_k]]

    def _extractive_answer(self, context, query):
        sentences = re.split(r'(?<=[.!?])\s+', context)
        stop_words = {"what","is","are","the","a","an","how","does","do","in","for","to","with"}
        q_words = set(query.lower().split()) - stop_words
        scored  = [(sum(1 for w in s.lower().split() if w in q_words), s)
                   for s in sentences if len(s) > 20]
        if not scored:
            return f"❤️ Health Info:\n• ✅ {context[:200]}\n• 🩺 Always consult your doctor."
        scored.sort(key=lambda x: x[0], reverse=True)
        seen, unique = [], []
        for _, s in scored[:8]:
            key = re.sub(r'[^a-z0-9 ]', '', s.lower())
            if not any(key[:30] in re.sub(r'[^a-z0-9 ]','',p.lower()) for p in seen):
                seen.append(s); unique.append(s)
            if len(unique) == 4:
                break
        topic   = query.split()[0].capitalize()
        heading = f"❤️ {topic} Info:"
        bullets = [f"• {self._pick_emoji(s)} {s.strip().rstrip('.')}" for s in unique]
        bullets.append("• 🩺 Always consult your doctor for personal medical advice.")
        return heading + "\n" + "\n".join(bullets)

    @staticmethod
    def _pick_emoji(text):
        tl = text.lower()
        for kw, em in EMOJI_MAP.items():
            if kw in tl:
                return em
        return "✅"

    def answer(self, query):
        chunks  = self.retrieve(query, top_k=2)
        context = " ".join(chunks) if chunks else " ".join(BUILTIN_MEDICAL_KB[:3])
        # Grok is called from app.py — this is extractive fallback only
        answer  = self._extractive_answer(context, query)
        return {"answer": answer, "sources": chunks[:2], "method": "extractive"}


_pipeline = None

def get_pipeline():
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
    return _pipeline