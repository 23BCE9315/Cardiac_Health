# 🫀 CardioAI — AI Health Assistant with Voice & RAG

A full-stack web application that combines **heart disease prediction** (ML),
**voice interaction** (Web Speech API), and **RAG-based medical Q&A** (FAISS + FLAN-T5)
into a single, beautifully designed interface.

---

## 🗂 Project Structure

```
AI-Health-Assistant/
│
├── app.py               ← Flask backend (routes: /, /predict, /chat, /features, /health)
├── model.py             ← ML pipeline: train, evaluate, save, predict
├── heart.csv            ← Cleveland Heart Disease dataset (13 features)
├── requirements.txt     ← Python dependencies
│
├── models/
│   ├── heart_model.pkl  ← Best trained model (Random Forest, ~91.8% acc)
│   └── scaler.pkl       ← StandardScaler fitted on training data
│
├── templates/
│   └── index.html       ← Main UI (Chat + Heart Scan tabs)
│
├── public/
│   ├── style.css        ← Premium dark medical UI (Syne + DM Sans)
│   └── script.js        ← All frontend logic (voice, chat, prediction)
│
└── rag/
    ├── rag_pipeline.py  ← RAG system (embeddings + FAISS + FLAN-T5)
    └── documents/       ← Drop medical PDFs here for RAG indexing
```

---

## ⚙️ Setup & Installation

### 1. Clone / download the project

```bash
cd AI-Health-Assistant
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows
```

### 3. Install core dependencies

```bash
pip install -r requirements.txt
```

### 4. Install RAG extras (optional but recommended)

Enables semantic embeddings (sentence-transformers), FAISS vector search,
and FLAN-T5 answer generation. Falls back to TF-IDF + extractive answers if not installed.

```bash
pip install sentence-transformers faiss-cpu transformers torch pypdf
```

---

## 🚀 Running the App

### Step 1 — Train the ML model

```bash
python model.py
```

**What this does:**
- Loads `heart.csv`
- Preprocesses data (missing values, scaling)
- Trains 4 models: Naive Bayes, KNN, SVM, Random Forest
- Prints accuracy + classification report for each
- Saves the best model to `models/heart_model.pkl` and `models/scaler.pkl`

**Expected output:**
```
✅ Training complete — best model: Random Forest  (91.80%)
   Artifacts saved to: models/
```

### Step 2 — Start the Flask server

```bash
python app.py
```

Open your browser at: **http://localhost:5000**

---

## 🎯 Features & How to Use

### 💬 Medical Chat (RAG)

- Type any medical question in the chat box
- Press Enter or click the Send button
- The RAG engine retrieves relevant knowledge and generates an answer
- Click the 🔊 speaker icon to toggle voice responses (Text-to-Speech)

**Sample questions to try:**
- "What are the symptoms of heart disease?"
- "What is the normal cholesterol level?"
- "How does diabetes affect the heart?"
- "What do the 13 features in heart disease prediction mean?"
- "How can I prevent heart disease?"

### 🫀 Heart Scan (Prediction)

- Switch to the **Heart Scan** tab
- Enter your 13 clinical measurements
- Click **Run Analysis** → get prediction + confidence score + risk level
- Click **Load Sample Data** to populate with example values

**13 Input Features:**

| # | Feature   | Description                              | Range   |
|---|-----------|------------------------------------------|---------|
| 1 | age       | Age in years                             | 1–120   |
| 2 | sex       | Sex (1=male, 0=female)                   | 0–1     |
| 3 | cp        | Chest pain type (0–3)                    | 0–3     |
| 4 | trestbps  | Resting blood pressure (mmHg)            | 50–300  |
| 5 | chol      | Serum cholesterol (mg/dl)                | 100–700 |
| 6 | fbs       | Fasting blood sugar >120 (1=true)        | 0–1     |
| 7 | restecg   | Resting ECG results (0–2)                | 0–2     |
| 8 | thalach   | Maximum heart rate achieved              | 60–220  |
| 9 | exang     | Exercise-induced angina (1=yes)          | 0–1     |
|10 | oldpeak   | ST depression (exercise vs rest)         | 0–10    |
|11 | slope     | Slope of peak exercise ST (0–2)          | 0–2     |
|12 | ca        | Major vessels colored by fluoroscopy     | 0–3     |
|13 | thal      | Thal: 1=normal 2=fixed 3=reversable      | 0–3     |

### 🎤 Voice Features

**Voice Chat:**
1. Click the 🎤 microphone button in the chat input bar
2. Speak your medical question
3. The transcript is automatically sent and answered

**Voice Prediction:**
1. Click **🎤 Voice Input** on the Heart Scan tab
2. Dictate all 13 values in order, comma-separated:
   > *"63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1"*
3. The numbers are extracted, the form is filled, and analysis runs automatically

> **Note:** Voice requires Chrome, Edge, or Safari. Firefox does not support the Web Speech API.

---

## 🧠 RAG Pipeline Details

The RAG (Retrieval-Augmented Generation) system:

1. **Knowledge Base** — Built-in medical KB (20 chunks covering heart disease, cholesterol, BP, diabetes, ECG, thalassemia, diet, exercise, and more)
2. **PDF Loading** — Drop any medical PDF into `rag/documents/` and restart; they'll be indexed automatically
3. **Embeddings** — `all-MiniLM-L6-v2` via sentence-transformers (falls back to TF-IDF)
4. **Vector Search** — FAISS IndexFlatL2 (falls back to cosine similarity)
5. **Generation** — `google/flan-t5-small` (falls back to extractive best-sentence)

### Adding Medical PDFs

```bash
# Copy your PDFs into the documents folder
cp my_cardiology_guide.pdf rag/documents/

# Restart the app — it will auto-index on startup
python app.py
```

---

## 🤖 ML Models Comparison

| Model         | Test Accuracy | CV (5-fold) |
|---------------|--------------|-------------|
| Naive Bayes   | ~78.7%       | ~83.5%      |
| KNN           | ~83.6%       | ~75.7%      |
| SVM           | ~85.3%       | ~85.2%      |
| Random Forest | ~91.8% ✅    | ~87.6%      |

**Random Forest** is selected automatically as the best model.

---

## 🌐 API Endpoints

| Method | Route        | Description                                  |
|--------|-------------|----------------------------------------------|
| GET    | `/`          | Serve the frontend UI                        |
| POST   | `/predict`   | Heart disease prediction (JSON body)         |
| POST   | `/chat`      | RAG medical chatbot (JSON body)              |
| GET    | `/features`  | Feature metadata for UI                      |
| GET    | `/health`    | Readiness check                              |

### `/predict` — Request & Response

```json
// Request
{ "features": [63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1] }

// Response
{
  "prediction":  1,
  "label":       "Heart Disease Detected",
  "confidence":  0.675,
  "risk_level":  "Moderate",
  "advice":      "⚠️ Our model suggests possible heart disease indicators...",
  "warnings":    []
}
```

### `/chat` — Request & Response

```json
// Request
{ "message": "What is normal blood pressure?" }

// Response
{
  "reply":   "Normal blood pressure is less than 120/80 mmHg...",
  "sources": ["Blood pressure is measured in millimeters of mercury..."],
  "method":  "extractive"
}
```

---

## 🐛 Debugging Tips

### "Model not found" error
```bash
# Train the model first
python model.py
```

### Voice input not working
- Use **Chrome** or **Edge** (Firefox does not support Web Speech API)
- Allow microphone permissions when prompted
- Ensure you're on `http://localhost:5000` (not file://)

### RAG answers seem generic
- Install the full RAG stack: `pip install sentence-transformers faiss-cpu transformers torch`
- Add domain-specific PDFs to `rag/documents/`
- Check logs — the method field in `/chat` response shows `flan-t5`, `extractive`, or `tfidf`

### Flask won't start (port in use)
```bash
# Change port in app.py last line:
app.run(debug=True, host="0.0.0.0", port=5001)
```

### Import errors
```bash
pip install --upgrade flask scikit-learn pandas numpy joblib
```

---

## 📦 All Dependencies

### Core (required)
```
flask>=3.0.0
flask-cors>=4.0.0
scikit-learn>=1.4.0
pandas>=2.1.0
numpy>=1.26.0
joblib>=1.3.0
```

### RAG (optional)
```
sentence-transformers>=2.6.0
faiss-cpu>=1.8.0
transformers>=4.40.0
torch>=2.2.0
pypdf>=4.2.0
```

---

## ⚠️ Disclaimer

This application is for **educational and research purposes only**.
It is **not a medical device** and should not be used for clinical diagnosis.
Always consult a qualified healthcare professional for medical advice.

---

*Built with Python, Flask, scikit-learn, FAISS, FLAN-T5, and Web Speech API.*
