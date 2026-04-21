"""
app.py — CardioAI v9 (ChatGPT-Style Conversational AI)
=======================================================
What's new:
  • Full conversation history — AI remembers context across messages
  • Streaming responses — text appears word by word like ChatGPT
  • Smarter system prompt — handles follow-up, vague, and contextual questions
  • All previous functionality preserved (predict, voice, KB fallback)
  • Groq llama3-8b-8192 (free, fast)
"""

import os, re, sys, logging, json
from flask import Flask, request, jsonify, render_template, Response, stream_with_context

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  [%(name)s]  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("app")

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "public"),
    static_url_path="/public",
)

# ── Lazy loaders ──────────────────────────────────────────────────────────────
_model = _scaler = _rag = None

def get_model():
    global _model, _scaler
    if _model is None:
        from model import load_artifacts
        _model, _scaler = load_artifacts()
    return _model, _scaler

def get_rag():
    global _rag
    if _rag is None:
        rag_dir = os.path.join(BASE_DIR, "rag")
        if rag_dir not in sys.path:
            sys.path.insert(0, rag_dir)
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "rag_pipeline", os.path.join(rag_dir, "rag_pipeline.py")
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _rag = mod.get_pipeline()
    return _rag


# ════════════════════════════════════════════════════════════════════════════
#  SYSTEM PROMPT — conversational, cardiac-focused, ChatGPT-style
# ════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are CardioAI, an expert cardiac health assistant. You have deep knowledge of all aspects of cardiology and heart health. You behave like ChatGPT — conversational, intelligent, and helpful.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONVERSATION RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Remember everything the user said in this conversation
• When the user says "it", "this", "that", "about it", "more", "explain further" — refer back to what was just discussed
• If the user seems worried or describes symptoms, be empathetic first, then informative
• Never start your reply with "I" — vary your opening
• Be conversational and warm, not robotic
• Never say "As an AI" or "I cannot provide medical advice" as your opening — just answer helpfully

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EMERGENCY RULE — HIGHEST PRIORITY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
If the user describes: chest pain, heart attack, not breathing, unconscious, emergency
→ Start with: "🚨 This sounds like a medical emergency."
→ Tell them to call emergency services IMMEDIATELY
→ Then give actionable steps
→ Keep it short and urgent — do NOT write long paragraphs in emergencies

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RESPONSE FORMAT — VARY BY QUESTION TYPE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

For LISTS (symptoms, causes, tips, steps):
🩺 Title:
• Point one
• Point two
• Point three
• Point four
💡 Tip: one helpful sentence

For EXPLANATIONS (what is, how does, why):
Write 2-3 short clear paragraphs. No bullet points needed.
End with: 💡 Tip: ...

For CONVERSATIONAL follow-ups ("is it serious?", "should I worry?", "what about X?"):
Respond naturally like a knowledgeable friend. 1-3 sentences.
Reference what was said earlier in the conversation.

For DOCTOR/SPECIALIST questions:
Be specific — name the specialist, explain what they do, when to see them.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONTENT RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Answer ANY cardiac health question — no topic is off-limits
• Topics: symptoms, causes, tests (ECG, echo, angiogram), procedures (angioplasty, bypass, pacemaker), medications (statins, beta-blockers, aspirin), diet, exercise, lifestyle, risk factors, specialists, follow-up care, rehabilitation, genetics, age-related concerns, women's heart health, diabetes connection, stress, sleep, smoking, alcohol
• If a question is outside cardiac health, politely redirect: "That's outside my cardiac specialty — for that, I'd recommend consulting a general physician."
• Never include: dataset features, ML terminology, mmHg/mg/dL units unless asked, numeric ranges unless directly asked
• Keep language simple, warm, and clear

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EXAMPLES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

User: I have chest pain
CardioAI: 🚨 This sounds like a medical emergency. Please call emergency services right now or have someone take you to the ER immediately. While waiting:
• Stay still, sit down, and do not exert yourself
• Loosen any tight clothing around your chest
• Chew one aspirin if you have it and are not allergic
• Tell someone nearby what is happening
💡 Do not drive yourself — call for help now.

User: what is a cardiologist
CardioAI: 🩺 What Is a Cardiologist:
• A cardiologist is a doctor who specialises in the heart and blood vessels
• They diagnose and treat conditions like high BP, heart disease, and arrhythmia
• Interventional cardiologists perform procedures like stents and angioplasty
• Electrophysiologists focus specifically on heart rhythm disorders
💡 Tip: Your GP can write you a referral to a cardiologist based on your symptoms.

User: I was told I have high blood pressure. What should I do?
CardioAI: That is an important diagnosis to take seriously — the good news is high blood pressure is very manageable.

The first steps are lifestyle changes: reducing salt in your diet, getting regular exercise, and managing stress. These alone can make a significant difference.

Your doctor will likely monitor you for a few weeks. If your readings stay high, they may prescribe medication — ACE inhibitors or diuretics are common first choices.

💡 Tip: Get a home blood pressure monitor so you can track it daily and share the readings with your doctor."""


# ════════════════════════════════════════════════════════════════════════════
#  GROQ API — streaming + conversation history
# ════════════════════════════════════════════════════════════════════════════

def call_groq_stream(messages: list):
    """
    Stream Groq response token by token.
    messages = full conversation history in OpenAI format.
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not set")

    from groq import Groq
    client = Groq(api_key=api_key)

    stream = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=messages,
        max_tokens=400,
        temperature=0.4,
        top_p=0.9,
        frequency_penalty=0.4,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta
        if hasattr(delta, "content") and delta.content:
            yield delta.content


def call_groq_full(messages: list) -> str:
    """Non-streaming version for fallback."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not set")

    from groq import Groq
    client = Groq(api_key=api_key)

    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=messages,
        max_tokens=400,
        temperature=0.4,
        top_p=0.9,
        frequency_penalty=0.4,
        stream=False,
    )
    return response.choices[0].message.content.strip()


# ════════════════════════════════════════════════════════════════════════════
#  MEDICAL KNOWLEDGE BASE — instant offline fallback for any cardiac topic
# ════════════════════════════════════════════════════════════════════════════

MEDICAL_KB = {
    "heart attack emergency": {
        "headline": "🚨 Heart Attack – Immediate Steps",
        "bullets": ["Call emergency services immediately — every minute matters",
                    "Sit down, stay calm, and stop all physical activity",
                    "Chew one aspirin slowly if you are not allergic to it",
                    "Loosen tight clothing around the chest and neck"],
        "tip": "If the person loses consciousness, start CPR immediately.",
    },
    "cpr": {
        "headline": "🚨 CPR – Step-by-Step",
        "bullets": ["Call emergency services first before starting CPR",
                    "Place heel of hand on centre of chest, push hard and fast",
                    "Give 30 compressions then 2 rescue breaths",
                    "Continue until help arrives or the person recovers"],
        "tip": "Hands-only CPR (no rescue breaths) is also effective and easier.",
    },
    "which doctor": {
        "headline": "👨‍⚕️ Which Doctor to Consult",
        "bullets": ["Start with your GP for routine heart concerns and referrals",
                    "A Cardiologist specialises in all heart and vessel conditions",
                    "An Interventional Cardiologist performs stents and angioplasty",
                    "A Cardiac Surgeon handles bypass surgery and major procedures"],
        "tip": "Your GP will refer you to the right specialist based on your symptoms.",
    },
    "cardiologist": {
        "headline": "👨‍⚕️ About Cardiologists",
        "bullets": ["A cardiologist specialises in the heart and blood vessels",
                    "They diagnose and treat heart disease, arrhythmia, and heart failure",
                    "Interventional cardiologists perform procedures like angioplasty",
                    "Electrophysiologists focus on heart rhythm disorders"],
        "tip": "Your GP can refer you — bring a list of your symptoms and medications.",
    },
    "blood pressure symptoms": {
        "headline": "🩺 Symptoms of High Blood Pressure",
        "bullets": ["Often no symptoms — called the silent killer for this reason",
                    "Persistent headaches, especially at the back of the head",
                    "Dizziness or lightheadedness when standing",
                    "Blurred vision or nosebleeds in more severe cases"],
        "tip": "Get regular check-ups even when you feel completely fine.",
    },
    "blood pressure causes": {
        "headline": "⚠️ Causes of High Blood Pressure",
        "bullets": ["Too much salt raises fluid retention and increases pressure",
                    "Being overweight forces the heart to work much harder",
                    "Physical inactivity weakens the cardiovascular system",
                    "Chronic stress, smoking, and excess alcohol are major triggers"],
        "tip": "Reducing salt and exercising daily makes a significant difference.",
    },
    "blood pressure treatment": {
        "headline": "💊 Treating High Blood Pressure",
        "bullets": ["Lifestyle changes — diet, exercise, reduced salt — come first",
                    "Doctors prescribe ACE inhibitors, diuretics, or beta-blockers",
                    "Monitor blood pressure regularly at home with a digital cuff",
                    "Stress reduction via yoga or meditation supports treatment"],
        "tip": "Never stop blood pressure medication without your doctor's approval.",
    },
    "blood pressure diet": {
        "headline": "🥗 Diet for High Blood Pressure",
        "bullets": ["Eat potassium-rich foods — bananas, spinach, sweet potatoes",
                    "Choose whole grains over white rice and refined flour",
                    "Cut sodium by avoiding salty snacks and canned food",
                    "Include oily fish like salmon for heart-healthy omega-3s"],
        "tip": "The DASH diet is clinically proven to lower blood pressure.",
    },
    "heart disease symptoms": {
        "headline": "🩺 Symptoms of Heart Disease",
        "bullets": ["Chest pain or tightness, especially during physical activity",
                    "Shortness of breath even with light daily activity",
                    "Pain spreading to left arm, jaw, neck, or upper back",
                    "Unusual fatigue, cold sweats, or sudden dizziness"],
        "tip": "Sudden onset of these symptoms needs emergency care immediately.",
    },
    "heart disease causes": {
        "headline": "⚠️ Causes of Heart Disease",
        "bullets": ["High blood pressure gradually damages artery walls",
                    "High cholesterol causes plaque to build up in arteries",
                    "Smoking is one of the strongest known risk factors",
                    "Diabetes and obesity both raise heart disease risk significantly"],
        "tip": "Family history of heart disease means earlier screening is vital.",
    },
    "heart disease prevention": {
        "headline": "✅ Preventing Heart Disease",
        "bullets": ["Eat a balanced diet low in saturated fats and sugar",
                    "Exercise regularly — at least 150 minutes per week",
                    "Quit smoking and limit alcohol consumption",
                    "Manage blood pressure, cholesterol, and blood sugar proactively"],
        "tip": "Prevention started early is far more effective than treatment later.",
    },
    "heart failure": {
        "headline": "🩺 What Is Heart Failure",
        "bullets": ["Heart failure means the heart cannot pump blood efficiently enough",
                    "It does not mean the heart has stopped — just works less well",
                    "Causes include coronary artery disease and high blood pressure",
                    "Symptoms include breathlessness, leg swelling, and fatigue"],
        "tip": "Heart failure is manageable with medication and lifestyle changes.",
    },
    "cholesterol symptoms": {
        "headline": "🩺 Signs of High Cholesterol",
        "bullets": ["High cholesterol usually has no visible symptoms at all",
                    "Fatty yellow deposits can form around eyes in rare cases",
                    "Chest pain may occur if arteries become severely narrowed",
                    "A blood test is the only reliable way to detect it"],
        "tip": "Get a cholesterol check every year after the age of 30.",
    },
    "cholesterol diet": {
        "headline": "🥗 Diet to Lower Cholesterol",
        "bullets": ["Eat oats, beans, and lentils which reduce LDL cholesterol",
                    "Choose olive oil and avocado instead of butter and saturated fats",
                    "Include oily fish like salmon or mackerel twice a week",
                    "Avoid fried food, fast food, and baked goods with trans fats"],
        "tip": "Nuts and seeds provide plant sterols that lower cholesterol naturally.",
    },
    "ecg test": {
        "headline": "🔬 ECG Test Explained",
        "bullets": ["An ECG records the electrical activity of your heart",
                    "Small electrodes are placed on chest, arms, and legs",
                    "The test is painless and takes about 5 to 10 minutes",
                    "It detects arrhythmia, heart attacks, and other conditions"],
        "tip": "An ECG is usually the first test ordered for heart concerns.",
    },
    "angioplasty": {
        "headline": "💊 Angioplasty Explained",
        "bullets": ["Angioplasty opens narrowed or blocked coronary arteries",
                    "A tiny balloon is inserted via catheter and inflated",
                    "A stent is usually placed to keep the artery open",
                    "Most patients go home within one to two days"],
        "tip": "Angioplasty restores blood flow without open-heart surgery.",
    },
    "pacemaker": {
        "headline": "💊 Pacemaker Explained",
        "bullets": ["A pacemaker is a small device implanted under the skin",
                    "It sends electrical pulses to regulate a slow heartbeat",
                    "Modern pacemakers are about the size of a matchbox",
                    "Most activities are safe — normal life continues with one"],
        "tip": "Avoid strong magnetic fields unless your pacemaker is MRI-safe.",
    },
    "arrhythmia": {
        "headline": "🩺 What Is Arrhythmia",
        "bullets": ["Arrhythmia means the heart beats too fast, slow, or irregularly",
                    "It is caused by problems with the heart's electrical signals",
                    "Some are harmless while others can be life-threatening",
                    "Palpitations, dizziness, and fainting are common signs"],
        "tip": "An ECG is the most common test used to detect arrhythmia.",
    },
    "atrial fibrillation": {
        "headline": "🩺 What Is Atrial Fibrillation (AFib)",
        "bullets": ["AFib is the most common type of irregular heart rhythm",
                    "The upper chambers beat chaotically instead of in rhythm",
                    "It raises stroke and heart failure risk significantly",
                    "Symptoms include palpitations, fatigue, and breathlessness"],
        "tip": "AFib is manageable with medications, cardioversion, or ablation.",
    },
    "statins": {
        "headline": "💊 Statins for Heart Health",
        "bullets": ["Statins lower LDL cholesterol by reducing its liver production",
                    "They reduce the risk of heart attack and stroke significantly",
                    "Common ones include atorvastatin, rosuvastatin, simvastatin",
                    "Side effects are rare but can include muscle aches"],
        "tip": "Take statins at the same time each day — never stop without advice.",
    },
    "stress heart": {
        "headline": "✅ Stress & Heart Health",
        "bullets": ["Chronic stress raises blood pressure and heart disease risk",
                    "It triggers release of cortisol which harms blood vessels",
                    "Deep breathing, meditation, and exercise lower stress effectively",
                    "Adequate sleep of 7 to 8 hours is essential for heart recovery"],
        "tip": "Even 10 minutes of daily relaxation significantly benefits the heart.",
    },
    "exercise heart": {
        "headline": "🏃 Exercise for Heart Health",
        "bullets": ["Aim for 30 minutes of brisk walking at least 5 days a week",
                    "Include light strength training twice a week",
                    "Start slowly if new to exercise and build up gradually",
                    "Avoid long sitting — short walks every hour help circulation"],
        "tip": "Even 10 minutes of movement daily significantly benefits heart health.",
    },
    "heart healthy diet": {
        "headline": "🥗 Heart-Healthy Foods",
        "bullets": ["Eat oily fish like salmon or sardines twice a week",
                    "Choose whole grains over white bread and refined carbs",
                    "Include nuts, seeds, and avocado for healthy fats",
                    "Reduce salt, red meat, fried food, and processed snacks"],
        "tip": "A Mediterranean-style diet is proven to protect the heart.",
    },
    "smoking heart": {
        "headline": "✅ Smoking & Heart Health",
        "bullets": ["Smoking damages blood vessel walls and speeds artery blockage",
                    "It reduces oxygen in blood, forcing the heart to work harder",
                    "Passive smoking also raises heart disease risk",
                    "Heart attack risk drops noticeably within one year of quitting"],
        "tip": "Nicotine replacement therapy greatly improves quit success rates.",
    },
    "diabetes heart": {
        "headline": "🩺 Diabetes & Heart Health",
        "bullets": ["Diabetes doubles the risk of developing heart disease",
                    "High blood sugar damages blood vessels and nerves over time",
                    "Managing blood sugar, BP, and cholesterol is essential",
                    "Lifestyle changes can reverse early-stage Type 2 diabetes"],
        "tip": "People with diabetes should have annual heart health check-ups.",
    },
    "cardiac rehabilitation": {
        "headline": "🩺 Cardiac Rehabilitation",
        "bullets": ["Cardiac rehab is supervised recovery after heart attack or surgery",
                    "It combines exercise training, education, and support",
                    "Reduces the risk of a second heart attack significantly",
                    "Sessions run for 8 to 12 weeks at a hospital or clinic"],
        "tip": "Cardiac rehab is one of the most effective recovery tools available.",
    },
    # ── Blood Pressure Definition (what is / general) ─────────────────────
    "blood pressure definition": {
        "headline": "🩺 What Is High Blood Pressure",
        "bullets": [
            "Blood pressure is the force of blood pushing against artery walls",
            "High blood pressure means this force is consistently too strong",
            "It strains the heart, damages arteries, and can lead to heart disease",
            "It is called the silent killer because most people feel no symptoms",
        ],
        "tip": "You can check your blood pressure at home with a digital monitor.",
    },
    "cholesterol definition": {
        "headline": "🩺 What Is Cholesterol",
        "bullets": [
            "Cholesterol is a fatty substance found naturally in the blood",
            "LDL is the bad type that builds up in artery walls causing blockage",
            "HDL is the good type that clears cholesterol from the bloodstream",
            "High LDL cholesterol is a major risk factor for heart disease",
        ],
        "tip": "A simple blood test called a lipid panel measures your cholesterol levels.",
    },
    "cholesterol causes": {
        "headline": "⚠️ Causes of High Cholesterol",
        "bullets": [
            "Eating saturated and trans fats raises LDL bad cholesterol",
            "Physical inactivity lowers HDL good cholesterol over time",
            "Genetics can cause high cholesterol even with a healthy diet",
            "Obesity, diabetes, and thyroid problems also raise cholesterol",
        ],
        "tip": "Even slim people can have high cholesterol due to family history.",
    },
    "cholesterol treatment": {
        "headline": "💊 Treating High Cholesterol",
        "bullets": [
            "Diet changes — less saturated fat and more fibre — are the foundation",
            "Statins are the most prescribed medication for high LDL cholesterol",
            "Regular exercise raises HDL good cholesterol significantly",
            "Your doctor will retest cholesterol after 3 months to see progress",
        ],
        "tip": "Lifestyle changes alone can lower cholesterol by up to 20 percent.",
    },
    "heart disease definition": {
        "headline": "🩺 What Is Heart Disease",
        "bullets": [
            "Heart disease refers to conditions that affect the heart or blood vessels",
            "Coronary artery disease is the most common type — caused by artery blockage",
            "It can lead to heart attacks, heart failure, and arrhythmia",
            "It is the leading cause of death globally but largely preventable",
        ],
        "tip": "Knowing your blood pressure and cholesterol is the first step to prevention.",
    },
    "bypass surgery": {
        "headline": "💊 Bypass Surgery Explained",
        "bullets": [
            "Bypass surgery creates new routes around blocked coronary arteries",
            "A vessel from the leg or chest is used to build the bypass route",
            "It is open-heart surgery for severe or multiple blockages",
            "Recovery takes 6 to 12 weeks with cardiac rehabilitation support",
        ],
        "tip": "Bypass surgery significantly improves quality of life and survival rates.",
    },
    "beta blockers": {
        "headline": "💊 Beta-Blockers Explained",
        "bullets": [
            "Beta-blockers slow the heart rate and reduce the heart's workload",
            "They are used for heart failure, arrhythmia, and after heart attacks",
            "Common ones include metoprolol, atenolol, and carvedilol",
            "Side effects may include fatigue, cold hands, and mild dizziness",
        ],
        "tip": "Never stop beta-blockers suddenly — always taper with your doctor's guidance.",
    },
    "heart medications": {
        "headline": "💊 Common Heart Medications",
        "bullets": [
            "Statins lower cholesterol and reduce heart attack and stroke risk",
            "Beta-blockers slow heart rate and help manage heart failure",
            "ACE inhibitors relax blood vessels and reduce blood pressure",
            "Anticoagulants prevent dangerous blood clots in AFib patients",
        ],
        "tip": "Never adjust or stop heart medications without your doctor's approval.",
    },
    "alcohol": {
        "headline": "✅ Alcohol & Heart Health",
        "bullets": [
            "Heavy drinking raises blood pressure and weakens the heart muscle",
            "Alcohol can trigger atrial fibrillation and other arrhythmias",
            "Binge drinking significantly increases sudden cardiac event risk",
            "Reducing alcohol improves blood pressure and overall heart function",
        ],
        "tip": "No amount of alcohol is completely risk-free for the heart.",
    },
    "sleep": {
        "headline": "✅ Sleep & Heart Health",
        "bullets": [
            "Poor sleep raises blood pressure and increases heart disease risk",
            "Adults need 7 to 8 hours of quality sleep for heart recovery",
            "Sleep apnea — stopping breathing during sleep — strains the heart",
            "Good sleep allows the heart to rest and blood pressure to fall",
        ],
        "tip": "If you snore heavily or feel tired after sleeping, ask about a sleep study.",
    },
    "weight": {
        "headline": "⚖️ Weight & Heart Health",
        "bullets": [
            "Excess weight raises blood pressure and cholesterol levels",
            "Even losing 5 to 10 percent of body weight improves heart health",
            "Belly fat is particularly harmful for cardiovascular risk",
            "A combination of reduced calories and daily exercise works best",
        ],
        "tip": "Slow, steady weight loss of half a kilo per week is most sustainable.",
    },
    "aspirin": {
        "headline": "💊 Aspirin for Heart Health",
        "bullets": [
            "Aspirin prevents blood platelets from clumping and forming clots",
            "Low-dose aspirin is prescribed after heart attacks to prevent another",
            "It is not recommended for everyone — only those at high risk",
            "Stomach bleeding is a possible side effect with long-term use",
        ],
        "tip": "Never start or stop aspirin therapy without consulting your doctor.",
    },
    "diabetes symptoms": {
        "headline": "🩺 Symptoms of Diabetes",
        "bullets": [
            "Frequent urination, especially waking up multiple times at night",
            "Excessive thirst even after drinking plenty of water",
            "Unexplained weight loss despite eating normally",
            "Fatigue, blurred vision, and wounds that heal very slowly",
        ],
        "tip": "Type 2 diabetes develops slowly — regular blood tests help catch it early.",
    },
    "general heart health": {
        "headline": "🩺 Heart Health Overview",
        "bullets": ["The heart pumps blood and oxygen to every organ in the body",
                    "Heart disease is the leading cause of death worldwide",
                    "Up to 80 percent of heart disease is preventable",
                    "Diet, exercise, sleep, and stress management protect the heart"],
        "tip": "Knowing your blood pressure and cholesterol is a great first step.",
    },
}


def get_kb_response(message: str) -> str:
    """
    2-stage smart matcher:
    Stage 1 — detect CONDITION (blood pressure, heart disease, etc.)
    Stage 2 — detect INTENT (symptoms, causes, treatment, definition, etc.)
    Falls back gracefully at every level. Never returns generic garbage.
    """
    q = message.lower().strip()

    # ── Stage 1: Detect the medical CONDITION ─────────────────────────────

    # Emergency overrides everything — check first
    EMERGENCY_KW = ["heart attack", "cardiac arrest", "chest pain",
                    "not breathing", "unconscious", "collapse", "crushing"]
    if any(kw in q for kw in EMERGENCY_KW):
        return _fmt(MEDICAL_KB["heart attack emergency"])

    if any(kw in q for kw in ["cpr", "resuscit", "how to save", "rescue breath", "compression"]):
        return _fmt(MEDICAL_KB["cpr"])

    if any(kw in q for kw in ["which doctor", "what doctor", "what specialist",
                                "who should i see", "who to see", "who to consult",
                                "should i see", "go to which", "what type of doctor"]):
        return _fmt(MEDICAL_KB["which doctor"])

    if any(kw in q for kw in ["cardiologist"]):
        return _fmt(MEDICAL_KB["cardiologist"])

    # Map condition keywords → condition name
    CONDITION_MAP = [
        (["blood pressure", "hypertension", "bp", "high bp",
          "high blood", "pressure"],                           "blood pressure"),
        (["heart attack"],                                     "heart attack"),
        (["heart failure", "chf", "congestive heart"],         "heart failure"),
        (["heart disease", "coronary", "cad", "ischemic",
          "blocked artery", "blocked arteries"],               "heart disease"),
        (["cholesterol", "ldl", "hdl", "triglyceride",
          "lipid"],                                            "cholesterol"),
        (["arrhythmia", "irregular heartbeat", "palpitation",
          "irregular heart", "heart rhythm"],                  "arrhythmia"),
        (["atrial fibrillation", "afib", "a-fib", "af "],     "atrial fibrillation"),
        (["ecg", "ekg", "electrocardiogram", "ecg test",
          "heart tracing"],                                    "ecg test"),
        (["angioplasty", "balloon", "stent"],                  "angioplasty"),
        (["pacemaker", "pace maker"],                          "pacemaker"),
        (["bypass", "cabg", "bypass surgery"],                 "bypass surgery"),
        (["statin", "atorvastatin", "rosuvastatin",
          "simvastatin", "lipitor"],                           "statins"),
        (["aspirin", "blood thinner", "antiplatelet"],         "aspirin"),
        (["beta block", "metoprolol", "atenolol",
          "carvedilol"],                                       "beta blockers"),
        (["medication", "medicine", "drug", "pill",
          "prescription", "tablet"],                          "heart medications"),
        (["diabetes", "blood sugar", "glucose", "insulin",
          "diabetic"],                                         "diabetes heart"),
        (["stress", "anxiety", "worried", "mental", "panic"],  "stress heart"),
        (["smok", "cigarette", "tobacco", "nicotine",
          "vaping"],                                           "smoking heart"),
        (["alcohol", "drinking", "wine", "beer", "liquor"],    "alcohol"),
        (["sleep", "insomnia", "snoring", "apnea", "tired",
          "fatigue"],                                          "sleep"),
        (["weight", "obesity", "overweight", "bmi", "fat",
          "lose weight"],                                      "weight"),
        (["exercise", "workout", "walk", "gym", "fitness",
          "physical activity", "yoga", "sport", "run"],        "exercise heart"),
        (["diet", "food", "eat", "meal", "nutrition",
          "mediterranean", "fruit", "vegetable", "salt",
          "sodium"],                                           "heart healthy diet"),
        (["omega", "fish oil", "fatty acid"],                  "heart healthy diet"),
        (["rehabilitation", "rehab", "recovery program",
          "cardiac program"],                                  "cardiac rehabilitation"),
        (["heart health", "cardiac health", "heart"],          "heart"),
    ]

    condition = None
    for keywords, cond in CONDITION_MAP:
        if any(kw in q for kw in keywords):
            condition = cond
            break

    if not condition:
        return _fmt(MEDICAL_KB["general heart health"])

    # ── Stage 2: Detect INTENT ─────────────────────────────────────────────

    # Detect intent from the question words
    is_symptom  = any(kw in q for kw in [
        "symptom", "sign", "feel", "notice", "warning", "how do i know",
        "what does it feel", "how does it feel", "what are the signs",
        "how can i tell", "indication"])

    is_cause    = any(kw in q for kw in [
        "cause", "reason", "why", "risk factor", "what causes",
        "responsible", "lead to", "trigger", "risk"])

    is_treatment = any(kw in q for kw in [
        "treat", "cure", "medication", "medicine", "drug", "therapy",
        "manage", "control", "lower", "reduce", "fix", "recover",
        "what can i do", "what should i do", "how to manage",
        "how to control", "how to lower", "how to reduce"])

    is_prevention = any(kw in q for kw in [
        "prevent", "avoid", "stop", "protect", "reduce risk",
        "not get", "how to avoid"])

    is_diet      = any(kw in q for kw in [
        "diet", "food", "eat", "nutrition", "meal", "drink",
        "fruit", "vegetable", "what to eat", "what not to eat"])

    is_exercise  = any(kw in q for kw in [
        "exercise", "workout", "walk", "gym", "yoga", "sport",
        "physical", "activity", "fitness"])

    is_definition = any(kw in q for kw in [
        "what is", "what are", "define", "definition", "explain",
        "mean", "meaning", "tell me about", "describe", "overview",
        "what does", "about"])

    is_general = any(kw in q for kw in [
        "how", "can", "is it", "are there", "do i", "should i",
        "is there", "help", "advice", "information", "info",
        "when", "how long", "how serious"])

    # ── Stage 3: Pick the best KB entry for condition + intent ─────────────

    # Special KB keys that have their own entries
    DIRECT_KEYS = {
        "blood pressure": {
            "symptom":    "blood pressure symptoms",
            "cause":      "blood pressure causes",
            "treatment":  "blood pressure treatment",
            "diet":       "blood pressure diet",
            "definition": "blood pressure definition",
            "general":    "blood pressure definition",
        },
        "heart disease": {
            "symptom":    "heart disease symptoms",
            "cause":      "heart disease causes",
            "prevention": "heart disease prevention",
            "definition": "heart disease definition",
            "general":    "heart disease definition",
        },
        "cholesterol": {
            "symptom":   "cholesterol symptoms",
            "diet":      "cholesterol diet",
            "treatment": "cholesterol treatment",
            "cause":     "cholesterol causes",
            "general":   "cholesterol definition",
        },
        "heart": {
            "diet":      "heart healthy diet",
            "exercise":  "exercise heart",
            "general":   "general heart health",
        },
        "diabetes heart": {
            "general":  "diabetes heart",
            "symptom":  "diabetes symptoms",
        },
    }

    key_map = DIRECT_KEYS.get(condition, {})

    # Determine intent priority
    # NOTE: treatment must come before diet because "treatment" contains "eat"
    intent = None
    if is_symptom:     intent = "symptom"
    elif is_cause:     intent = "cause"
    elif is_treatment: intent = "treatment"
    elif is_prevention:intent = "prevention"
    elif is_exercise:  intent = "exercise"
    elif is_diet:      intent = "diet"
    elif is_definition:intent = "definition"
    elif is_general:   intent = "general"
    else:              intent = "general"

    # Try to find matching KB entry
    kb_key = key_map.get(intent) or key_map.get("general") or condition

    entry = MEDICAL_KB.get(kb_key)
    if entry:
        return _fmt(entry)

    # Fallback: try the condition directly as a key
    entry = MEDICAL_KB.get(condition)
    if entry:
        return _fmt(entry)

    # Last resort: general heart health
    return _fmt(MEDICAL_KB["general heart health"])


def _fmt(entry: dict) -> str:
    """Format a MEDICAL_KB entry into clean response string."""
    lines = [f"{entry['headline']}:"]
    lines += [f"• {b}" for b in entry["bullets"]]
    if entry.get("tip"):
        lines.append(f"💡 Tip: {entry['tip']}")
    return "\n".join(lines)


EMOJI_RE = re.compile(
    r'[\U00010000-\U0010ffff\U00002600-\U000027BF'
    r'\U0001F300-\U0001F9FF\U00002702-\U000027B0]'
)

def format_for_tts(text: str) -> str:
    text = EMOJI_RE.sub('', text)
    text = re.sub(r'^[•\-\*]\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s{2,}', ' ', text)
    return text.strip()


# ── Feature metadata ──────────────────────────────────────────────────────────
FEATURES = [
    {"name": "age",      "desc": "Age in years",                          "min": 1,   "max": 120},
    {"name": "sex",      "desc": "Sex (1=male, 0=female)",                "min": 0,   "max": 1},
    {"name": "cp",       "desc": "Chest pain type (0-3)",                 "min": 0,   "max": 3},
    {"name": "trestbps", "desc": "Resting blood pressure (mmHg)",         "min": 50,  "max": 300},
    {"name": "chol",     "desc": "Serum cholesterol (mg/dl)",             "min": 100, "max": 700},
    {"name": "fbs",      "desc": "Fasting blood sugar >120mg/dl (1=yes)", "min": 0,   "max": 1},
    {"name": "restecg",  "desc": "Resting ECG results (0-2)",             "min": 0,   "max": 2},
    {"name": "thalach",  "desc": "Max heart rate achieved",               "min": 60,  "max": 220},
    {"name": "exang",    "desc": "Exercise-induced angina (1=yes)",       "min": 0,   "max": 1},
    {"name": "oldpeak",  "desc": "ST depression (exercise vs rest)",      "min": 0,   "max": 10},
    {"name": "slope",    "desc": "Slope of peak exercise ST (0-2)",       "min": 0,   "max": 2},
    {"name": "ca",       "desc": "Major vessels colored (0-3)",           "min": 0,   "max": 3},
    {"name": "thal",     "desc": "Thal: 0=norm 1=fixed 2=reversable",    "min": 0,   "max": 3},
]


# ════════════════════════════════════════════════════════════════════════════
#  ROUTES
# ════════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True, silent=True)
        if not data:
            return jsonify({"error": "Request body must be JSON"}), 400
        features = data.get("features")
        if not isinstance(features, list) or len(features) != 13:
            return jsonify({"error": "Provide exactly 13 numeric features"}), 422
        try:
            features = [float(v) for v in features]
        except (TypeError, ValueError) as e:
            return jsonify({"error": f"All features must be numeric: {e}"}), 422
        input_warnings = [
            f"{FEATURES[i]['name']}={v} out of range"
            for i, v in enumerate(features)
            if not (FEATURES[i]["min"] <= v <= FEATURES[i]["max"])
        ]
        model, scaler = get_model()
        from model import predict as ml_predict
        result = ml_predict(features, model=model, scaler=scaler)
        conf = result["confidence"]
        if result["prediction"] == 1:
            risk_level = "High" if (conf or 0) > 0.75 else "Moderate"
            advice = "⚠️ The model detected possible indicators. Please consult a cardiologist."
        else:
            risk_level = "Low"
            advice = "✅ No significant indicators detected. Maintain healthy habits."
        return jsonify({**result, "risk_level": risk_level,
                        "advice": advice, "warnings": input_warnings})
    except FileNotFoundError:
        return jsonify({"error": "Run python model.py first"}), 503
    except Exception as e:
        log.exception("Error in /predict")
        return jsonify({"error": str(e)}), 500


@app.route("/chat/stream", methods=["POST"])
def chat_stream():
    """
    Streaming chat endpoint.
    Accepts full conversation history from frontend.
    Returns Server-Sent Events (SSE) stream.

    Request body:
    {
      "messages": [
        {"role": "user",      "content": "I have chest pain"},
        {"role": "assistant", "content": "🚨 This sounds like..."},
        {"role": "user",      "content": "what should I do"}
      ]
    }
    """
    try:
        data = request.get_json(force=True, silent=True)
        if not data:
            return jsonify({"error": "Request body must be JSON"}), 400

        messages = data.get("messages", [])
        if not messages:
            return jsonify({"error": "No messages provided"}), 400

        # Validate last message
        last = messages[-1]
        if last.get("role") != "user" or not last.get("content", "").strip():
            return jsonify({"error": "Last message must be from user"}), 400

        # Trim history to last 10 messages (5 exchanges) to stay within token limits
        history = messages[-10:]

        api_key = os.environ.get("GROQ_API_KEY")

        def generate():
            if not api_key:
                # Offline: use KB fallback, stream it word by word
                kb_reply = get_kb_response(last["content"])
                words    = kb_reply.split(" ")
                full     = ""
                for i, word in enumerate(words):
                    chunk = word + (" " if i < len(words) - 1 else "")
                    full += chunk
                    yield f"data: {json.dumps({'delta': chunk, 'done': False})}\n\n"
                tts = format_for_tts(full)
                yield f"data: {json.dumps({'delta': '', 'done': True, 'full': full, 'tts': tts, 'method': 'knowledge-base'})}\n\n"
                return

            # Build Groq messages with system prompt
            groq_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + [
                {"role": m["role"], "content": m["content"]}
                for m in history
            ]

            full = ""
            try:
                for token in call_groq_stream(groq_messages):
                    full += token
                    yield f"data: {json.dumps({'delta': token, 'done': False})}\n\n"
                tts = format_for_tts(full)
                yield f"data: {json.dumps({'delta': '', 'done': True, 'full': full, 'tts': tts, 'method': 'groq-llama3'})}\n\n"

            except Exception as e:
                log.warning("Groq stream failed: %s — falling back to KB", e)
                kb_reply = get_kb_response(last["content"])
                tts      = format_for_tts(kb_reply)
                yield f"data: {json.dumps({'delta': kb_reply, 'done': True, 'full': kb_reply, 'tts': tts, 'method': 'knowledge-base'})}\n\n"

        return Response(
            stream_with_context(generate()),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            }
        )

    except Exception as e:
        log.exception("Error in /chat/stream")
        return jsonify({"error": str(e)}), 500


@app.route("/chat", methods=["POST"])
def chat():
    """
    Non-streaming fallback chat endpoint.
    Also accepts conversation history.
    """
    try:
        data = request.get_json(force=True, silent=True)
        if not data:
            return jsonify({"error": "Request body must be JSON"}), 400

        # Support both old format {message: "..."} and new {messages: [...]}
        if "messages" in data:
            messages = data["messages"]
            last_msg = messages[-1]["content"] if messages else ""
        else:
            last_msg = data.get("message", "").strip()
            messages = [{"role": "user", "content": last_msg}]

        if not last_msg:
            return jsonify({"error": "Message cannot be empty"}), 400

        history  = messages[-10:]
        api_key  = os.environ.get("GROQ_API_KEY")
        method   = "knowledge-base"
        reply    = None

        if api_key:
            try:
                groq_msgs = [{"role": "system", "content": SYSTEM_PROMPT}] + [
                    {"role": m["role"], "content": m["content"]}
                    for m in history
                ]
                reply  = call_groq_full(groq_msgs)
                method = "groq-llama3"
            except Exception as e:
                log.warning("Groq failed: %s", e)

        if reply is None:
            reply = get_kb_response(last_msg)

        tts = format_for_tts(reply)
        return jsonify({"reply": reply, "tts_text": tts, "method": method})

    except Exception as e:
        log.exception("Error in /chat")
        return jsonify({"error": str(e)}), 500


@app.route("/features", methods=["GET"])
def feature_info():
    return jsonify({"features": FEATURES})


@app.route("/health", methods=["GET"])
def health_check():
    groq_on = bool(os.environ.get("GROQ_API_KEY"))
    return jsonify({
        "status":   "ok",
        "service":  "CardioAI v9",
        "groq":     "configured" if groq_on else "not set",
        "model":    "llama3-8b-8192" if groq_on else "knowledge-base",
        "features": ["streaming", "conversation-history", "cardiac-kb"],
    })


if __name__ == "__main__":
    key = os.environ.get("GROQ_API_KEY")
    log.info("═" * 55)
    log.info("  CardioAI v9 — Conversational Edition")
    log.info("═" * 55)
    log.info("  Groq  : %s", "✅ Ready" if key else "❌ Not set — KB fallback active")
    if not key:
        log.info("  Key   : https://console.groq.com (free)")
        log.info("  Set   : $env:GROQ_API_KEY='gsk_...'")
    log.info("═" * 55)
    get_model()
    get_rag()
    log.info("Ready → http://127.0.0.1:5000")
    app.run(debug=False, host="0.0.0.0", port=5000, threaded=True)