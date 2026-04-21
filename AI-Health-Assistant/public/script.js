/**
 * script.js — CardioAI v9 (ChatGPT-Style Conversational)
 * ========================================================
 * Key features:
 *  • Full conversation history stored client-side
 *  • Streaming responses — text appears token by token
 *  • Auto-scroll, auto-resize textarea
 *  • Emergency message detection with red styling
 *  • Manual TTS — click Speak on any message
 *  • Voice input (STT) for chat and prediction form
 */

"use strict";

/* ══════════════════════════════════════════════════════════
   STATE
══════════════════════════════════════════════════════════ */
let conversationHistory = [];   // Full chat history [{role, content}, ...]
let isStreaming         = false; // Prevent double-sends during stream
let isRecording         = false;
let recognition         = null;
let voicePredMode       = false;
let featureMeta         = [];

/* ══════════════════════════════════════════════════════════
   INIT
══════════════════════════════════════════════════════════ */
document.addEventListener("DOMContentLoaded", () => {
  loadFeatureMeta();
  renderWelcome();
  setupSTT();
  checkStatus();
});

/* ══════════════════════════════════════════════════════════
   TAB SWITCHING
══════════════════════════════════════════════════════════ */
function switchTab(name, btn) {
  document.querySelectorAll(".tab").forEach(t => t.classList.remove("active"));
  document.querySelectorAll(".nav-btn").forEach(b => b.classList.remove("active"));
  document.getElementById(`tab-${name}`).classList.add("active");
  btn.classList.add("active");
}

/* ══════════════════════════════════════════════════════════
   STATUS CHECK
══════════════════════════════════════════════════════════ */
async function checkStatus() {
  try {
    const r = await fetch("/health");
    const d = await r.json();
    const el = document.getElementById("status-text");
    if (el) {
      el.textContent = d.groq === "configured"
        ? `Groq · LLaMA 3 · Streaming`
        : "Knowledge Base Mode";
      el.style.color = d.groq === "configured" ? "var(--teal)" : "var(--gold)";
    }
  } catch { /* ignore */ }
}

/* ══════════════════════════════════════════════════════════
   TTS — manual speak button
══════════════════════════════════════════════════════════ */
function speak(text, btn) {
  if (!window.speechSynthesis) {
    alert("Text-to-speech requires Chrome or Edge.");
    return;
  }
  if (btn.classList.contains("speaking")) {
    window.speechSynthesis.cancel();
    resetTTSBtn(btn);
    return;
  }
  window.speechSynthesis.cancel();
  document.querySelectorAll(".tts-btn.speaking").forEach(resetTTSBtn);

  const u    = new SpeechSynthesisUtterance(text);
  u.lang     = "en-US";
  u.rate     = 0.93;
  u.pitch    = 1.0;
  u.volume   = 1.0;
  u.onstart  = () => { btn.classList.add("speaking"); btn.innerHTML = ttsIcon() + " Stop"; };
  u.onend    = u.onerror = () => resetTTSBtn(btn);
  window.speechSynthesis.speak(u);
}

function resetTTSBtn(btn) {
  btn.classList.remove("speaking");
  btn.innerHTML = ttsIcon() + " Speak";
}

function ttsIcon() {
  return `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
    <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"/>
    <path d="M15.54 8.46a5 5 0 010 7.07"/>
  </svg>`;
}

/* ══════════════════════════════════════════════════════════
   RESPONSE RENDERER
   Handles: headings, bullets, paragraphs, tips
══════════════════════════════════════════════════════════ */
function renderBotText(text, isEmergency) {
  const lines = text.split("\n").map(l => l.trim()).filter(Boolean);
  let html    = "";
  let first   = true;

  for (const line of lines) {
    if (line.startsWith("💡")) {
      const content = line.replace(/^💡\s*(Tip:\s*)?/i, "");
      html += `<div class="resp-tip">💡 ${content}</div>`;
    } else if (line.startsWith("•")) {
      const content = line.replace(/^•\s*/, "");
      html += `<div class="resp-bullet">
                 <span class="resp-dot">•</span>
                 <span class="resp-text">${content}</span>
               </div>`;
    } else if (first && (line.endsWith(":") || line.includes("–") || line.includes("🩺") || line.includes("🚨") || line.includes("⚠️") || line.includes("✅") || line.includes("🥗") || line.includes("🏃") || line.includes("💊") || line.includes("🔬") || line.includes("👨"))) {
      const cls = isEmergency ? "resp-heading emergency" : "resp-heading";
      html  += `<div class="${cls}">${line}</div>`;
      first  = false;
    } else {
      // Regular paragraph text — for conversational responses
      html  += `<p class="resp-para">${line}</p>`;
      first  = false;
    }
  }
  return html || `<p>${esc(text)}</p>`;
}

function isEmergency(text) {
  return /🚨|heart attack|cardiac arrest|emergency|immediate steps|call emergency/i.test(text);
}

/* ══════════════════════════════════════════════════════════
   WELCOME SCREEN
══════════════════════════════════════════════════════════ */
function renderWelcome() {
  conversationHistory = []; // Reset history on clear
  document.getElementById("chat-window").innerHTML = `
    <div class="welcome-msg">
      <div class="welcome-icon">🫀</div>
      <h2>CardioAI — Your Heart Health Assistant</h2>
      <p>Ask me anything about cardiac health — symptoms, tests, medications,
         diet, emergencies, or follow-up questions. I remember our conversation.</p>
      <div class="quick-chips">
        ${[
          ["I have chest pain",               true ],
          ["Symptoms of high blood pressure",  false],
          ["What is a cardiologist?",          false],
          ["How do I lower my cholesterol?",   false],
          ["What is an ECG test?",             false],
          ["How to perform CPR?",              true ],
        ].map(([c, isEm]) =>
          `<div class="chip ${isEm ? 'emergency-chip' : ''}"
                onclick="quickChat('${c.replace(/'/g,"\\'")}')">
             ${c}
           </div>`
        ).join("")}
      </div>
    </div>`;
}

function quickChat(text) {
  document.getElementById("chat-input").value = text;
  sendChat();
}

function handleKey(e) {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendChat();
  }
}

function autoResize(el) {
  el.style.height = "auto";
  el.style.height = Math.min(el.scrollHeight, 200) + "px";
}

/* ══════════════════════════════════════════════════════════
   MAIN SEND — with streaming + conversation history
══════════════════════════════════════════════════════════ */
async function sendChat() {
  if (isStreaming) return;

  const inp = document.getElementById("chat-input");
  const msg = inp.value.trim();
  if (!msg) return;

  inp.value = "";
  inp.style.height = "auto";
  window.speechSynthesis?.cancel();
  document.querySelectorAll(".tts-btn.speaking").forEach(resetTTSBtn);

  // Add to history and display user bubble
  conversationHistory.push({ role: "user", content: msg });
  addUserBubble(msg);

  // Create bot bubble with streaming placeholder
  const { el: botEl, contentEl, metaEl } = createBotBubble();
  scrollBottom();

  isStreaming = true;
  setSendState(false);

  let fullText = "";
  let ttsText  = "";
  let method   = "knowledge-base";

  try {
    const res = await fetch("/chat/stream", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ messages: conversationHistory }),
    });

    if (!res.ok) {
      throw new Error(`Server error ${res.status}`);
    }

    const reader   = res.body.getReader();
    const decoder  = new TextDecoder();
    let   buffer   = "";

    // Show typing indicator first
    contentEl.innerHTML = typingHTML();

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop(); // keep incomplete line in buffer

      for (const line of lines) {
        if (!line.startsWith("data: ")) continue;
        try {
          const parsed = JSON.parse(line.slice(6));

          if (parsed.done) {
            fullText = parsed.full || fullText;
            ttsText  = parsed.tts  || "";
            method   = parsed.method || method;
          } else if (parsed.delta) {
            fullText += parsed.delta;
            // Render progressively
            const emergency = isEmergency(fullText);
            if (emergency && !botEl.classList.contains("emergency")) {
              botEl.classList.add("emergency");
              botEl.querySelector(".bot-av").textContent = "🚨";
            }
            contentEl.innerHTML = renderBotText(fullText, emergency);
            scrollBottom();
          }
        } catch { /* skip malformed SSE line */ }
      }
    }

    // Final render
    const emergency = isEmergency(fullText);
    contentEl.innerHTML = renderBotText(fullText, emergency);

    // Add Speak button + method badge
    const badge = method === "groq-llama3"
      ? `<span class="method-badge">⚡ Groq</span>`
      : `<span class="method-badge" style="color:var(--teal)">📚 KB</span>`;

    const safeTts = (ttsText || fullText)
      .replace(/\\/g, "\\\\")
      .replace(/'/g, "\\'")
      .replace(/\n/g, " ")
      .replace(/"/g, "&quot;");

    const btnId = "tts-" + Date.now();
    metaEl.innerHTML = `
      <button class="tts-btn" id="${btnId}" onclick="speak('${safeTts}', this)">
        ${ttsIcon()} Speak
      </button>
      ${badge}
      <span class="msg-time">${now()}</span>`;

    // Add assistant response to conversation history
    conversationHistory.push({ role: "assistant", content: fullText });

    // Keep history manageable — last 20 messages (10 exchanges)
    if (conversationHistory.length > 20) {
      conversationHistory = conversationHistory.slice(-20);
    }

  } catch (err) {
    contentEl.innerHTML = `<p style="color:var(--accent)">⚠️ ${esc(err.message)}</p>`;
    metaEl.innerHTML    = `<span class="msg-time">${now()}</span>`;
  } finally {
    isStreaming = false;
    setSendState(true);
    scrollBottom();
  }
}

/* ══════════════════════════════════════════════════════════
   BUBBLE BUILDERS
══════════════════════════════════════════════════════════ */
function addUserBubble(text) {
  rmWelcome();
  const el = make("div", "msg user");
  el.innerHTML = `
    <div class="msg-avatar">You</div>
    <div class="msg-body">
      <div class="msg-bubble user-bubble"><p>${esc(text)}</p></div>
      <div class="msg-meta"><span class="msg-time">${now()}</span></div>
    </div>`;
  win().appendChild(el);
}

function createBotBubble() {
  rmWelcome();
  const el        = make("div", "msg bot");
  const contentEl = make("div", "msg-bubble bot-bubble");
  const metaEl    = make("div", "msg-meta");
  const body      = make("div", "msg-body");

  body.appendChild(contentEl);
  body.appendChild(metaEl);

  el.innerHTML = `<div class="msg-avatar bot-av">🤖</div>`;
  el.appendChild(body);
  win().appendChild(el);

  return { el, contentEl, metaEl };
}

function typingHTML() {
  return `<div class="typing-bubble">
    <span class="tdot"></span>
    <span class="tdot"></span>
    <span class="tdot"></span>
  </div>`;
}

function clearChat() {
  window.speechSynthesis?.cancel();
  conversationHistory = [];
  renderWelcome();
}

/* ══════════════════════════════════════════════════════════
   UI HELPERS
══════════════════════════════════════════════════════════ */
function setSendState(enabled) {
  const btn = document.querySelector(".send-btn");
  if (btn) {
    btn.disabled = !enabled;
    btn.style.opacity = enabled ? "1" : "0.5";
  }
}

function win()          { return document.getElementById("chat-window"); }
function make(t, c)     { const e = document.createElement(t); e.className = c; return e; }
function scrollBottom() { const w = win(); w.scrollTop = w.scrollHeight; }
function rmWelcome()    { document.querySelector(".welcome-msg")?.remove(); }
function removeEl(id)   { document.getElementById(id)?.remove(); }
function esc(s)         { return String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;"); }
function now()          { return new Date().toLocaleTimeString([], {hour:"2-digit",minute:"2-digit"}); }

/* ══════════════════════════════════════════════════════════
   PREDICTION FORM
══════════════════════════════════════════════════════════ */
async function loadFeatureMeta() {
  try {
    featureMeta = (await (await fetch("/features")).json()).features;
  } catch {
    featureMeta = [
      {name:"age",desc:"Age in years",min:1,max:120},
      {name:"sex",desc:"Sex (1=male 0=female)",min:0,max:1},
      {name:"cp",desc:"Chest pain type (0-3)",min:0,max:3},
      {name:"trestbps",desc:"Resting blood pressure",min:50,max:300},
      {name:"chol",desc:"Serum cholesterol mg/dl",min:100,max:700},
      {name:"fbs",desc:"Fasting blood sugar>120 (1=yes)",min:0,max:1},
      {name:"restecg",desc:"Resting ECG (0-2)",min:0,max:2},
      {name:"thalach",desc:"Max heart rate",min:60,max:220},
      {name:"exang",desc:"Exercise angina (1=yes)",min:0,max:1},
      {name:"oldpeak",desc:"ST depression",min:0,max:10},
      {name:"slope",desc:"Slope of ST (0-2)",min:0,max:2},
      {name:"ca",desc:"Major vessels (0-3)",min:0,max:3},
      {name:"thal",desc:"Thal (0-3)",min:0,max:3},
    ];
  }
  document.getElementById("predict-form").innerHTML =
    featureMeta.map((f, i) => `
      <div class="form-field">
        <label for="f${i}">${f.name.toUpperCase()}</label>
        <span class="field-desc">${f.desc}</span>
        <input type="number" id="f${i}"
               placeholder="${f.min}–${f.max}"
               min="${f.min}" max="${f.max}" step="any"/>
      </div>`).join("");
}

function fillSample() {
  [63,1,3,145,233,1,0,150,0,2.3,0,0,1].forEach((v,i) => {
    const el = document.getElementById(`f${i}`);
    if (el) el.value = v;
  });
  setHint("Sample loaded — click Run Analysis");
}

function clearForm() {
  featureMeta.forEach((_,i) => {
    const el = document.getElementById(`f${i}`);
    if (el) el.value = "";
  });
  document.getElementById("predict-result").classList.add("hidden");
}

async function submitPrediction() {
  const features = featureMeta.map((_,i) => {
    const v = document.getElementById(`f${i}`)?.value;
    return (v === "" || v === undefined) ? NaN : parseFloat(v);
  });
  if (features.filter(isNaN).length > 0) {
    alert("Please fill all 13 fields.");
    return;
  }
  showResult(null, "loading");
  try {
    const res  = await fetch("/predict", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ features }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error);
    showResult(data);
  } catch (err) {
    showResult(null, "error", err.message);
  }
}

function showResult(data, state="ok", errMsg="") {
  const el = document.getElementById("predict-result");
  el.classList.remove("hidden","positive","negative");
  if (state === "loading") {
    el.innerHTML = `<div style="color:var(--text-2);font-size:14px">⏳ Analysing your data…</div>`;
    el.classList.remove("hidden"); return;
  }
  if (state === "error") {
    el.innerHTML = `<div style="color:var(--accent);font-size:14px">❌ ${errMsg}</div>`;
    el.classList.remove("hidden"); return;
  }
  const pos  = data.prediction === 1;
  const conf = data.confidence != null ? Math.round(data.confidence * 100) : null;
  el.classList.add(pos ? "positive" : "negative");
  el.innerHTML = `
    <div class="result-header">
      <div class="result-icon">${pos ? "🚨" : "💚"}</div>
      <div>
        <div class="result-title">${data.label}</div>
        <div class="result-risk">Risk Level: <strong>${data.risk_level}</strong></div>
      </div>
    </div>
    ${conf != null ? `<div class="result-conf">
      <div class="conf-bar"><div class="conf-fill" style="width:${conf}%"></div></div>
      <span class="conf-label">Confidence: ${conf}%</span></div>` : ""}
    <div class="result-advice">${data.advice}</div>
    ${data.warnings?.length ? `<div class="result-warnings">⚠️ ${data.warnings.join("; ")}</div>` : ""}`;
  el.classList.remove("hidden");
}

/* ══════════════════════════════════════════════════════════
   VOICE INPUT (STT)
══════════════════════════════════════════════════════════ */
function setupSTT() {
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SR) return;
  recognition = new SR();
  recognition.continuous = false;
  recognition.interimResults = true;
  recognition.lang = "en-US";

  recognition.onresult = (e) => {
    const t = Array.from(e.results).map(r => r[0].transcript).join("");
    if (voicePredMode) {
      setHint(`🎤 "${t}"`);
      if (e.results[e.results.length-1].isFinal) handleVoicePred(t);
    } else {
      document.getElementById("chat-input").value = t;
      if (e.results[e.results.length-1].isFinal) { setHint(""); sendChat(); }
      else setHint(`🎤 "${t}"`);
    }
  };
  recognition.onend   = () => { setRec(false); voicePredMode = false; setHint(""); };
  recognition.onerror = (e) => { setRec(false); voicePredMode = false; setHint(`❌ ${e.error}`); };
}

function toggleVoice() {
  if (!recognition) { alert("Voice requires Chrome or Edge."); return; }
  if (isStreaming)   { return; }
  if (isRecording)   { recognition.stop(); setRec(false); }
  else               { recognition.start(); setRec(true); setHint("🎤 Listening…"); }
}

function startVoicePrediction() {
  if (!recognition) { alert("Voice requires Chrome or Edge."); return; }
  if (isRecording) recognition.stop();
  voicePredMode = true;
  recognition.start(); setRec(true);
  setHint("🎤 Say all 13 values separated by commas…");
  switchTab("predict", document.querySelector('[data-tab="predict"]'));
}

function handleVoicePred(t) {
  const nums = (t.match(/-?\d+\.?\d*/g) || []).map(Number);
  setHint(`Captured ${nums.length}/13 values`);
  if (nums.length === 13) {
    nums.forEach((v,i) => { const el=document.getElementById(`f${i}`); if(el) el.value=v; });
    setTimeout(submitPrediction, 500);
  } else { setHint(`⚠️ Got ${nums.length}/13 — please retry`); }
}

function setRec(s)    { isRecording=s; document.getElementById("mic-btn")?.classList.toggle("recording",s); }
function setHint(msg) { const el=document.getElementById("voice-hint"); if(el) el.textContent=msg; }