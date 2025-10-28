// src/Game.js
import React, { useEffect, useMemo, useRef, useState } from "react";

function injectOnce(id, css) {
  if (typeof document === "undefined") return; // SSR guard
  if (document.getElementById(id)) return;
  const s = document.createElement("style");
  s.id = id;
  s.innerHTML = css;
  document.head.appendChild(s);
}

/* --- GAME CSS --- */
const gameCSS = `
:root{
  --g-bg:#0b1020;
  --g-card:#0d1429cc;
  --g-text:#e6e9f5;
  --g-muted:#9aa4c4;
  --g-border:#30406b88;
  --g-green:#22c55e; --g-amber:#f59e0b; --g-red:#ef4444; --g-ring:#60a5fa;
}
.game-root{
  min-height:100vh;
  display:grid;
  grid-template-rows:auto 1fr;
  background:#0b1020;
  color:#e5e7eb;
}
.game-wrap{
  height:100%;
  background:#0b1020;
  color:var(--g-text);
  font-family: Inter, system-ui, Arial, sans-serif;
  padding:16px; box-sizing:border-box;
  display:grid; grid-template-rows:auto 1fr; gap:8px;
}
.game-header{
  display:flex; justify-content:space-between; align-items:center;
  position:sticky; top:0; z-index:2; padding:6px 0;
  backdrop-filter: blur(10px); background:#0b1020aa; border-bottom:1px solid #ffffff17;
}
.controls{display:flex; align-items:center; gap:8px; flex-wrap:wrap}
.select{background:#0f1833; border:1px solid #2a3b74; color:#e6e9f5; border-radius:12px; padding:8px 10px}
.head-btn{
  padding:8px 12px; border-radius:10px; border:1px solid rgba(255,255,255,.25); background:#ffffff; color:#0f172a; cursor:pointer
}
.head-btn[disabled]{opacity:.5; cursor:not-allowed}
.game-shell{display:grid; place-items:center; padding:10px;}

.card-wrapper{ width:100%; display:grid; place-items:center; gap:8px; }
.progress{height:12px; width:min(980px,96vw); background:#0b1430; border-radius:999px; overflow:hidden; border:1px solid #2a3b74}
.bar{height:100%; background:linear-gradient(90deg,#3b82f6,#8b5cf6); width:0%}
.card{width:min(980px,96vw); height:520px; perspective:1500px}
.card-inner{position:relative; width:100%; height:100%; transform-style:preserve-3d; transition:transform .55s cubic-bezier(.2,.8,.2,1)}
.card.is-back .card-inner{transform:rotateY(180deg)}
.face{
  position:absolute; inset:0; padding:28px; background:var(--g-card);
  border:1px solid var(--g-border); border-radius:22px;
  box-shadow: 0 18px 60px rgba(0,0,0,.45);
  backface-visibility:hidden; display:flex; flex-direction:column; gap:14px;
}
.back{transform:rotateY(180deg)}
.tinycaps{font-size:12px; text-transform:uppercase; letter-spacing:.08em; color:var(--g-muted)}
.badge .lbl, .lbl{font-size:12px; color:var(--g-muted)}
.desc{font-size:24px; font-weight:900; letter-spacing:.2px}
.choices{margin-top:auto; display:flex; gap:14px; justify-content:center}
.btn{
  padding:16px 22px; border-radius:14px; font-size:16px; font-weight:800;
  border:none; cursor:pointer; color:#0b1020; background:#e6e9f5;
  box-shadow:0 10px 24px rgba(0,0,0,.25); transition:transform .12s ease, box-shadow .2s;
}
.btn:hover{transform:translateY(-2px); box-shadow:0 12px 26px rgba(0,0,0,.35)}
.b-green{background: #22c55e}
.b-amber{background:#f59e0b}
.b-red{background:#ef4444}
.tag{display:inline-block; padding:8px 14px; border-radius:999px; font-size:14px; font-weight:900}
.t-CLEAR{background:#22c55e; color:#042c16}
.t-OMITTED{background:#ef4444; color:#3f0a0a}
.t-AMBIGUOUS{background:#f59e0b; color:#2a1600}
.result{margin:0 0 8px}
.meta{margin-top:4px; color:var(--g-muted)}
.modal{position:fixed; inset:0; display:none; place-items:center; background:rgba(4,6,14,.55); z-index:9999}
.modal.show{display:grid}
.card-modal{width:min(640px,92vw); background:#0f1633; color:var(--g-text); border-radius:18px; border:1px solid #2a3b74; box-shadow:0 22px 80px rgba(0,0,0,.6); padding:22px}
.modal-title{margin:0 0 6px}
.stats{display:grid; grid-template-columns:repeat(3,1fr); gap:12px; margin:16px 0}
.stat{background:#0b1430; border:1px solid var(--g-border); border-radius:12px; padding:14px; text-align:center}
.explanation{white-space:normal; word-break:break-word; max-height:220px; overflow:auto; line-height:1.55; color:#d6ddff}

/* --- Loading overlay --- */
.loader-overlay{
  position:fixed; inset:0; display:flex; flex-direction:column; align-items:center; justify-content:center;
  background:radial-gradient(1200px 700px at 50% -20%, #1f2a5a44, transparent), #0b1020; z-index:9998;
}
.spinner{
  width:72px; height:72px; border-radius:9999px; border:6px solid rgba(255,255,255,.12); border-top-color:#8b5cf6;
  animation:spin 1s linear infinite;
}
@keyframes spin{to{transform:rotate(360deg)}}
.loading-text{
  margin-top:16px; color:#cdd7ff; font-weight:700; letter-spacing:.02em;
  animation:pulse 1.6s ease-in-out infinite;
}
@keyframes pulse{
  0%,100%{opacity:.55} 50%{opacity:1}
}
.error-box{
  margin-top:16px; color:#ffb4b4; font-weight:600;
  border:1px solid #ff6b6b66; background:#2a0f16; padding:10px 14px; border-radius:12px;
}
`;

/* ---------- Utilities ---------- */
function safeText(x, fb = "") {
  if (x === null || x === undefined) return fb;
  const s = String(x).trim();
  return s || fb;
}

function isGarbage(s) {
  if (!s) return true;
  const t = String(s).trim();
  if (!t) return true;
  if (t.length < 2) return true;
  const nonAlpha = t.replace(/[A-Za-zğüşöçıİĞÜŞÖÇ]/g, "");
  if (nonAlpha.length / t.length > 0.7) return true;
  if (/https?:\/\//i.test(t)) return true;
  if (/^[0-9a-f]{16,}$/i.test(t)) return true;
  return false;
}

const STOP = new Set([
  "the","a","an","and","or","of","to","in","on","for","with","by","at","from",
  "we","our","you","your","is","are","be","as","that","this","it","they","them","their","may","can","will",
  "data","information","info","veri","bilgi"
]);

function topK(text = "", k = 3) {
  const words = text
    .toLowerCase()
    .replace(/[^a-zA-ZğüşöçıİĞÜŞÖÇ\s]/g, " ")
    .split(/\s+/)
    .filter(w => w && !STOP.has(w) && w.length > 2);
  const freq = new Map();
  for (const w of words) freq.set(w, (freq.get(w) || 0) + 1);
  return [...freq.entries()].sort((a,b)=>b[1]-a[1]).slice(0,k).map(([w])=>w);
}

function caps(s = "") {
  return s.split(/\s+/).map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(" ");
}

function deriveNameFromText(...candidates) {
  const text = candidates.filter(Boolean).join(" ");
  const kws = topK(text, 3);
  if (!kws.length) return text.split(" ").slice(0,3).join(" ") || "Unknown Data";
  return caps(kws.join(" "));
}

function sanitizeType(t) {
  if (isGarbage(t)) return "Other";
  const lower = String(t).toLowerCase();
  if (/(email|mail)/.test(lower)) return "Identifier";
  if (/(phone|tel)/.test(lower)) return "Identifier";
  if (/(location|geo|city|country|address)/.test(lower)) return "Location";
  if (/(credential|password|secret|token|apikey|api key)/.test(lower)) return "Credentials";
  if (/(usage|analytics|log|telemetry|events?)/.test(lower)) return "Usage";
  return caps(t.replace(/[^A-Za-zğüşöçıİĞÜŞÖÇ\s]/g," ").replace(/\s+/g," ").trim());
}

/** normalize a raw item */
function normalizeItem(x) {
  const rawName = x?.data_name ?? x?.name;
  const rawType = x?.question_type ?? x?.data_type;
  const desc = x?.question_desc ?? x?.description ?? x?.explanation ?? "";

  const derivedName = deriveNameFromText(rawName, desc);
  const cleanName = isGarbage(rawName)
    ? derivedName
    : caps(String(rawName).replace(/[_-]+/g," ").trim());

  const cleanType = sanitizeType(rawType);

  let snippet = safeText(x?.description || x?.question_desc || x?.explanation, "");
  snippet = snippet.replace(/\s+#\s*/g, " ").replace(/\n{2,}/g, "\n");
  if (snippet.length > 400) snippet = snippet.slice(0, 400) + "…";

  return {
    ...x,
    __name: x?.norm_data_name || cleanName || "Unknown Data",
    __type: x?.norm_data_type || cleanType || "Other",
    __nameDerived: isGarbage(rawName) || Boolean(x?.name_derived),
    __typeDerived: isGarbage(rawType) || Boolean(x?.type_derived),
    __snippet: snippet,
  };
}

/* --------- AI explanation helper --------- */
async function fetchAIExplanation(API_BASE, cur) {
  const payload = {
    data_name: cur?.data_name || cur?.__name || "",
    data_type: cur?.data_type || cur?.__type || "",
    description: cur?.description || cur?.question_desc || cur?.__snippet || "",
    label: cur?.label || "",
    lang: "en",
  };
  const r = await fetch(`${API_BASE}/game/explain`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!r.ok) throw new Error(`/game/explain failed: ${r.status}`);
  const j = await r.json();
  return j?.explanation || "";
}

/* ---------- Loading Overlay Component ---------- */
function LoadingOverlay({ text = "Loading dataset…", error, onRetry }) {
  return (
    <div className="loader-overlay">
      <div className="spinner" />
      <div className="loading-text">{text}</div>
      {error ? (
        <div className="error-box">
          {String(error)}
          <div style={{ marginTop: 10, textAlign: "right" }}>
            <button className="head-btn" onClick={onRetry}>Retry</button>
          </div>
        </div>
      ) : null}
    </div>
  );
}

export default function Game({ API_BASE, onBack }) {
  // Game gating
  const [started, setStarted] = useState(false);
  const [isLoading, setIsLoading] = useState(false); // NEW
  const [loadError, setLoadError] = useState(null);   // NEW

  const [items, setItems] = useState([]);
  const [limit, setLimit] = useState(() => {
    try { return localStorage.getItem("game_limit") || "10"; } catch { return "10"; }
  });
  const [i, setI] = useState(0);
  const [answered, setAnswered] = useState(null); // "CLEAR" | "OMITTED" | "AMBIGUOUS" | null
  const [correct, setCorrect] = useState(0);
  const [wrong, setWrong] = useState(0);
  const [flipped, setFlipped] = useState(false);
  const [showSummary, setShowSummary] = useState(false);
  const [showIntro, setShowIntro] = useState(true);

  const cardRef = useRef(null);
  const barRef = useRef(null);

  useEffect(() => injectOnce("game-css", gameCSS), []);

  const pct = useMemo(() => {
    const total = items.length || 1;
    return Math.round((i / total) * 100);
  }, [i, items.length]);

  useEffect(() => {
    if (barRef.current) barRef.current.style.width = pct + "%";
  }, [pct]);

  function shuffle(arr) {
    const a = [...arr];
    for (let k = a.length - 1; k > 0; k--) {
      const j = Math.floor(Math.random() * (k + 1));
      [a[k], a[j]] = [a[j], a[k]];
    }
    return a;
  }
  function applyLimit(arr) {
    if (!arr || arr.length === 0) return [];
    if (limit === "all") return arr;
    const n = Math.max(1, parseInt(limit, 10) || 10);
    return arr.slice(0, Math.min(n, arr.length));
  }

  // Weighted order
  function makeWeighted(all) {
    const allowed = all.filter(x => ["CLEAR","AMBIGUOUS","OMITTED"].includes(x.label));
    const omitted = shuffle(allowed.filter(x => x.label === "OMITTED"));
    const clear = shuffle(allowed.filter(x => x.label === "CLEAR"));
    const ambiguous = shuffle(allowed.filter(x => x.label === "AMBIGUOUS"));
    const non = shuffle([...clear, ...ambiguous]);

    const out = [];
    let oi = 0, ni = 0;
    while (oi < omitted.length || ni < non.length) {
      const run = 2 + Math.floor(Math.random() * 3); // 2..4
      let pushed = 0;
      while (pushed < run && oi < omitted.length) { out.push(omitted[oi++]); pushed++; }
      if (ni < non.length) out.push(non[ni++]);
      if (ni >= non.length && oi < omitted.length) {
        const restRun = 2 + Math.floor(Math.random() * 3);
        let r = 0; while (r < restRun && oi < omitted.length) { out.push(omitted[oi++]); r++; }
      }
    }
    return out;
  }

  async function load() {
    setIsLoading(true);
    setLoadError(null);
    try {
      const r = await fetch(`${API_BASE}/game/data`, { cache: "no-store" });
      if (!r.ok) throw new Error(`/game/data failed: ${r.status}`);
      const j = await r.json();
      const all = (j.items || []).map(normalizeItem);
      const weighted = makeWeighted(all);
      setItems(applyLimit(weighted));
      setI(0);
      setAnswered(null);
      setCorrect(0);
      setWrong(0);
      setFlipped(false);
      setShowSummary(false);
      cardRef.current?.classList.remove("is-back");
    } catch (e) {
      console.error(e);
      setItems([]);
      setLoadError(e.message || String(e));
    } finally {
      setIsLoading(false);
    }
  }

  // After start, load dataset
  useEffect(() => {
    if (!started) return;
    load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [limit, started]);

  async function answer(choice) {
    if (!started || isLoading) return;
    if (answered) return;
    const cur = items[i];
    if (!cur) return;

    setAnswered(choice);
    if (choice === cur.label) setCorrect(c => c + 1);
    else setWrong(w => w + 1);
    setFlipped(true);
    cardRef.current?.classList.add("is-back");

    // Fetch AI explanation (non-blocking)
    setItems(prev => {
      const arr = [...prev];
      arr[i] = { ...arr[i], __aiLoading: true };
      return arr;
    });
    try {
      const exp = await fetchAIExplanation(API_BASE, cur);
      setItems(prev => {
        const arr = [...prev];
        arr[i] = { ...arr[i], __aiLoading: false, __aiExplanation: exp };
        return arr;
      });
    } catch (e) {
      console.error(e);
      setItems(prev => {
        const arr = [...prev];
        arr[i] = { ...arr[i], __aiLoading: false, __aiExplanation: "" };
        return arr;
      });
    }
  }

  function nextCard() {
    if (i < items.length - 1) {
      setI(i + 1);
      setAnswered(null);
      setFlipped(false);
      cardRef.current?.classList.remove("is-back");
    } else {
      setShowSummary(true);
    }
  }

  useEffect(() => {
    const onKey = (e) => {
      if (!started || isLoading) return;
      const k = e.key.toLowerCase();
      if (k === "c") answer("CLEAR");
      else if (k === "a") answer("AMBIGUOUS");
      else if (k === "o") answer("OMITTED");
      else if (k === "n") nextCard();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [i, answered, items, started, isLoading]);

  const cur = items[i];
  const pretty = { CLEAR: "CLEAR", OMITTED: "OMITTED", AMBIGUOUS: "AMBIGUOUS" };
  const accuracy = (() => {
    const total = correct + wrong;
    return total ? Math.round((correct * 100) / total) : 0;
  })();

  return (
    <div className="game-root">
      <div className="game-wrap">
        <header className="game-header">
          <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <strong style={{ fontSize: 18 }}>Privacy Policy Card Game</strong>
            {started && !isLoading && (
              <span className="tinycaps" style={{ color: "var(--g-muted)", marginLeft: 8 }}>
                {items.length ? `Progress ${i}/${items.length}` : "Loading…"}
              </span>
            )}
          </div>
          <div className="controls">
            <span className="tinycaps" style={{ marginRight: 4, color: "var(--g-muted)" }}>Cards</span>
            <select
              className="select"
              value={limit}
              onChange={(e) => { const v = e.target.value; setLimit(v); try { localStorage.setItem("game_limit", v); } catch {} }}
              disabled={!started || isLoading}
            >
              <option value="5">5</option>
              <option value="10">10</option>
              <option value="15">15</option>
              <option value="20">20</option>
              <option value="30">30</option>
              <option value="50">50</option>
              <option value="all">All</option>
            </select>

            <button className="head-btn" onClick={() => started && load()} disabled={!started || isLoading}>Apply</button>
            <button className="head-btn" onClick={() => { if (started && !isLoading) { setI(items.length - 1); setShowSummary(true); } }} disabled={!started || isLoading}>Finish</button>
            <button className="head-btn" onClick={() => started && load()} disabled={!started || isLoading}>Restart</button>
            <button className="head-btn" onClick={onBack} disabled={isLoading}>← Back to Chat</button>
          </div>
        </header>

        {/* Main area */}
        {started && !isLoading && (
          <main className="game-shell">
            <div className="card-wrapper">
              <div className="progress"><div className="bar" ref={barRef} /></div>

              <div className={`card ${flipped ? "is-back" : ""}`} ref={cardRef}>
                <div className="card-inner">
                  {/* FRONT */}
                  <section className="face front">
                    <div className="tinycaps">DATA CARD</div>

                    <div
                      style={{
                        flex: 1,
                        display: "flex",
                        flexDirection: "column",
                        justifyContent: "center",
                        alignItems: "center",
                        textAlign: "center",
                        padding: "20px",
                        gap: "20px",
                        width: "100%",
                        maxWidth: "90%",
                        margin: "0 auto"
                      }}
                    >
                      {/* Data Name */}
                      <div style={{ textAlign: "left", width: "100%" }}>
                        <div className="lbl" style={{ fontSize: 12, color: "var(--g-muted)", textTransform: "uppercase", letterSpacing: ".08em" }}>
                          Data Name
                        </div>
                        <div style={{ fontSize: 22, fontWeight: 800, color: "var(--g-text)", lineHeight: 1.3 }}>
                          {safeText(cur?.data_name || cur?.__name || cur?.name || "Unknown Data")}
                        </div>
                      </div>

                      {/* Description */}
                      <div style={{ textAlign: "left", width: "100%" }}>
                        <div className="lbl" style={{ fontSize: 12, color: "var(--g-muted)", textTransform: "uppercase", letterSpacing: ".08em" }}>
                          Description
                        </div>
                        <div
                          style={{
                            fontSize: 18,
                            fontWeight: 400,
                            color: "var(--g-text)",
                            lineHeight: 1.6,
                            background: "rgba(255,255,255,0.04)",
                            borderRadius: 10,
                            padding: "12px 16px",
                            border: "1px solid var(--g-border)",
                            whiteSpace: "normal"
                          }}
                        >
                          {safeText(cur?.description || cur?.question_desc || cur?.__snippet || "(no description)")}
                        </div>
                      </div>
                    </div>

                    {/* Choices */}
                    <div className="choices" style={{ marginTop: "auto" }}>
                      <button className="btn b-green" onClick={() => answer("CLEAR")}>Clear</button>
                      <button className="btn b-amber" onClick={() => answer("AMBIGUOUS")}>Ambiguous</button>
                      <button className="btn b-red" onClick={() => answer("OMITTED")}>Omitted</button>
                    </div>
                  </section>

                  {/* BACK */}
                  <section className="face back" style={{ textAlign: "center", justifyContent: "center" }}>
                    <div className="tinycaps">Correct Answer</div>
                    <h2 className="result" style={{ marginTop: 12 }}>
                      <span className={`tag t-${cur?.label || "AMBIGUOUS"}`}>
                        {pretty[cur?.label] || cur?.label || "AMBIGUOUS"}
                      </span>
                    </h2>
                    <div style={{ fontSize: 18, marginTop: 12 }}>
                      {answered === cur?.label ? "✅ Correct!" : "❌ Wrong!"}
                    </div>

                    <div className="badge" style={{ marginTop: 18, textAlign: "left" }}>
                      <div className="lbl">Why this label?</div>
                      <div className="explanation" style={{ minHeight: 60 }}>
                        {cur?.__aiLoading
                          ? "Generating a short explanation…"
                          : safeText(cur?.__aiExplanation, "(no AI explanation)")}
                      </div>
                    </div>

                    <button className="head-btn" style={{ marginTop: 24 }} onClick={nextCard}>
                      Next Card ▶
                    </button>
                  </section>
                </div>
              </div>

              <div className="tinycaps meta"> Progress {i}/{items.length}</div>
            </div>
          </main>
        )}
      </div>

      {/* INTRO */}
      <div className={`modal ${showIntro ? "show" : ""}`}>
        <div className="card-modal">
          <h2 className="modal-title">How to Play</h2>
          <p style={{opacity:.9}}>
            Each card shows a <b>Data Name</b> and a <b>Description</b> from a real privacy policy. Decide whether the policy covers this data clearly, vaguely, or not at all.
          </p>
          <ul style={{lineHeight:1.6, marginTop:8}}>
            <li>🟢 <b>CLEAR</b> – The policy <u>explicitly</u> states this data is collected/used.</li>
            <li>🟠 <b>AMBIGUOUS</b> – The policy might imply it, but it’s vague or non-specific.</li>
            <li>🔴 <b>OMITTED</b> – The data appears in practice, but is <u>not mentioned</u> in the policy.</li>
          </ul>
          <div style={{ display: "flex", gap: 10, justifyContent: "flex-end", marginTop: 14 }}>
            <button
              className="head-btn"
              onClick={() => {
                setShowIntro(false);
                setStarted(true);  // Start
              }}
            >
              Start ▶
            </button>
          </div>
        </div>
      </div>

      {/* SUMMARY */}
      <div className={`modal ${showSummary ? "show" : ""}`}>
        <div className="card-modal">
          <h2 className="modal-title">Session Summary</h2>
          <div style={{ color: "var(--g-muted)" }}>Total cards: {items.length}</div>
          <div className="stats">
            <div className="stat"><div className="lbl">Correct</div><b>{correct}</b></div>
            <div className="stat"><div className="lbl">Wrong</div><b>{wrong}</b></div>
            <div className="stat"><div className="lbl">Accuracy</div><b>{accuracy}%</b></div>
          </div>
          <div style={{ display: "flex", gap: 10, justifyContent: "flex-end" }}>
            <button className="head-btn" onClick={() => setShowSummary(false)}>Close</button>
            <button className="head-btn" onClick={() => { setShowSummary(false); load(); }}>Play Again</button>
            <button className="head-btn" onClick={() => { setShowSummary(false); setStarted(false); setItems([]); setShowIntro(true); }}>Exit to Chat</button>
          </div>
        </div>
      </div>

      {/* FULLSCREEN LOADER — gösterim koşulu */}
      {started && isLoading && (
        <LoadingOverlay
          text="Loading dataset…"
          error={loadError}
          onRetry={() => load()}
        />
      )}
    </div>
  );
}
