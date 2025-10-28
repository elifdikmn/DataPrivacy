// App.js — Full-screen Chat (no sidebar/new chat) + Hash back/forward + Slide-up transitions
// Entegre: glass/nice-scroll/typing-dots + DİKEY "Hazır Sorular" (üstte)
// Bu sürüm: viz route fix + backend viz_path kullanımı + Game view + Play the Game butonu

import React, { useState, useEffect, useRef } from "react";
import Game from "./Game";



/* ---------- Safe API base resolution ---------- */
function resolveApiBase() {
  try { if (typeof window !== "undefined" && window.APP_API_BASE) return String(window.APP_API_BASE); } catch(_) {}
  try {
    if (typeof window !== "undefined") {
      const u = new URL(window.location.href);
      const qp = u.searchParams.get("api");
      if (qp) return qp;
    }
  } catch(_) {}
  try {
    if (typeof window !== "undefined") {
      // eslint-disable-next-line no-undef
      if (process?.env?.REACT_APP_API_BASE) return process.env.REACT_APP_API_BASE;
    }
  } catch(_) {}
  try {
    // eslint-disable-next-line no-new-func
    const viteVal = new Function(
      "try { return import.meta && import.meta.env && import.meta.env.VITE_API_BASE } catch(e) { return null }"
    )();
    if (viteVal) return viteVal;
  } catch(_) {}
  try {
    if (typeof localStorage !== "undefined") {
      const ls = localStorage.getItem("API_BASE");
      if (ls) return ls;
    }
  } catch(_) {}
  return "http://127.0.0.1:8000";
}
const API_BASE = resolveApiBase();

/* ---------- Theme (LIGHT for chat) ---------- */
const theme = {
  primary: "#7c3aed",
  primary2: "#3b82f6",
  bg: "#ffffff",           // chat background
  border: "rgba(0,0,0,.10)",
  text: "#0f172a",
  userBubble: "#f3f4f6",
  botBubble: "#f8fafc",
};

/* ---------- Suggested questions with viz routes ---------- */
const suggested = [
  { label: "What data are collected by GPT Actions?", viz: "/viz/main" },
  { label: "Which sensitive data types appear most often?", viz: "/viz/hier" }, // backend ile hizalı
];

/* ---------- Views (hash-based) ---------- */
const VIEW_INTRO = "intro";
const VIEW_CHAT = "chat";
const VIEW_GAME = "game";

function getInitialViewFromHash() {
  if (typeof window === "undefined") return VIEW_INTRO;
  if (window.location.hash === "#chat") return VIEW_CHAT;
  if (window.location.hash === "#game") return VIEW_GAME;
  return VIEW_INTRO;
}

/* ---------- Global CSS ---------- */
const globalCSS = `
@keyframes pageEnterUp { from { opacity: 0; transform: translateY(32px); } to { opacity: 1; transform: translateY(0); } }
@keyframes pageEnterUpSlow { from { opacity: 0; transform: translateY(60px); } to { opacity: 1; transform: translateY(0); } }
.page-enter-up { animation: pageEnterUp 480ms cubic-bezier(.2,.8,.2,1) both; }
.page-enter-up-slow { animation: pageEnterUpSlow 2000ms cubic-bezier(.22,.8,.22,1) both; }
@media (prefers-reduced-motion: reduce) { .page-enter-up, .page-enter-up-slow { animation: none !important; } }

/* Light glass look */
.glass {
  background: linear-gradient(180deg, rgba(255,255,255,0.65), rgba(255,255,255,0.55));
  border: 1px solid rgba(0,0,0,0.08);
  backdrop-filter: blur(8px);
  box-shadow: 0 8px 28px rgba(0,0,0,0.08);
  border-radius: 16px;
}

/* Nice scroll (light) */
.nice-scroll::-webkit-scrollbar { width: 10px; }
.nice-scroll::-webkit-scrollbar-thumb { background: rgba(0,0,0,.20); border-radius: 20px; border: 2px solid transparent; }
.nice-scroll { scrollbar-color: rgba(0,0,0,.20) transparent; }

.dot { animation: dotBlink 1.2s infinite ease-in-out; display:inline-block; }
.dot:nth-child(2){ animation-delay:.2s }
.dot:nth-child(3){ animation-delay:.4s }
@keyframes dotBlink { 0%, 80%, 100% { opacity:.35; transform: translateY(0) } 40% { opacity:1; transform: translateY(-1px) } }
`;

function App() {
  /* inject CSS once */
  useEffect(() => {
    if (!document.getElementById("global-css-injected")) {
      const s = document.createElement("style");
      s.id = "global-css-injected";
      s.innerHTML = globalCSS;
      document.head.appendChild(s);
    }
  }, []);

  const [view, setView] = useState(getInitialViewFromHash());
  const [transitionKey, setTransitionKey] = useState(0);
  const [enterClass, setEnterClass] = useState(
    getInitialViewFromHash() === VIEW_CHAT ? "page-enter-up-slow" : "page-enter-up"
  );

  // Tek konuşma
  const [conversations, setConversations] = useState([
    { id: 1, messages: [{ from: "bot", text: "Hello! Ask me anything about Data Privacy.", typing: false }] },
  ]);
  const [activeConvId] = useState(1);
  const [input, setInput] = useState("");

  const [showViz, setShowViz] = useState(false);     // başlangıçta kapalı
  const [vizSrc, setVizSrc] = useState("");          // hangi viz?
  const [vizReady, setVizReady] = useState(false);   // iframe hazır mı?

  const [showIntroSuggestions, setShowIntroSuggestions] = useState(true); // üstteki ilk görünüm
  const [remainingSuggestions, setRemainingSuggestions] = useState(suggested); // kalanlar
  const [showBottomSuggestions, setShowBottomSuggestions] = useState(false);   // cevap geldikten sonra altta göster
  const chatEndRef = useRef(null);

  const activeConversation = conversations.find((c) => c.id === activeConvId);

  /* ---------- Navigation ---------- */
  function navigate(nextView) {
    setEnterClass(nextView === VIEW_CHAT ? "page-enter-up-slow" : "page-enter-up");
    setView(nextView);
    setTransitionKey((k) => k + 1);
    if (typeof window !== "undefined") {
      const hash =
        nextView === VIEW_CHAT ? "#chat" :
        nextView === VIEW_GAME ? "#game" : "#intro";
      if (window.location.hash !== hash) {
        window.history.pushState({ view: nextView }, "", hash);
      }
    }
  }
  useEffect(() => {
    const onPop = () => {
      const v =
        window.location.hash === "#chat"
          ? VIEW_CHAT
          : window.location.hash === "#game"
          ? VIEW_GAME
          : VIEW_INTRO;
      setEnterClass(v === VIEW_CHAT ? "page-enter-up-slow" : "page-enter-up");
      setView(v);
      setTransitionKey((k) => k + 1);
    };
    window.addEventListener("popstate", onPop);
    return () => window.removeEventListener("popstate", onPop);
  }, []);

  /* ---------- Auto-scroll only inside messages ---------- */
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [activeConversation?.messages]);

  /* ---------- Persist (optional) ---------- */
  useEffect(() => { try { const s = localStorage.getItem("conversations"); if (s) setConversations(JSON.parse(s)); } catch(_) {} }, []);
  useEffect(() => { try { localStorage.setItem("conversations", JSON.stringify(conversations)); } catch(_) {} }, [conversations]);

  /* ---------- Helpers ---------- */
  function openVizFor(src) {
    // Cevap geldikten sonra panel açılacak; burada sadece hedefi not ediyoruz
    setVizSrc(src);
    setVizReady(false);
  }

  /* ---------- Messaging ---------- */
  const sendMessage = async (text, opts = {}) => {
    const viaSuggestion = !!opts.viaSuggestion;
    const clean = (text || "").trim();
    if (!clean) return;

    // Hazır sorudan geldiyse: üst listeden kaldır ve gizle
    if (viaSuggestion) {
      setRemainingSuggestions(prev => prev.filter(q => q.label !== text));
      setShowIntroSuggestions(false);
      setShowBottomSuggestions(false);
    }

    // user message
    setConversations(prev => prev.map(conv =>
      conv.id === activeConvId
        ? { ...conv, messages: [...conv.messages, { from: "user", text: clean, viaSuggestion }] }
        : conv
    ));
    setInput("");

    // typing
    const typingId = `typing-${Date.now()}`;
    setConversations(prev => prev.map(conv =>
      conv.id === activeConvId
        ? { ...conv, messages: [...conv.messages, { from: "bot", text: "", typing: true, id: typingId }] }
        : conv
    ));

    // call backend
    const ctrl = new AbortController();
    const timeout = setTimeout(() => ctrl.abort(), 10000);
    try {
      const res = await fetch(`${API_BASE}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: clean }),
        signal: ctrl.signal,
      });
      clearTimeout(timeout);
      if (!res.ok) {
        const t = await res.text().catch(() => "");
        throw new Error(`HTTP ${res.status} ${res.statusText} ${t || ""}`.trim());
      }
      const data = await res.json().catch(() => ({}));
      const txt = data?.text || "No response";

      // replace typing with real answer + altta önerileri aç
      setConversations(prev => prev.map(conv => {
        if (conv.id !== activeConvId) return conv;
        const msgs = conv.messages.filter(m => !m.typing);
        const lastUser = [...msgs].reverse().find(m => m.from === "user");
        const suggestedAns = Boolean(lastUser?.viaSuggestion);
        return { ...conv, messages: [...msgs, { from: "bot", text: txt, suggestedAns }] };
      }));

      // Cevap geldikten sonra: kalan soruları altta göster
      if (viaSuggestion && remainingSuggestions.length > 0) {
        setShowBottomSuggestions(true);
      }

      // Cevap geldikten sonra viz panelini aç
      if (viaSuggestion) {
        // Backend viz_path önerdiyse onu kullan
        if (data?.viz_path) {
          const cb = Date.now(); // cache-buster
          setVizSrc(`/viz/${data.viz_path}?cb=${cb}`);
        }
        setShowViz(true);
        setVizReady(false);
        setTimeout(() => setVizReady(true), 150);
        setTimeout(() => {
          try {
            document.querySelector("aside.viz-panel")?.scrollIntoView({ behavior: "smooth", block: "nearest" });
          } catch (_) {}
        }, 180);
      }

      // Backend viz istemiyorsa paneli kapat
      if (viaSuggestion && data?.viz_relevant === false) {
        setShowViz(false);
      }
    } catch (e) {
      clearTimeout(timeout);
      setConversations(prev => prev.map(conv =>
        conv.id === activeConvId
          ? { ...conv, messages: [...conv.messages.filter(m => !m.typing), { from: "bot", text: `Error: ${e?.message || "Request failed."}` }] }
          : conv
      ));
    }
  };

  /* ---------- Small UI atoms ---------- */
  const TypingDots = () => (
    <span style={{ color: theme.text }}>
      <span className="dot">●</span><span className="dot">●</span><span className="dot">●</span>
    </span>
  );

  const Bubble = ({ from, children }) => {
    const isUser = from === "user";
    return (
      <div
        style={{
          maxWidth: "82%",
          background: isUser ? theme.userBubble : theme.botBubble,
          color: theme.text,
          padding: "12px 16px",
          borderRadius: 18,
          borderTopRightRadius: isUser ? 10 : 18,
          borderTopLeftRadius: isUser ? 18 : 10,
          border: `1px solid ${theme.border}`,
          lineHeight: 1.6,
          fontSize: 16,
        }}
      >
        {children}
      </div>
    );
  };

  /* ---------- Game View ---------- */
  if (view === VIEW_GAME) {
    return (
      <div key={transitionKey} className={enterClass} style={{ height: "100vh" }}>
        <Game API_BASE={API_BASE} onBack={() => navigate(VIEW_CHAT)} />
      </div>
    );
  }
  

  /* ---------- Intro (DARK) ---------- */
  if (view === VIEW_INTRO) {
    return (
      <div
        key={transitionKey}
        className={enterClass}
        style={{
          height: "100vh",
          display: "grid",
          placeItems: "center",
          background: `radial-gradient(900px 500px at 10% -10%, ${theme.primary2}33 0%, transparent 60%), radial-gradient(1000px 600px at 110% 10%, ${theme.primary}33 0%, transparent 60%), linear-gradient(180deg, #0b1224, #0f172a)`,
          color: "#e5e7eb",
          fontFamily: "Inter, system-ui, Arial, sans-serif",
          padding: 24,
          boxSizing: "border-box",
        }}
      >
        <div style={{ maxWidth: 1500, width: "100%", textAlign: "center", marginTop: -200 }}>
          <h1 style={{ fontSize: 70, fontWeight: 900, marginBottom: 16 }}>Data Privacy Assistant</h1>
          <p style={{ maxWidth: 820, margin: "0 auto 28px", fontSize: 20, lineHeight: 1.9, color: "#bfdbfe" }}>
          
          Is your personal data really safe? This assistant makes it simple. Just click on a question to get clear answers about data privacy. 
          Then explore the interactive visualization to see which data types appear most often and helps you stay aware and in control of your information.
          </p>
           {/* Start Exploring button */}
        <button
          onClick={() => navigate(VIEW_CHAT)}
          style={{
            padding: "14px 22px",
            borderRadius: 14,
            border: "1px solid rgba(255,255,255,.25)",
            background: "linear-gradient(135deg, #7c3aed, #3b82f6)",
            color: "white",
            fontWeight: 800,
            fontSize: 16,
            cursor: "pointer",
            boxShadow: "0 14px 32px rgba(59,130,246,.35)",
          }}
          aria-label="Start exploring"
          title="Start exploring"
        >
          Start exploring
        </button>
        </div>
      </div>
    );
  }

  /* ---------- Chat UI (light) ---------- */
  return (
    <div
      key={transitionKey}
      className={enterClass}
      style={{
        display: "grid",
        gridTemplateColumns: showViz ? "1fr 1fr" : "1fr",
        gap: 0,
        height: "100vh",
        background: theme.bg,
        color: theme.text,
        fontFamily: "Inter, system-ui, Arial, sans-serif",
        padding: 16,
        boxSizing: "border-box",
        overflow: "hidden",
      }}
    >
      {/* LEFT: Messages */}
      <main
        
        style={{
          height: "100%",
          display: "grid",
          gridTemplateRows: "auto 1fr auto", // üst (hazır sorular) / ortada mesajlar / altta composer
          padding: 16,
          minHeight: 0, // iç scroll için kritik
          background: "#fff", 
        }}
      >
        {/* Hazır Sorular — üstte, DİKEY, BÜYÜK butonlar (yalnızca ilk ekranda) */}
        {showIntroSuggestions && remainingSuggestions.length > 0 && (
          <section style={{ marginBottom: 12 }}>
            <div style={{ display: "grid", rowGap: 10 }}>
              {remainingSuggestions.map((q) => (
                <button
                  key={q.label}
                  onClick={() => { openVizFor(q.viz.startsWith("/") ? `${q.viz}` : `/${q.viz}`); sendMessage(q.label, { viaSuggestion: true }); }}
                  style={{
                    textAlign: "left",
                    padding: "16px 18px", // büyük
                    borderRadius: 14,
                    border: `1px solid ${theme.border}`,
                    background: "#ffffff",
                    color: theme.text,
                    cursor: "pointer",
                    fontSize: 18,         // büyük
                    fontWeight: 600,
                  }}
                >
                  {q.label}
                </button>
              ))}
            </div>
          </section>
        )}

        {/* Messages (iç scroll) */}
        <section className="nice-scroll" style={{ overflowY: "auto", paddingRight: 6, minHeight: 0 }}>
          {activeConversation.messages.map((m, i) => (
            <div key={i} style={{ display: "flex", gap: 10, marginBottom: 10, alignItems: "flex-start" }}>
              <div
                style={{
                  width: 32, height: 32, borderRadius: 10,
                  display: "grid", placeItems: "center",
                  background: m.from === "user" ? "#bae6fd" : "#bbf7d0",
                  color: "#0b1020", fontSize: 16, fontWeight: 700, flexShrink: 0,
                }}
                aria-label={m.from === "user" ? "User" : "Assistant"}
                title={m.from === "user" ? "User" : "Assistant"}
              >
                {m.from === "user" ? "🧑" : "🤖"}
              </div>
              <Bubble from={m.from}>
                {m.typing ? <TypingDots /> : m.text}
              </Bubble>
            </div>
          ))}

          {/* Cevap geldikten sonra kalan sorular MESAJLARIN ALTINDA */}
          {showBottomSuggestions && remainingSuggestions.length > 0 && (
            <div style={{ marginTop: 20 }}>
              {remainingSuggestions.map(q => (
                <button
                  key={q.label}
                  onClick={() => { openVizFor(q.viz.startsWith("/") ? `${q.viz}` : `/${q.viz}`); sendMessage(q.label, { viaSuggestion: true }); }}
                  style={{
                    display: "block",
                    width: "100%",
                    textAlign: "left",
                    padding: "16px 18px",
                    marginBottom: 8,
                    borderRadius: 12,
                    fontSize: 18,
                    fontWeight: 600,
                    background: "#fff",
                    border: `1px solid ${theme.border}`,
                    cursor: "pointer"
                  }}
                >
                  {q.label}
                </button>
              ))}
            </div>
          )}

          <div ref={chatEndRef} />
        </section>

        {/* Play the Game — tüm sorular bitince */}
        {remainingSuggestions.length === 0 && !showIntroSuggestions && (
          <div style={{ display: "flex", justifyContent: "center", marginTop: 8 }}>
            <button
              onClick={() => navigate(VIEW_GAME)}
              style={{
                padding: "12px 18px",
                borderRadius: 12,
                border: `1px solid ${theme.border}`,
                background: "linear-gradient(135deg, #10b981, #3b82f6)",
                color: "white",
                fontWeight: 800,
                cursor: "pointer",
                boxShadow: "0 10px 24px rgba(16,185,129,.25)",
              }}
            >
              ▶ Play the Game
            </button>
          </div>
        )}

        {/* Composer */}
        <div style={{ display: "flex", gap: 10, marginTop: 12, alignItems: "flex-end" }}>
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendMessage(input); } }}
            placeholder="Type your question."
            rows={1}
            style={{
              flex: 1,
              padding: "12px 16px",
              borderRadius: 16,
              border: `1px solid ${theme.border}`,
              background: "#ffffff",
              color: theme.text,
              outline: "none",
              resize: "none",
              minHeight: 48,
              maxHeight: 180,
            }}
          />
          <button
            onClick={() => sendMessage(input)}
            style={{
              padding: "12px 16px",
              borderRadius: 12,
              border: `1px solid ${theme.border}`,
              background: "linear-gradient(135deg, #7c3aed, #3b82f6)",
              color: "white",
              fontWeight: 800,
              cursor: "pointer",
              boxShadow: "0 10px 24px rgba(124,58,237,.25)",
            }}
          >
            Send
          </button>
        </div>
      </main>

      {/* RIGHT: Visualization (compact, no header) */}
{showViz && (
  <aside
    className="viz-panel"
    style={{
      height: "100%",
      padding: 16,                // daha az boşluk
      display: "flex",
      flexDirection: "column",
      minWidth: 0,
      background: "transparent", // çerçevesiz
    }}
  >
    <div
      style={{
        position: "relative",
        flex: 1,
        overflow: "hidden",
        background: "#fff",
        border: "none",
        borderRadius: 0,
        minHeight: 0,
      }}
    >
      {/* Üst sağ köşede küçük Hide butonu (başlık yok) */}
      {/* Üst sağ köşede küçük Back/Forward + Hide (başlık yok) */}
{/* --- VIZ HISTORY SYNC: vizSrc her değiştiğinde hafif bir bellek tutar --- */}
{(() => {
  if (typeof window !== "undefined") {
    const H = (window.__VIZ_HIST__ ||= { list: [], idx: -1 });
    if (vizSrc && H.list[H.idx] !== vizSrc) {
      // ileriye doğru dalı kes, yeni src'yi ekle
      H.list = H.list.slice(0, H.idx + 1).concat(vizSrc);
      H.idx = H.list.length - 1;
    }
  }
  return null;
})()}

<div style={{ position: "absolute", top: 16, right: 16, zIndex: 2, display: "flex", gap: 8 }}>
  {/* Geri */}
  <button
    onClick={() => {
      const H = window.__VIZ_HIST__;
      if (!H) return;
      const next = Math.max(H.idx - 1, 0);
      if (next !== H.idx) {
        H.idx = next;
        const s = H.list[next];
        if (s) {
          setVizReady(false);
          setVizSrc(s);
          setTimeout(() => setVizReady(true), 120);
          setShowViz(true);
        }
      }
    }}
    disabled={!window.__VIZ_HIST__ || window.__VIZ_HIST__.idx <= 0}
    style={{
      padding: "6px 10px",
      borderRadius: 10,
      border: "1px solid rgba(0,0,0,.10)",
      background:
        window.__VIZ_HIST__ && window.__VIZ_HIST__.idx > 0 ? "#ffffff" : "#f3f4f6",
      color: "#0f172a",
      cursor:
        window.__VIZ_HIST__ && window.__VIZ_HIST__.idx > 0 ? "pointer" : "not-allowed",
      boxShadow: "0 2px 8px rgba(0,0,0,.06)",
    }}
    aria-label="Back visualization"
    title="Back"
  >
    ←
  </button>

  {/* İleri */}
  <button
    onClick={() => {
      const H = window.__VIZ_HIST__;
      if (!H) return;
      const next = Math.min(H.idx + 1, H.list.length - 1);
      if (next !== H.idx) {
        H.idx = next;
        const s = H.list[next];
        if (s) {
          setVizReady(false);
          setVizSrc(s);
          setTimeout(() => setVizReady(true), 120);
          setShowViz(true);
        }
      }
    }}
    disabled={
      !window.__VIZ_HIST__ ||
      !window.__VIZ_HIST__.list ||
      window.__VIZ_HIST__.idx >= window.__VIZ_HIST__.list.length - 1
    }
    style={{
      padding: "6px 10px",
      borderRadius: 10,
      border: "1px solid rgba(0,0,0,.10)",
      background:
        window.__VIZ_HIST__ &&
        window.__VIZ_HIST__.list &&
        window.__VIZ_HIST__.idx < window.__VIZ_HIST__.list.length - 1
          ? "#ffffff"
          : "#f3f4f6",
      color: "#0f172a",
      cursor:
        window.__VIZ_HIST__ &&
        window.__VIZ_HIST__.list &&
        window.__VIZ_HIST__.idx < window.__VIZ_HIST__.list.length - 1
          ? "pointer"
          : "not-allowed",
      boxShadow: "0 2px 8px rgba(0,0,0,.06)",
    }}
    aria-label="Forward visualization"
    title="Forward"
  >
    →
  </button>

  
</div>


      {/* Loader */}
      {vizSrc && !vizReady && (
        <div style={{ height: "100%", display: "grid", placeItems: "center" }}>
          <div style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 16 }}>
            Loading visualization <span className="dot">●</span><span className="dot">●</span><span className="dot">●</span>
          </div>
        </div>
      )}

      {/* Iframe — başlık yok, direkt görsel en üste yapışık */}
      {vizSrc && vizReady && (
        <iframe
          title="Visualization"
          src={`${API_BASE}${vizSrc}`}
          style={{ width: "100%", height: "100%", border: 0, display: "block" }}
          sandbox="allow-scripts allow-same-origin"
        />
      )}
    </div>
  </aside>
)}
    </div>
  );
}

export default App;
