import React, { useEffect, useState } from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import CategoryChart from './CategoryChart';
import './App.css';

const API_BASE = process.env.REACT_APP_API_BASE || 'http://127.0.0.1:8000';

const SUGGESTED_QUESTIONS = [
  'What data are collected by GPT Actions?',
  'What percentage of collected data is sensitive?',
  'Which sensitive data types appear most often?',
  'Do plugins write descriptions less often for sensitive parameters?',
  'How accurately can a parameter\'s category be predicted from its name?',
  'Which words predict sensitive categories?',
  'Do natural risky vs. safe clusters emerge among plugins?',
  'Which plugin clusters have the highest sensitive-data share?',
  'Can mislabeled "Other" records be identified automatically?',
  'Are there hidden sensitive parameters mislabeled as "Other"?',
  'Which parameters collect passwords?',
  'Do plugins disclose what they collect in their privacy policies?',
];

function getInitialTheme() {
  try {
    const saved = localStorage.getItem('theme');
    if (saved === 'light' || saved === 'dark') return saved;
  } catch {
    // localStorage unavailable — fall through to system preference.
  }
  return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
}

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [theme, setTheme] = useState(getInitialTheme);

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    try {
      localStorage.setItem('theme', theme);
    } catch {
      // Ignore — theme just won't persist across reloads.
    }
  }, [theme]);

  function toggleTheme() {
    setTheme((t) => (t === 'dark' ? 'light' : 'dark'));
  }

  async function submitQuestion(question) {
    question = question.trim();
    if (!question || loading) return;

    setMessages((prev) => [...prev, { role: 'user', text: question }]);
    setInput('');
    setLoading(true);
    setError(null);

    try {
      let res;
      try {
        res = await fetch(`${API_BASE}/ask`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ question, top_k: 5 }),
        });
      } catch {
        throw new Error(
          `Can't reach the backend at ${API_BASE}. Is it running? (uvicorn app.main:app --reload)`
        );
      }

      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.detail || `Request failed (${res.status})`);
      }

      const data = await res.json();
      setMessages((prev) => [
        ...prev,
        { role: 'assistant', text: data.answer, sources: data.sources, chart: data.chart },
      ]);
    } catch (err) {
      setError(err.message || 'Something went wrong.');
    } finally {
      setLoading(false);
    }
  }

  function handleSubmit(e) {
    e.preventDefault();
    submitQuestion(input);
  }

  function clearChat() {
    setMessages([]);
    setError(null);
  }

  return (
    <div className="app">
      <header className="app-header">
        <button
          type="button"
          className="theme-toggle"
          onClick={toggleTheme}
          aria-label={theme === 'dark' ? 'Switch to light theme' : 'Switch to dark theme'}
        >
          {theme === 'dark' ? '☀️' : '🌙'}
        </button>
        <h1>GPT Plugin Privacy Assistant</h1>
        <p>Ask about what data GPT plugins collect and the privacy risks involved.</p>
        {messages.length > 0 && (
          <motion.button
            type="button"
            className="clear-chat"
            onClick={clearChat}
            disabled={loading}
            whileHover={{ scale: 1.04 }}
            whileTap={{ scale: 0.96 }}
          >
            Clear chat
          </motion.button>
        )}
      </header>

      <div className="suggestions">
        {SUGGESTED_QUESTIONS.map((q, i) => (
          <motion.button
            key={i}
            type="button"
            className="suggestion-chip"
            onClick={() => submitQuestion(q)}
            disabled={loading}
            whileHover={{ scale: 1.03, y: -1 }}
            whileTap={{ scale: 0.97 }}
          >
            {q}
          </motion.button>
        ))}
      </div>

      <div className="chat">
        {messages.length === 0 && !loading && !error && (
          <p className="empty-hint">Pick a question above, or type your own below.</p>
        )}

        <AnimatePresence initial={false}>
          {messages.map((m, i) => (
            <motion.div
              key={i}
              className={`message ${m.role}`}
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.25, ease: 'easeOut' }}
            >
              <div className="bubble">
                <p>{m.text}</p>

                {m.chart && (
                  <div className="chart-wrapper">
                    <p className="chart-title">Data categories among the retrieved results</p>
                    <CategoryChart data={m.chart} theme={theme} />
                  </div>
                )}

                {m.sources && m.sources.length > 0 && (
                  <details className="sources">
                    <summary>Sources ({m.sources.length})</summary>
                    <ul>
                      {m.sources.map((s, j) => (
                        <li key={j}>
                          <span className="score">{s.score.toFixed(2)}</span> {s.text}
                        </li>
                      ))}
                    </ul>
                  </details>
                )}
              </div>
            </motion.div>
          ))}

          {loading && (
            <motion.div
              key="typing"
              className="message assistant"
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.2, ease: 'easeOut' }}
            >
              <div className="bubble">
                <span className="typing-indicator">
                  <span />
                  <span />
                  <span />
                </span>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {error && <div className="error">Error: {error}</div>}
      </div>

      <form className="composer" onSubmit={handleSubmit}>
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Or type your own question…"
          disabled={loading}
        />
        <motion.button
          type="submit"
          disabled={loading || !input.trim()}
          whileHover={{ scale: 1.03 }}
          whileTap={{ scale: 0.97 }}
        >
          Send
        </motion.button>
      </form>
    </div>
  );
}

export default App;
