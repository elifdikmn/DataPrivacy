import React, { useEffect, useRef, useState } from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import './App.css';

export function resolveApiBase(configured, environment) {
  const base = (configured || '').trim().replace(/\/+$/, '');
  return base || (environment === 'production' ? '' : 'http://127.0.0.1:8000');
}

// How many earlier messages are sent with each question, so follow-ups
// ("and its confidence interval?") keep their context.
const HISTORY_LENGTH = 6;
const MAX_QUESTION_LENGTH = 4000;

const API_BASE = resolveApiBase(process.env.REACT_APP_API_BASE, process.env.NODE_ENV);

// Keep the canonical query for the backend's exact chart mappings while showing
// non-technical wording to general readers.
const GENERAL_QUESTIONS = [
  { label: 'What information can GPT Actions ask for?', query: 'What data are collected by GPT Actions?' },
  { label: 'How much of that information is sensitive?', query: 'What percentage of collected data is sensitive?' },
  { label: 'Which sensitive details appear most often?', query: 'Which sensitive data types appear most often?' },
  { label: 'Are sensitive requests explained as often as other requests?', query: 'Do plugins write descriptions less often for sensitive parameters?' },
  { label: 'Where do password requests appear?', query: 'Which parameters collect passwords?' },
  { label: 'Do privacy policies explain what these tools ask for?', query: 'Do plugins disclose what they collect in their privacy policies?' },
  { label: 'Are the sensitive requests also missing from privacy policies?', query: 'Are sensitive parameters also the undisclosed ones?' },
];

const RESEARCHER_QUESTIONS = [
  'What are the model performance confidence intervals?',
  'How accurately can a parameter\'s category be predicted from its name?',
  'Which words predict sensitive categories?',
  'Do natural risky vs. safe clusters emerge among plugins?',
  'Which plugin clusters have the highest sensitive-data share?',
  'Can mislabeled "Other" records be identified automatically?',
  'Are there hidden sensitive parameters mislabeled as "Other"?',
].map((question) => ({ label: question, query: question }));

const AUDIENCES = [
  { value: 'general', label: 'General audience' },
  { value: 'researcher', label: 'Researcher' },
];

// Renders the small Markdown subset the assistant is asked to use:
// **bold** key terms and "- " list items. Everything else stays plain text,
// so no HTML from the model is ever injected into the page.
export function FormattedAnswer({ text }) {
  const lines = text.split('\n');
  return (
    <p>
      {lines.map((line, i) => {
        const isBullet = /^\s*[-*]\s+/.test(line);
        const content = isBullet ? line.replace(/^\s*[-*]\s+/, '') : line;
        const parts = content.split(/(\*\*[^*\n]+\*\*)/g).map((part, j) =>
          part.length > 4 && part.startsWith('**') && part.endsWith('**') ? (
            <strong key={j}>{part.slice(2, -2)}</strong>
          ) : (
            <React.Fragment key={j}>{part}</React.Fragment>
          )
        );
        return (
          <React.Fragment key={i}>
            {isBullet && '• '}
            {parts}
            {i < lines.length - 1 && '\n'}
          </React.Fragment>
        );
      })}
    </p>
  );
}

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
  const [audience, setAudience] = useState(() => {
    try {
      return localStorage.getItem('audience') === 'researcher' ? 'researcher' : 'general';
    } catch {
      return 'general';
    }
  });
  const [visibleCharts, setVisibleCharts] = useState(new Set());
  const composerRef = useRef(null);

  // Keep the newest message (and the input box) in view.
  useEffect(() => {
    if (messages.length > 0 || loading) {
      composerRef.current?.scrollIntoView?.({ behavior: 'smooth', block: 'end' });
    }
  }, [messages.length, loading]);

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    try {
      localStorage.setItem('theme', theme);
    } catch {
      // Ignore — theme just won't persist across reloads.
    }
  }, [theme]);

  useEffect(() => {
    try {
      localStorage.setItem('audience', audience);
    } catch {
      // The selector still works for the current session.
    }
  }, [audience]);

  function toggleTheme() {
    setTheme((t) => (t === 'dark' ? 'light' : 'dark'));
  }

  async function submitQuestion(question, displayQuestion = question) {
    question = question.trim();
    if (!question || loading) return;
    if (!API_BASE) {
      setError('The backend address is not configured. Set REACT_APP_API_BASE and rebuild the site.');
      return;
    }

    const history = messages
      .slice(-HISTORY_LENGTH)
      .map(({ role, text }) => ({ role, text: text.slice(0, MAX_QUESTION_LENGTH) }));

    setMessages((prev) => [...prev, { role: 'user', text: displayQuestion }]);
    setInput('');
    setLoading(true);
    setError(null);

    try {
      let res;
      try {
        res = await fetch(`${API_BASE}/ask`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ question, top_k: 5, audience, history }),
        });
      } catch {
        throw new Error(
          `Can't reach the backend at ${API_BASE}. Please try again later.`
        );
      }

      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.detail || `Request failed (${res.status})`);
      }

      const data = await res.json();
      setMessages((prev) => [
        ...prev,
        { role: 'assistant', text: data.answer, sources: data.sources, chartImage: data.chart_image },
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
    setVisibleCharts(new Set());
  }

  function toggleChart(i) {
    setVisibleCharts((prev) => {
      const next = new Set(prev);
      if (next.has(i)) {
        next.delete(i);
      } else {
        next.add(i);
      }
      return next;
    });
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
        <div className="audience-control">
          <label htmlFor="audience">Explain for</label>
          <select
            id="audience"
            value={audience}
            onChange={(event) => setAudience(event.target.value)}
            disabled={loading}
          >
            {AUDIENCES.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
        </div>
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
        {(audience === 'general' ? GENERAL_QUESTIONS : [...GENERAL_QUESTIONS, ...RESEARCHER_QUESTIONS]).map((item) => (
          <motion.button
            key={item.query}
            type="button"
            className="suggestion-chip"
            onClick={() => submitQuestion(item.query, item.label)}
            disabled={loading}
            whileHover={{ scale: 1.03, y: -1 }}
            whileTap={{ scale: 0.97 }}
          >
            {item.label}
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
                {m.role === 'assistant' ? <FormattedAnswer text={m.text} /> : <p>{m.text}</p>}

                {m.chartImage && (
                  <div className="chart-toggle-area">
                    <motion.button
                      type="button"
                      className="show-visualization"
                      onClick={() => toggleChart(i)}
                      whileHover={{ scale: 1.03 }}
                      whileTap={{ scale: 0.97 }}
                    >
                      {visibleCharts.has(i) ? 'Hide visualization' : 'Show visualization'}
                    </motion.button>

                    {visibleCharts.has(i) && (
                      <div className="chart-wrapper">
                        {m.chartImage.endsWith('.html') ? (
                          <iframe
                            src={`${API_BASE}${m.chartImage}`}
                            title="Interactive chart from the analysis"
                            className="chart-iframe"
                          />
                        ) : (
                          <img
                            src={`${API_BASE}${m.chartImage}`}
                            alt="Supporting chart from the analysis"
                            className="chart-image"
                          />
                        )}
                      </div>
                    )}
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

      <form className="composer" onSubmit={handleSubmit} ref={composerRef}>
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Or type your own question…"
          maxLength={MAX_QUESTION_LENGTH}
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
