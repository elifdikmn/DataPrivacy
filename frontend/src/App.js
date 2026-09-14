import React, { useState } from 'react';
import Plotly from 'plotly.js-dist-min';
import createPlotlyComponent from 'react-plotly.js/factory';
import './App.css';

const Plot = createPlotlyComponent(Plotly);

const API_BASE = process.env.REACT_APP_API_BASE || 'http://127.0.0.1:8000';

const SUGGESTED_QUESTIONS = [
  'What percentage of collected data is sensitive?',
  'Do plugins write descriptions less often for sensitive parameters?',
  'How accurately can a parameter\'s category be predicted from its name?',
  'Do natural risky vs. safe clusters emerge among plugins?',
  'Can mislabeled "Other" records be identified automatically?',
  'Which parameters collect passwords?',
  'Do plugins disclose what they collect in their privacy policies?',
];

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  async function submitQuestion(question) {
    question = question.trim();
    if (!question || loading) return;

    setMessages((prev) => [...prev, { role: 'user', text: question }]);
    setInput('');
    setLoading(true);
    setError(null);

    try {
      const res = await fetch(`${API_BASE}/ask`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question, top_k: 5 }),
      });

      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.detail || `Request failed (${res.status})`);
      }

      const data = await res.json();
      const chart = data.chart ? JSON.parse(data.chart) : null;
      setMessages((prev) => [
        ...prev,
        { role: 'assistant', text: data.answer, sources: data.sources, chart },
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

  return (
    <div className="app">
      <header className="app-header">
        <h1>GPT Plugin Privacy Assistant</h1>
        <p>Ask about what data GPT plugins collect and the privacy risks involved.</p>
      </header>

      <div className="suggestions">
        {SUGGESTED_QUESTIONS.map((q, i) => (
          <button
            key={i}
            type="button"
            className="suggestion-chip"
            onClick={() => submitQuestion(q)}
            disabled={loading}
          >
            {q}
          </button>
        ))}
      </div>

      <div className="chat">
        {messages.map((m, i) => (
          <div key={i} className={`message ${m.role}`}>
            <div className="bubble">
              <p>{m.text}</p>

              {m.chart && (
                <div className="chart-wrapper">
                  <Plot
                    data={m.chart.data}
                    layout={{ ...m.chart.layout, autosize: true, height: 320 }}
                    style={{ width: '100%' }}
                    config={{ displayModeBar: false, responsive: true }}
                  />
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
          </div>
        ))}

        {loading && (
          <div className="message assistant">
            <div className="bubble">Thinking…</div>
          </div>
        )}

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
        <button type="submit" disabled={loading || !input.trim()}>
          Send
        </button>
      </form>
    </div>
  );
}

export default App;
