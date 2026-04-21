import React, { useState } from 'react';
import './App.css';

function App() {
  const [query, setQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!query.trim()) return;

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await fetch('http://localhost:8000/query/', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: query, top_k: 5 }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      console.error(err);
      setError('Failed to get response. Is the backend running?');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-container">
      <h1>📞 Decision Intelligence Assistant</h1>
      <form onSubmit={handleSubmit}>
        <textarea
          rows="3"
          cols="50"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Describe your customer support issue..."
          disabled={loading}
        />
        <button type="submit" disabled={loading}>
          {loading ? 'Processing...' : 'Ask'}
        </button>
      </form>

      {error && <div className="error">{error}</div>}

      {result && (
        <div className="results">
          <div className="answers">
            <div className="answer-card">
              <h2>📚 RAG Answer</h2>
              <p>{result.rag_answer}</p>
              <div className="metrics">
                <span>⏱️ {result.rag_latency_ms.toFixed(0)} ms</span>
                <span>💰 ${result.rag_cost_usd.toFixed(6)}</span>
              </div>
            </div>
            <div className="answer-card">
              <h2>💬 Non‑RAG Answer</h2>
              <p>{result.non_rag_answer}</p>
              <div className="metrics">
                <span>⏱️ {result.non_rag_latency_ms.toFixed(0)} ms</span>
                <span>💰 ${result.non_rag_cost_usd.toFixed(6)}</span>
              </div>
            </div>
          </div>

          <div className="priority-comparison">
            <h2>⚖️ Priority Predictors Comparison</h2>
            <table>
              <thead>
                <tr><th>Method</th><th>Label</th><th>Confidence</th><th>Latency (ms)</th><th>Cost (USD)</th></tr>
              </thead>
              <tbody>
                <tr>
                  <td>🤖 ML Classifier</td>
                  <td>{result.ml_priority.label === '1' ? 'Urgent' : 'Normal'}</td>
                  <td>{(result.ml_priority.confidence * 100).toFixed(1)}%</td>
                  <td>{result.ml_latency_ms.toFixed(2)}</td>
                  <td>0.000000 (local)</td>
                </tr>
                <tr>
                  <td>🧠 LLM Zero‑shot</td>
                  <td>{result.llm_priority.label}</td>
                  <td>{(result.llm_priority.confidence * 100).toFixed(1)}%</td>
                  <td>{result.llm_priority_latency_ms.toFixed(2)}</td>
                  <td>${result.llm_priority_cost_usd.toFixed(6)}</td>
                </tr>
              </tbody>
            </table>
          </div>

          <div className="retrieved-chunks">
            <h2>📄 Retrieved Similar Tickets</h2>
            {result.retrieved_chunks.map((chunk, idx) => (
              <div key={idx} className="chunk-card">
                <div className="chunk-meta">
                  <span>Tweet ID: {chunk.metadata.original_tweet_id}</span>
                  <span>Priority: {chunk.metadata.priority === 1 ? 'Urgent' : 'Normal'}</span>
                  <span>Similarity: {(chunk.similarity * 100).toFixed(2)}%</span>
                </div>
                <p>{chunk.text}</p>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
