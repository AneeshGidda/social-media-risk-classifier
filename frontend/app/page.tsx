"use client";

import { useMemo, useState, useEffect } from "react";

const MAX_CHARS = 280;

interface PredictionResponse {
  label: string;
  confidence: number;
  text: string;
  prob_not_suicide: number;
  prob_potential_suicide: number;
}

export default function SocialMediaRiskPage() {
  const [mounted, setMounted] = useState(false);
  const [text, setText] = useState("");
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState("");
  const [res, setRes] = useState<PredictionResponse | null>(null);

  useEffect(() => {
    setMounted(true);
  }, []);

  const remaining = MAX_CHARS - text.length;

  const examples = useMemo(
    () => [
      "I don't see the point in anything anymore. Everything feels meaningless.",
      "Just had the best coffee and ready to tackle the day!",
      "I'm so tired of feeling this way. Nothing seems to help.",
      "Excited for the weekend plans with everyone!",
    ],
    []
  );

  // Use environment variable for API URL, fallback to localhost for development
  const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000/predict";

  const predict = async () => {
    setErr("");
    setRes(null);

    const trimmed = text.trim();
    if (!trimmed) {
      setErr("Type something first.");
      return;
    }
    if (trimmed.length > MAX_CHARS) {
      setErr(`Keep it under ${MAX_CHARS} characters.`);
      return;
    }

    setLoading(true);
    const startTime = Date.now();
    
    try {
      const r = await fetch(apiUrl, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: trimmed }),
      });

      const data = await r.json();
      
      // Ensure minimum 1 second delay
      const elapsed = Date.now() - startTime;
      const remainingDelay = Math.max(0, 1000 - elapsed);
      
      await new Promise(resolve => setTimeout(resolve, remainingDelay));
      
      if (!r.ok || data.error) {
        setErr(data?.error || "Request failed.");
      } else {
        setRes(data);
      }
    } catch (e) {
      // Ensure minimum 1 second delay even on error
      const elapsed = Date.now() - startTime;
      const remainingDelay = Math.max(0, 1000 - elapsed);
      await new Promise(resolve => setTimeout(resolve, remainingDelay));
      
      setErr("Could not reach the API. Is it running?");
    } finally {
      setLoading(false);
    }
  };

  const badge = useMemo(() => {
    if (!res) return null;
    const isPotential = res.label?.toLowerCase().includes("potential");
    const conf = res.confidence ?? 0;

    if (!isPotential) return { text: "Low risk", tone: "ok" };
    if (conf >= 0.85) return { text: "High risk", tone: "bad" };
    if (conf >= 0.65) return { text: "Medium risk", tone: "warn" };
    return { text: "Low confidence", tone: "warn" };
  }, [res]);

  if (!mounted) {
    return null;
  }

  return (
    <div className="page">
      <div className="shell">
        <header className="topbar">
          <div className="title">Social Media Risk Classifier</div>
          <div className="subtitle">Tweet-style demo powered by DistilBERT</div>
        </header>

        <div className="card compose">
          <div className="avatar" aria-hidden />
          <div className="composeBody">
            <div className="handleRow">
              <div className="name">Aneesh</div>
              <div className="handle">@aneeshgidda</div>
              <div className="dot">·</div>
              <div className="muted">Demo</div>
            </div>

            <textarea
              className="textarea"
              value={text}
              onChange={(e) => setText(e.target.value)}
              placeholder="What's happening?"
              spellCheck={false}
            />

            <div className="row">
              <div className="chips">
                {examples.map((ex) => (
                  <button
                    key={ex}
                    className="chip"
                    type="button"
                    onClick={() => setText(ex)}
                    disabled={loading}
                    title="Use example"
                  >
                    {ex.length > 34 ? ex.slice(0, 34) + "…" : ex}
                  </button>
                ))}
              </div>
            </div>

            <div className="actions">
              <div className={`counter ${remaining < 0 ? "bad" : remaining < 20 ? "warn" : ""}`}>
                {remaining}
              </div>

              <button className="btn secondary" type="button" onClick={() => setText("")} disabled={loading}>
                Clear
              </button>
              <button className="btn primary" type="button" onClick={predict} disabled={loading}>
                {loading ? "Predicting…" : "Predict"}
              </button>
            </div>

            {err && <div className="error">{err}</div>}
          </div>
        </div>

        {res && (
          <div className="card tweet">
            <div className="avatar" aria-hidden />
            <div className="tweetBody">
              <div className="handleRow">
                <div className="name">Model Output</div>
                <div className="handle">@classifier</div>
                <div className="dot">·</div>
                <div className="muted">now</div>

                {badge && (
                  <div className={`badge ${badge.tone}`}>
                    {badge.text}
                  </div>
                )}
              </div>

              <div className="tweetText">{res.text}</div>

              <div className="metrics">
                <div className="metric">
                  <div className="label">Prediction</div>
                  <div className="value">{res.label}</div>
                </div>
                <div className="metric">
                  <div className="label">Confidence</div>
                  <div className="value">{(res.confidence * 100).toFixed(2)}%</div>
                </div>
                <div className="metric">
                  <div className="label">Not suicide</div>
                  <div className="value">{(res.prob_not_suicide * 100).toFixed(2)}%</div>
                </div>
                <div className="metric">
                  <div className="label">Potential</div>
                  <div className="value">{(res.prob_potential_suicide * 100).toFixed(2)}%</div>
                </div>
              </div>

              <div className="disclaimer">
                Educational demo only. Not a clinical tool.
              </div>
            </div>
          </div>
        )}

      </div>

      <style jsx>{`
        :global(html, body) {
          height: 100%;
          background: #0b0f14;
          color: #e7e9ea;
        }

        .page {
          min-height: 100%;
          display: flex;
          justify-content: center;
          padding: 28px 14px;
        }

        .shell {
          width: 100%;
          max-width: 720px;
        }

        .topbar {
          margin-bottom: 14px;
          padding: 8px 6px;
        }

        .title {
          font-size: 24px;
          font-weight: 800;
          letter-spacing: -0.02em;
        }

        .subtitle {
          margin-top: 6px;
          color: #8b98a5;
          font-size: 14px;
        }

        .card {
          border: 1px solid #22303c;
          background: #0f1419;
          border-radius: 18px;
          padding: 18px;
          display: flex;
          gap: 12px;
          box-shadow: 0 6px 24px rgba(0,0,0,0.25);
        }

        .compose {
          margin-top: 10px;
        }

        .tweet {
          margin-top: 14px;
        }

        .avatar {
          width: 44px;
          height: 44px;
          border-radius: 999px;
          background: linear-gradient(180deg, #1d9bf0, #0a66c2);
          flex: 0 0 auto;
        }

        .composeBody, .tweetBody {
          flex: 1;
          min-width: 0;
        }

        .handleRow {
          display: flex;
          align-items: center;
          gap: 8px;
          flex-wrap: wrap;
        }

        .name {
          font-weight: 800;
          letter-spacing: -0.01em;
        }

        .handle, .muted {
          color: #8b98a5;
        }

        .dot {
          color: #8b98a5;
        }

        .textarea {
          width: 100%;
          margin-top: 10px;
          resize: none;
          border: none;
          outline: none;
          background: transparent;
          color: #e7e9ea;
          font-size: 18px;
          line-height: 1.35;
          min-height: 120px;
        }

        .textarea::placeholder {
          color: #536471;
        }

        .row {
          margin-top: 10px;
        }

        .chips {
          display: flex;
          gap: 8px;
          flex-wrap: wrap;
        }

        .chip {
          border: 1px solid #22303c;
          background: rgba(255,255,255,0.03);
          color: #e7e9ea;
          padding: 6px 10px;
          border-radius: 999px;
          font-size: 12px;
          cursor: pointer;
        }

        .chip:hover {
          border-color: #1d9bf0;
        }

        .actions {
          margin-top: 12px;
          display: flex;
          justify-content: flex-end;
          align-items: center;
          gap: 10px;
        }

        .counter {
          font-size: 12px;
          color: #8b98a5;
          margin-right: auto;
        }

        .counter.warn { color: #f6c177; }
        .counter.bad { color: #ff6b6b; }

        .btn {
          border: 1px solid #22303c;
          border-radius: 999px;
          padding: 9px 14px;
          font-weight: 700;
          cursor: pointer;
          font-size: 14px;
        }

        .btn.primary {
          background: #1d9bf0;
          border-color: #1d9bf0;
          color: #0b0f14;
        }

        .btn.primary:hover {
          filter: brightness(1.03);
        }

        .btn.secondary {
          background: transparent;
          color: #e7e9ea;
        }

        .btn:disabled {
          opacity: 0.6;
          cursor: not-allowed;
        }

        .error {
          margin-top: 10px;
          color: #ff6b6b;
          font-size: 13px;
        }

        .tweetText {
          margin-top: 10px;
          font-size: 16px;
          line-height: 1.45;
          white-space: pre-wrap;
        }

        .metrics {
          margin-top: 12px;
          display: grid;
          grid-template-columns: repeat(2, minmax(0, 1fr));
          gap: 10px;
        }

        .metric {
          border: 1px solid #22303c;
          background: rgba(255,255,255,0.02);
          border-radius: 14px;
          padding: 10px 12px;
        }

        .metric .label {
          font-size: 12px;
          color: #8b98a5;
        }

        .metric .value {
          margin-top: 4px;
          font-weight: 800;
        }

        .badge {
          margin-left: auto;
          padding: 6px 10px;
          border-radius: 999px;
          font-size: 12px;
          font-weight: 800;
          border: 1px solid #22303c;
          background: rgba(255,255,255,0.02);
        }

        .badge.ok { border-color: rgba(0, 186, 124, 0.35); }
        .badge.warn { border-color: rgba(246, 193, 119, 0.45); }
        .badge.bad { border-color: rgba(255, 107, 107, 0.5); }

        .disclaimer {
          margin-top: 10px;
          color: #8b98a5;
          font-size: 12px;
        }
      `}</style>
    </div>
  );
}
