import { useState } from 'react'

export default function TryYourOwn({ onAnalyze, loading, mock }) {
  const [value, setValue] = useState('')

  const run = () => {
    if (!value.trim()) return
    onAnalyze(value)
  }

  return (
    <>
      <div
        style={{
          display: 'flex',
          gap: 10,
          alignItems: 'center',
          marginTop: 12,
          background: '#fcfaf4',
          border: '1px solid #d9d2c2',
          borderRadius: 8,
          padding: '11px 14px',
        }}
      >
        <span style={{ font: "600 13px 'Barlow Condensed',sans-serif", letterSpacing: '.22em', color: '#bf3b21', flex: 'none' }}>
          TRY YOUR OWN ▸
        </span>
        <input
          type="text"
          placeholder="Enter a prompt for the safety ops board…"
          value={value}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter') {
              e.preventDefault()
              run()
            }
          }}
          style={{
            flex: 1,
            background: '#f5f1e8',
            border: '1px solid #d9d2c2',
            borderRadius: 6,
            padding: '9px 12px',
            font: "400 13px 'JetBrains Mono',monospace",
            color: '#211d16',
            outline: 'none',
          }}
        />
        <button
          onClick={run}
          disabled={loading}
          style={{
            cursor: loading ? 'default' : 'pointer',
            flex: 'none',
            font: "700 12px 'Barlow Condensed',sans-serif",
            letterSpacing: '.14em',
            textTransform: 'uppercase',
            padding: '10px 18px',
            borderRadius: 6,
            border: '1px solid #bf3b21',
            background: 'rgba(191,59,33,.1)',
            color: '#bf3b21',
            opacity: loading ? 0.6 : 1,
          }}
        >
          Analyze
        </button>
      </div>
      <div style={{ display: 'flex', gap: 7, alignItems: 'center', marginTop: 7, flexWrap: 'wrap' }}>
        <span style={{ font: "400 10px 'JetBrains Mono',monospace", color: '#a49a83' }}>
          Pick a beat above to load its prompt, or type your own ·{' '}
          {mock ? 'keyword-based mock analyzer — no model is running.' : 'runs the live Gemma-2-2B + SAE model.'}
        </span>
      </div>
    </>
  )
}
