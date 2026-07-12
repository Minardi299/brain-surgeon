export default function PromptOutput({ beat, mock }) {
  const prompt = beat?.prompt ? `“${beat.prompt}”` : ''
  const output = beat?.response ?? ''
  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, marginTop: 14 }}>
      <div style={{ background: '#fcfaf4', border: '1px solid #d9d2c2', borderRadius: 8, padding: '13px 16px' }}>
        <div style={{ font: "600 12px 'Barlow Condensed',sans-serif", letterSpacing: '.24em', color: '#6f6656', marginBottom: 8 }}>
          PROMPT
        </div>
        <div style={{ font: "400 12.5px/1.6 'JetBrains Mono',monospace" }}>{prompt}</div>
      </div>
      <div style={{ background: '#fcfaf4', border: '1px solid #d9d2c2', borderRadius: 8, padding: '13px 16px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
          <span style={{ font: "600 12px 'Barlow Condensed',sans-serif", letterSpacing: '.24em', color: '#6f6656', marginBottom: 8 }}>
            MODEL OUTPUT
          </span>
          <span style={{ font: "400 9px 'JetBrains Mono',monospace", color: '#a49a83' }}>
            {mock ? 'mock · redacted' : 'live · redacted'}
          </span>
        </div>
        <div style={{ font: "400 12.5px/1.6 'JetBrains Mono',monospace", color: '#544c3d' }}>{output}</div>
      </div>
    </div>
  )
}
