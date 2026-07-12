export default function Header() {
  return (
    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', gap: 16 }}>
      <div style={{ font: "600 15px 'Barlow Condensed',sans-serif", letterSpacing: '.3em', color: '#6f6656' }}>
        GEMMA-2-2B SAFETY OPS
      </div>
      <div style={{ font: "700 44px/1 'Barlow Condensed',sans-serif", letterSpacing: '.06em', textTransform: 'uppercase' }}>
        Recognize <span style={{ color: '#bf3b21' }}>≠</span> Refuse
      </div>
      <div style={{ font: "400 10.5px/1.6 'JetBrains Mono',monospace", color: '#6f6656', textAlign: 'right' }}>
        SAE @ L20 · observe
        <br />
        direction @ L8–12 · intervene
      </div>
    </div>
  )
}
