import { PALETTE } from '../constants'
import { cellStyle } from '../helpers'

// 01 — PER-TOKEN ATTRIBUTION: one row per harm category, one heatmap cell per token.
export default function Panel01({ rows, spotStyle }) {
  return (
    <div style={{ marginTop: 24, transition: 'opacity .45s ease,filter .45s ease', ...spotStyle }}>
      <div style={{ font: "600 15px 'Barlow Condensed',sans-serif", letterSpacing: '.24em', color: PALETTE.danger, marginBottom: 10 }}>
        01 — PER-TOKEN ATTRIBUTION{' '}
        <span style={{ color: PALETTE.faint, letterSpacing: '.08em', fontSize: 12 }}>
          hover a cell · raw activation 0–150 · &lt;bos&gt; excluded
        </span>
      </div>
      <div style={{ background: '#fcfaf4', border: '1px solid #d9d2c2', borderRadius: 8, padding: '14px 16px', display: 'flex', flexDirection: 'column', gap: 6 }}>
        {(rows ?? []).map((r) => (
          <div key={r.category} style={{ display: 'grid', gridTemplateColumns: '170px 1fr', gap: 12, alignItems: 'center' }}>
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: 8,
                font: "600 13px 'Barlow Condensed',sans-serif",
                letterSpacing: '.08em',
                textTransform: 'uppercase',
                color: PALETTE.dim,
              }}
            >
              <span style={{ width: 8, height: 8, borderRadius: 2, background: `rgb(${r.rgb})`, display: 'inline-block', flex: 'none' }} />
              {r.name}
            </div>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 3 }}>
              {r.cells.map(([token, value], i) => {
                const { bg, fg } = cellStyle(r.rgb, value)
                return (
                  <span
                    key={i}
                    title={`“${token}” · ${r.name} · activation ${value} / 150`}
                    style={{
                      padding: '5px 7px',
                      borderRadius: 3,
                      font: "500 12px 'JetBrains Mono',monospace",
                      background: bg,
                      color: fg,
                      cursor: 'default',
                      transition: 'background .5s',
                    }}
                  >
                    {token}
                  </span>
                )
              })}
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
