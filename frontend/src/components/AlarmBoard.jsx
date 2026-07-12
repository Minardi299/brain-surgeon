import { CATEGORY_RGB } from '../constants'

// The 5-column alarm chip grid at the top of the board. The backend only ever emits
// "FIRING" | "QUIET" (see AlarmChip in src/danger_tracker/app.py) — the design mock also
// had an "ELEVATED" tier, kept here for visual completeness but never hit by real/mock data.
export default function AlarmBoard({ chips }) {
  return (
    <>
      <div style={{ display: 'grid', gridTemplateColumns: `repeat(${chips.length || 4}, 1fr)`, gap: 10, marginTop: 18 }}>
        {chips.map((t) => {
          const firing = t.state === 'FIRING'
          const elevated = t.state === 'ELEVATED'
          const chipBorder = firing ? '#bf3b21' : elevated ? 'rgba(224,138,46,.5)' : '#d9d2c2'
          const chipBg = firing ? '#bf3b21' : elevated ? 'rgba(255,158,64,.16)' : 'transparent'
          const chipFg = firing ? '#ffffff' : elevated ? '#e08a2e' : '#6f6656'
          const barBg = `rgb(${CATEGORY_RGB[t.category] || '191,59,33'})`
          const anim = firing ? 'omAlarm 1.5s ease-in-out infinite' : 'none'
          return (
            <div
              key={t.category}
              style={{
                background: '#fcfaf4',
                border: `1px solid ${chipBorder}`,
                borderRadius: 8,
                padding: '12px 14px',
                animation: anim,
                transition: 'border-color .4s',
              }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span
                  style={{
                    font: "600 13px 'Barlow Condensed',sans-serif",
                    letterSpacing: '.1em',
                    textTransform: 'uppercase',
                    color: '#6f6656',
                  }}
                >
                  {t.name}
                </span>
                <span
                  style={{
                    display: 'inline-block',
                    padding: '2px 8px',
                    borderRadius: 3,
                    font: "700 9px 'JetBrains Mono',monospace",
                    letterSpacing: '.08em',
                    background: chipBg,
                    color: chipFg,
                  }}
                >
                  {t.state}
                </span>
              </div>
              <div style={{ font: "700 34px/1.1 'JetBrains Mono',monospace", marginTop: 6 }}>{Math.round(t.raw ?? 0)}</div>
              <div style={{ height: 6, background: '#e6dfd0', borderRadius: 3, overflow: 'hidden', marginTop: 8 }}>
                <div
                  style={{
                    height: '100%',
                    width: `${Math.round((t.bar_frac ?? 0) * 100)}%`,
                    background: barBg,
                    transition: 'width .7s',
                  }}
                />
              </div>
            </div>
          )
        })}
      </div>
      <div style={{ font: "400 10px 'JetBrains Mono',monospace", color: '#a49a83', marginTop: 8, textAlign: 'right' }}>
        alarm board: raw SAE activation · 0–150
      </div>
    </>
  )
}
