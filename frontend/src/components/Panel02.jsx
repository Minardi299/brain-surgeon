import { PALETTE } from '../constants'
import { pct } from '../helpers'

// 02 — TOP FIRING FEATURES: ranked feature bars; the refusal-category feature gets the
// "BLINKING WIRE" badge (is_wire, from the backend's TopFeatureView).
export default function Panel02({ features, spotStyle }) {
  return (
    <div style={{ transition: 'opacity .45s ease,filter .45s ease', ...spotStyle }}>
      <div style={{ font: "600 15px 'Barlow Condensed',sans-serif", letterSpacing: '.24em', color: PALETTE.danger, marginBottom: 10 }}>
        02 — TOP FIRING FEATURES <span style={{ color: PALETTE.faint, letterSpacing: '.08em', fontSize: 12 }}>raw activation 0–150</span>
      </div>
      <div style={{ background: '#fcfaf4', border: '1px solid #d9d2c2', borderRadius: 8, padding: '6px 16px' }}>
        {(features ?? []).map((f, i) => (
          <div
            key={`${f.feature_id}-${i}`}
            style={{
              display: 'grid',
              gridTemplateColumns: '70px 1fr 120px',
              gap: 12,
              alignItems: 'center',
              padding: '9px 0',
              borderBottom: i === features.length - 1 ? 'none' : `1px solid ${PALETTE.hair}`,
            }}
          >
            <span style={{ font: "500 11.5px 'JetBrains Mono',monospace", color: PALETTE.dim }}>#{f.feature_id}</span>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, minWidth: 0 }}>
              <span style={{ font: "400 15px 'Barlow Condensed',sans-serif", letterSpacing: '.02em' }}>“{f.label}”</span>
              {f.is_wire && (
                <span
                  style={{
                    flex: 'none',
                    font: "700 8px 'JetBrains Mono',monospace",
                    letterSpacing: '.06em',
                    padding: '2px 6px',
                    borderRadius: 3,
                    background: 'rgba(31,95,191,.14)',
                    color: PALETTE.refusal,
                  }}
                >
                  BLINKING WIRE
                </span>
              )}
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <div style={{ flex: 1, height: 7, background: PALETTE.segOff, borderRadius: 3, overflow: 'hidden' }}>
                <div
                  style={{
                    height: '100%',
                    width: pct(f.act),
                    background: f.is_wire ? PALETTE.refusal : PALETTE.danger,
                    transition: 'width .7s',
                  }}
                />
              </div>
              <span style={{ font: "600 11.5px 'JetBrains Mono',monospace", width: 28, textAlign: 'right' }}>{Math.round(f.act)}</span>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
