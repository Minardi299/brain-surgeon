import HeroSpine from './HeroSpine'
import { PALETTE } from '../constants'
import { fmt2, gapStr } from '../helpers'

function spineCaption(beat, severed) {
  const danger = beat?.danger_norm ?? 0
  const refusal = beat?.refusal_norm ?? 0
  if (severed) {
    return 'Refusal direction severed at L8–12. Danger still ignites at the L20 readout — recognition is intact, but nothing carries it to a refusal.'
  }
  if (danger < 0.15) {
    return 'The stream runs quiet — nothing ignites at the readout and the refusal link idles. Nothing to refuse.'
  }
  if (refusal > 0.4) {
    return 'Danger ignites at the L20 readout and the refusal link engages at L8–12 — recognition wired to refusal.'
  }
  return 'Danger ignites — the model recognizes it — but the refusal link barely engages. The gap is open.'
}

// 00 — MODEL INTERNALS: the residual-stream spine SVG + the OBSERVE/INTERVENE readouts +
// the cut/restore control. This is the only place the cut/restore toggle lives (matches the
// design, where panel 03 below it is a static illustrative diptych, not another button).
export default function Panel00({ beat, severed, loading, onToggleCut, spotStyle }) {
  const danger = beat?.danger_norm ?? 0
  const refusal = beat?.refusal_norm ?? 0
  const gap = beat?.gap ?? danger - refusal

  return (
    <div style={{ marginTop: 24, transition: 'opacity .45s ease,filter .45s ease', ...spotStyle }}>
      <div style={{ font: "600 15px 'Barlow Condensed',sans-serif", letterSpacing: '.24em', color: PALETTE.danger, marginBottom: 10 }}>
        00 — MODEL INTERNALS{' '}
        <span style={{ color: PALETTE.faint, letterSpacing: '.08em', fontSize: 12 }}>
          live · residual stream through 26 layers · observe @ L20, intervene @ L8–12
        </span>
      </div>
      <div
        style={{
          background: '#fcfaf4',
          border: '1px solid #d9d2c2',
          borderRadius: 8,
          padding: '10px 16px 16px',
          display: 'grid',
          gridTemplateColumns: '392px 1fr',
          gap: 20,
          alignItems: 'center',
        }}
      >
        <div>
          <HeroSpine beat={beat} severed={severed} />
        </div>
        <div>
          <div style={{ font: "italic 400 17px/1.55 'Newsreader',serif", color: '#544c3d', maxWidth: 560 }}>
            {spineCaption(beat, severed)}
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, marginTop: 20 }}>
            <div style={{ border: '1px solid #d9d2c2', borderRadius: 8, padding: '13px 15px' }}>
              <div style={{ font: "600 12px 'Barlow Condensed',sans-serif", letterSpacing: '.16em', color: PALETTE.danger }}>
                OBSERVE · SAE @ L20
              </div>
              <div style={{ font: "700 34px 'JetBrains Mono',monospace", color: PALETTE.danger, marginTop: 4 }}>{fmt2(danger)}</div>
              <div style={{ font: "500 11px 'Barlow Condensed',sans-serif", letterSpacing: '.14em', color: PALETTE.dim, marginTop: 2 }}>
                DANGER RECOGNIZED
              </div>
            </div>
            <div style={{ border: '1px solid #d9d2c2', borderRadius: 8, padding: '13px 15px' }}>
              <div style={{ font: "600 12px 'Barlow Condensed',sans-serif", letterSpacing: '.16em', color: PALETTE.refusal }}>
                INTERVENE · @ L8–12
              </div>
              <div style={{ font: "700 34px 'JetBrains Mono',monospace", color: PALETTE.refusal, marginTop: 4 }}>{fmt2(refusal)}</div>
              <div style={{ font: "500 11px 'Barlow Condensed',sans-serif", letterSpacing: '.14em', color: PALETTE.dim, marginTop: 2 }}>
                REFUSAL ENGAGED
              </div>
            </div>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 16, marginTop: 18, flexWrap: 'wrap' }}>
            <button
              onClick={onToggleCut}
              disabled={loading}
              style={{
                cursor: loading ? 'default' : 'pointer',
                font: "700 12px 'Barlow Condensed',sans-serif",
                letterSpacing: '.12em',
                textTransform: 'uppercase',
                padding: '12px 22px',
                borderRadius: 6,
                border: `1px solid ${PALETTE.danger}`,
                background: 'rgba(191,59,33,.1)',
                color: PALETTE.danger,
                opacity: loading ? 0.6 : 1,
              }}
            >
              {severed ? '↺ RESTORE THE DIRECTION' : '✂ CUT THE REFUSAL DIRECTION'}
            </button>
            <div style={{ font: "600 14px 'Barlow Condensed',sans-serif", letterSpacing: '.14em', color: PALETTE.dim }}>
              WIRE {severed ? 'severed' : 'intact'} · GAP{' '}
              <span style={{ color: PALETTE.ink, fontFamily: "'JetBrains Mono',monospace" }}>{gapStr(gap)}</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
