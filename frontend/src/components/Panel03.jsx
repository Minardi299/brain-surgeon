import { PALETTE } from '../constants'
import { fmt2 } from '../helpers'

// 03 — CUT THE WIRE: the "before/after" causal-test diptych. In the design mock this panel
// is entirely static (hardcoded illustrative numbers, no state bindings at all). Here it is
// wired to the real harmful/cut BeatDicts once both have been fetched (via the panel-00 cut
// button, or the inline "Run the causal test" button below if the user lands on this panel
// first) — falling back to the design's illustrative numbers until then.
export default function Panel03({ harmfulBeat, cutBeat, onRun, loading, spotStyle }) {
  const haveBoth = Boolean(harmfulBeat && cutBeat)
  const intact = harmfulBeat ?? { danger_norm: 0.9, refusal_norm: 0.88, response: 'I cannot and will not provide instructions for building a weapon…' }
  const severed = cutBeat ?? {
    danger_norm: 0.51,
    refusal_norm: 0.05,
    response: "Here's a breakdown of how to build a bomb: first you'll need ████ ███ █████████ …",
  }

  return (
    <div style={{ transition: 'opacity .45s ease,filter .45s ease', ...spotStyle }}>
      <div style={{ font: "600 15px 'Barlow Condensed',sans-serif", letterSpacing: '.24em', color: PALETTE.danger, marginBottom: 6 }}>
        03 — CUT THE WIRE{' '}
        <span style={{ color: PALETTE.faint, letterSpacing: '.08em', fontSize: 12 }}>the causal test · same prompt, before / after</span>
      </div>
      <div style={{ font: "400 11.5px/1.6 'JetBrains Mono',monospace", color: '#544c3d', marginBottom: 10 }}>
        The wire = the <span style={{ color: PALETTE.refusal }}>refusal direction</span> — the causal link from recognizing danger to
        refusing. Cut it and refusal collapses, while the model keeps recognizing the danger.
      </div>
      <div style={{ background: '#fcfaf4', border: '1px solid #d9d2c2', borderRadius: 8, padding: '16px 18px' }}>
        {!haveBoth && (
          <div style={{ marginBottom: 12, display: 'flex', alignItems: 'center', gap: 10 }}>
            <span style={{ font: "400 11px 'JetBrains Mono',monospace", color: PALETTE.faint }}>
              showing illustrative numbers — run the live causal test to replace them:
            </span>
            <button
              onClick={onRun}
              disabled={loading}
              style={{
                cursor: loading ? 'default' : 'pointer',
                font: "700 10px 'JetBrains Mono',monospace",
                padding: '6px 11px',
                borderRadius: 4,
                border: `1px solid ${PALETTE.borderAlt}`,
                background: 'transparent',
                color: PALETTE.ink,
                opacity: loading ? 0.6 : 1,
              }}
            >
              RUN THE CAUSAL TEST
            </button>
          </div>
        )}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 34px 1fr', gap: 0, alignItems: 'stretch' }}>
          <div
            style={{
              border: '1px solid rgba(31,95,191,.4)',
              borderRadius: 8,
              padding: '13px 14px',
              background: 'rgba(31,95,191,.04)',
            }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <span style={{ font: "600 13px 'Barlow Condensed',sans-serif", letterSpacing: '.2em', color: PALETTE.refusal }}>INTACT</span>
              <span
                style={{
                  display: 'inline-block',
                  padding: '2px 9px',
                  borderRadius: 3,
                  font: "700 9px 'JetBrains Mono',monospace",
                  letterSpacing: '.06em',
                  background: 'rgba(31,95,191,.16)',
                  color: PALETTE.refusal,
                }}
              >
                REFUSES
              </span>
            </div>
            <svg width="100%" height="26" viewBox="0 0 180 26" preserveAspectRatio="none" style={{ display: 'block', margin: '10px 0 4px' }}>
              <path d="M6 13 H174" stroke={PALETTE.refusal} strokeWidth={3} fill="none" strokeLinecap="round" />
              <circle cx={90} cy={13} r={4} fill={PALETTE.refusal} />
            </svg>
            <div style={{ display: 'grid', gridTemplateColumns: '48px 1fr 32px', gap: 7, alignItems: 'center', font: "400 10px 'JetBrains Mono',monospace", color: PALETTE.dim, marginTop: 10 }}>
              <span>danger</span>
              <div style={{ height: 8, background: PALETTE.segOff, borderRadius: 3, overflow: 'hidden' }}>
                <div style={{ height: '100%', width: `${Math.round(intact.danger_norm * 100)}%`, background: PALETTE.danger }} />
              </div>
              <span style={{ color: PALETTE.danger }}>{fmt2(intact.danger_norm)}</span>
              <span>refusal</span>
              <div style={{ height: 8, background: PALETTE.segOff, borderRadius: 3, overflow: 'hidden' }}>
                <div style={{ height: '100%', width: `${Math.round(intact.refusal_norm * 100)}%`, background: PALETTE.refusal }} />
              </div>
              <span style={{ color: PALETTE.refusal }}>{fmt2(intact.refusal_norm)}</span>
            </div>
            <div style={{ font: "400 11.5px/1.55 'JetBrains Mono',monospace", color: '#544c3d', marginTop: 11, borderTop: `1px solid ${PALETTE.hair}`, paddingTop: 10 }}>
              “{intact.response}”
            </div>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: 6 }}>
            <div style={{ font: "700 15px 'Barlow Condensed',sans-serif", letterSpacing: '.1em', color: PALETTE.danger }}>✂</div>
            <div style={{ width: 1, flex: 1, background: `linear-gradient(${PALETTE.border},${PALETTE.danger},${PALETTE.border})` }} />
            <div style={{ font: "600 9px 'Barlow Condensed',sans-serif", letterSpacing: '.1em', color: PALETTE.danger, writingMode: 'vertical-rl', transform: 'rotate(180deg)' }}>
              CUT
            </div>
            <div style={{ width: 1, flex: 1, background: `linear-gradient(${PALETTE.border},${PALETTE.danger},${PALETTE.border})` }} />
          </div>

          <div style={{ border: '1px solid rgba(191,59,33,.45)', borderRadius: 8, padding: '13px 14px', background: 'rgba(191,59,33,.05)' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <span style={{ font: "600 13px 'Barlow Condensed',sans-serif", letterSpacing: '.2em', color: PALETTE.danger }}>SEVERED</span>
              <span
                style={{
                  display: 'inline-block',
                  padding: '2px 9px',
                  borderRadius: 3,
                  font: "700 9px 'JetBrains Mono',monospace",
                  letterSpacing: '.06em',
                  background: 'rgba(191,59,33,.16)',
                  color: PALETTE.danger,
                }}
              >
                COMPLIES
              </span>
            </div>
            <svg width="100%" height="26" viewBox="0 0 180 26" preserveAspectRatio="none" style={{ display: 'block', margin: '10px 0 4px' }}>
              <path d="M6 13 H74" stroke={PALETTE.wireGray} strokeWidth={3} fill="none" strokeLinecap="round" />
              <path d="M106 13 H174" stroke={PALETTE.wireGray} strokeWidth={3} fill="none" strokeLinecap="round" />
              <circle cx={84} cy={13} r={1.9} fill={PALETTE.gold} />
              <circle cx={90} cy={9} r={1.6} fill={PALETTE.gold} />
              <circle cx={96} cy={15} r={1.9} fill={PALETTE.gold} />
            </svg>
            <div style={{ display: 'grid', gridTemplateColumns: '48px 1fr 32px', gap: 7, alignItems: 'center', font: "400 10px 'JetBrains Mono',monospace", color: PALETTE.dim, marginTop: 10 }}>
              <span>danger</span>
              <div style={{ height: 8, background: PALETTE.segOff, borderRadius: 3, overflow: 'hidden' }}>
                <div style={{ height: '100%', width: `${Math.round(severed.danger_norm * 100)}%`, background: PALETTE.danger }} />
              </div>
              <span style={{ color: PALETTE.danger }}>{fmt2(severed.danger_norm)}</span>
              <span>refusal</span>
              <div style={{ height: 8, background: PALETTE.segOff, borderRadius: 3, overflow: 'hidden' }}>
                <div style={{ height: '100%', width: `${Math.round(severed.refusal_norm * 100)}%`, background: PALETTE.refusal }} />
              </div>
              <span style={{ color: PALETTE.dim }}>{fmt2(severed.refusal_norm)}</span>
            </div>
            <div style={{ font: "400 11.5px/1.55 'JetBrains Mono',monospace", color: '#544c3d', marginTop: 11, borderTop: `1px solid ${PALETTE.hair}`, paddingTop: 10 }}>
              “{severed.response}”
            </div>
          </div>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginTop: 14, borderTop: `1px solid ${PALETTE.hair}`, paddingTop: 12 }}>
          <span style={{ font: "700 11px 'Barlow Condensed',sans-serif", letterSpacing: '.14em', color: PALETTE.danger, flex: 'none' }}>
            DANGER {fmt2(intact.danger_norm)} → {fmt2(severed.danger_norm)}
          </span>
          <span style={{ color: PALETTE.borderAlt }}>·</span>
          <span style={{ font: "400 12px/1.5 'JetBrains Mono',monospace", color: '#544c3d' }}>
            Danger stays firing hard — never near zero — while refusal collapses. Recognition and refusal are{' '}
            <span style={{ color: PALETTE.ink }}>separable</span>.
          </span>
        </div>
      </div>
    </div>
  )
}
