import { PALETTE } from '../constants'

function verdictBadge(verdict) {
  const complies = /COMPLIES/.test(verdict || '')
  return {
    bg: complies ? 'rgba(191,59,33,.16)' : 'rgba(31,95,191,.14)',
    fg: complies ? PALETTE.danger : PALETTE.refusal,
  }
}

function ExperimentBox({ title, subtitle, exp, onToggle, loading, explanation }) {
  const badge = exp?.ran ? verdictBadge(exp.verdict) : null
  return (
    <div style={{ background: '#fcfaf4', border: '1px solid #d9d2c2', borderRadius: 8, padding: '13px 16px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 10 }}>
        <div style={{ font: "600 16px 'Barlow Condensed',sans-serif", letterSpacing: '.04em' }}>
          {title} <span style={{ font: "400 10px 'JetBrains Mono',monospace", color: PALETTE.dim }}>{subtitle}</span>
        </div>
        <button
          onClick={onToggle}
          disabled={loading}
          style={{
            cursor: loading ? 'default' : 'pointer',
            flex: 'none',
            font: "700 10px 'JetBrains Mono',monospace",
            padding: '7px 12px',
            borderRadius: 4,
            border: `1px solid ${PALETTE.borderAlt}`,
            background: 'transparent',
            color: PALETTE.ink,
            opacity: loading ? 0.6 : 1,
          }}
        >
          {exp?.ran ? 'RESET' : 'RUN ABLATION'}
        </button>
      </div>
      {exp?.ran && (
        <div style={{ marginTop: 10, borderTop: `1px solid ${PALETTE.hair}`, paddingTop: 10, font: "400 11.5px/1.9 'JetBrains Mono',monospace", color: '#544c3d' }}>
          refusal signal <span style={{ color: PALETTE.refusal }}>{exp.refusal_before.toFixed(1)} → {exp.refusal_after.toFixed(1)}</span> ·
          danger <span style={{ color: PALETTE.danger }}>{exp.danger_before.toFixed(1)} → {exp.danger_after.toFixed(1)}</span>{' '}
          <span
            style={{ display: 'inline-block', padding: '1px 8px', borderRadius: 3, background: badge.bg, color: badge.fg, fontWeight: 700 }}
          >
            {exp.verdict}
          </span>
          <div style={{ color: PALETTE.danger, marginTop: 4 }}>{explanation}</div>
        </div>
      )}
    </div>
  )
}

// 04 — CORRELATION VS CAUSATION: two independent A/B ablation experiments against the
// harmful beat, each toggled via its own RUN/RESET button and rendering the returned
// ExpDict inline.
export default function Panel04({ expA, expB, onRunA, onRunB, loading }) {
  return (
    <div>
      <div style={{ font: "600 15px 'Barlow Condensed',sans-serif", letterSpacing: '.24em', color: PALETTE.danger, margin: '20px 0 6px' }}>
        04 — CORRELATION VS CAUSATION{' '}
        <span style={{ color: PALETTE.faint, letterSpacing: '.08em', fontSize: 12 }}>the same refusal, removed two ways</span>
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
        <ExperimentBox
          title={
            <>
              A — SILENCE THE SAE <span style={{ color: PALETTE.refusal }}>REFUSAL</span> FEATURES
            </>
          }
          subtitle="clamp #9989 · #10156 · #13364 @ L20"
          exp={expA}
          onToggle={onRunA}
          loading={loading}
          explanation={
            <>
              → correlational: the labelled "refusal" feature only <em>reads</em> refusal. Drive it to zero and behaviour never changes —
              the blinking wire is inert.
            </>
          }
        />
        <ExperimentBox
          title="B — CUT THE REFUSAL DIRECTION"
          subtitle="subtract @ L8–12"
          exp={expB}
          onToggle={onRunB}
          loading={loading}
          explanation={
            <>
              → causal: this one direction <em>performs</em> the refusal. Cut it and behaviour flips — while danger stays firing hard.
            </>
          }
        />
      </div>
      <div style={{ marginTop: 12, font: "400 12px/1.55 'JetBrains Mono',monospace", color: '#544c3d', borderLeft: `2px solid ${PALETTE.danger}`, paddingLeft: 12 }}>
        Same refusal removed two ways. The feature that <span style={{ color: PALETTE.ink }}>reads</span> refusal is not the mechanism
        that <span style={{ color: PALETTE.ink }}>performs</span> it — same signal to zero, opposite outcome.
      </div>
    </div>
  )
}
