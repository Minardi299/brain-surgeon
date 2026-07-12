import { PALETTE } from '../constants'

// Fixed bar geometry (matches src/danger_tracker/app.py's _render_panel05 exactly) — only
// the label/verdict/color per bar come from GET /api/layer_sweep; positions are constant.
const XS = [58, 128, 198, 268, 338, 408]
const YS = [32, 32, 39, 102, 141, 158]
const HEIGHTS = [140, 140, 133, 70, 31, 14]

// 05 — LAYER SWEEP: a measured, static panel (not recomputed live — six generations per
// view would be too slow for a demo booth). Fetched once from GET /api/layer_sweep.
export default function Panel05({ layers }) {
  const bars = (layers ?? []).slice(0, 6)
  return (
    <div>
      <div style={{ font: "600 15px 'Barlow Condensed',sans-serif", letterSpacing: '.24em', color: PALETTE.danger, margin: '20px 0 6px' }}>
        05 — LAYER SWEEP{' '}
        <span style={{ color: PALETTE.faint, letterSpacing: '.08em', fontSize: 12 }}>
          where the cut works · the source layer we ablate at · measured
        </span>
      </div>
      <div style={{ background: '#fcfaf4', border: '1px solid #d9d2c2', borderRadius: 8, padding: '14px 16px 12px' }}>
        <svg width={520} height={196} viewBox="0 0 520 196" style={{ display: 'block', maxWidth: '100%' }}>
          <rect x={58} y={26} width={182} height={146} fill="rgba(31,95,191,0.07)" />
          <text x={149} y={20} textAnchor="middle" style={{ font: "9px 'JetBrains Mono',monospace" }} fill={PALETTE.refusal}>
            refusal direction lives @ L8–12
          </text>
          <path d="M44 172 H500" stroke={PALETTE.borderAlt} strokeWidth={1} />
          <text x={40} y={36} textAnchor="end" style={{ font: "9px 'JetBrains Mono',monospace" }} fill={PALETTE.dim}>
            flips
          </text>
          <text x={40} y={170} textAnchor="end" style={{ font: "9px 'JetBrains Mono',monospace" }} fill={PALETTE.dim}>
            holds
          </text>
          {bars.map((b, i) => (
            <g key={b.label}>
              <rect x={XS[i]} y={YS[i]} width={40} height={HEIGHTS[i]} rx={3} fill={b.color} opacity={0.85} />
              <text x={XS[i] + 20} y={YS[i] - 6} textAnchor="middle" fill={b.color} style={{ font: "700 8px 'JetBrains Mono',monospace" }}>
                {b.verdict}
              </text>
              <text x={XS[i] + 20} y={186} textAnchor="middle" fill={PALETTE.dim} style={{ font: "600 11px 'JetBrains Mono',monospace" }}>
                {b.label}
              </text>
            </g>
          ))}
        </svg>
        <div style={{ font: "400 11px/1.7 'JetBrains Mono',monospace", color: '#544c3d', marginTop: 8 }}>
          Cutting the refusal direction flips refuse→comply only at <span style={{ color: PALETTE.danger }}>early-middle layers 8–12</span>;
          by <span style={{ color: PALETTE.gold }}>14</span> the effect is partial, and at{' '}
          <span style={{ color: PALETTE.refusal }}>16–18</span> the model re-refuses.{' '}
          <span style={{ color: PALETTE.dim }}>(Real layer sweep — measured, not recomputed live: refusal_dir.log / verify_dir2.log.)</span>
        </div>
      </div>
    </div>
  )
}
