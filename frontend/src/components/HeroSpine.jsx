import { CATEGORY_RGB } from '../constants'

function hexA(hex, a) {
  const n = parseInt(hex.slice(1), 16)
  return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`
}

function catColor(cat) {
  const rgb = CATEGORY_RGB[cat]
  return rgb ? `rgb(${rgb})` : '#bf3b21'
}

// Builds the same pseudo-activation curve as the design's heroDataFromScenario(): a
// residual-stream "glow" profile across 26 layers, peaked at the L20 SAE readout and the
// L8-12 intervention band, scaled by how hard danger is firing on the active beat.
function layerGlow(danger) {
  const layers = []
  for (let l = 0; l < 26; l++) {
    let a = 0.06 + 0.04 * Math.sin(l * 0.7)
    a += danger * 0.55 * Math.exp(-((l - 20) ** 2) / 42)
    a += danger * 0.26 * Math.exp(-((l - 11) ** 2) / 34)
    layers.push(Math.min(1, a))
  }
  return layers
}

// Panel 00 — "residual stream" SVG. Ported from the design's heroSpine(). Category feature
// dots are driven by the active beat's alarm_chips (the 4 real harm categories), not the
// design mock's 5 invented ones.
export default function HeroSpine({ beat, severed }) {
  const danger = beat?.danger_norm ?? 0
  const refusal = severed ? 0.05 : beat?.refusal_norm ?? 0
  const dangerOn = danger > 0.3
  const refusalOn = !severed && refusal > 0.4
  const layers = layerGlow(danger)
  const feats = (beat?.alarm_chips ?? [])
    .map((c) => ({ k: c.category, s: Math.min(1, (c.raw ?? 0) / 150), c: catColor(c.category) }))
    .filter((f) => f.s > 0.12)

  const cFg = '#211d16'
  const cDim = '#6f6656'
  const cSpine = '#e6dfd0'
  const cNode = '#e6dfd0'
  const cDanger = '#bf3b21'
  const cRefusal = '#1f5fbf'
  const cObs = '#bf3b21'
  const cObs2 = '#6f6656'
  const cRise = '#bf3b21'
  const cRiseCold = '#6f6656'
  const cDot = '#211d16'

  const W = 400
  const H = 486
  const cx = 196
  const yTop = 56
  const yBot = 444
  const ly = (l) => yBot - (l / 25) * (yBot - yTop)
  const y20 = ly(20)
  const top = ly(12)
  const bot = ly(8)
  const bh = bot - top
  const mid = (top + bot) / 2

  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" preserveAspectRatio="xMidYMid meet" style={{ display: 'block' }}>
      <text x={20} y={27} fill={cFg} style={{ font: "600 13px 'Space Grotesk',sans-serif", letterSpacing: '.12em' }}>
        RESIDUAL STREAM
      </text>
      <text x={20} y={43} fill={cDim} style={{ font: "400 10px 'JetBrains Mono',monospace" }}>
        2304-dim · 26 layers
      </text>
      <rect x={cx - 6} y={yTop} width={12} height={yBot - yTop} rx={6} fill={cSpine} />

      {danger > 0.1 && (
        <g opacity={danger}>
          <circle cx={cx} cy={y20} r={78} fill={cDanger} opacity={0.13} />
          <circle cx={cx} cy={y20} r={78 * 0.62} fill={cDanger} opacity={0.2} />
          <circle cx={cx} cy={y20} r={78 * 0.34} fill={cDanger} opacity={0.42} />
        </g>
      )}

      {layers.map((a, l) => {
        const y = ly(l)
        return (
          <g key={l}>
            <circle cx={cx} cy={y} r={3.2} fill={cNode} />
            {a > 0.16 && (
              <circle
                cx={cx}
                cy={y}
                r={3.2 + a * 3.6}
                fill={cDanger}
                opacity={Math.min(0.95, a * 1.1)}
                style={{ transition: 'all .5s' }}
              />
            )}
            {l % 5 === 0 && (
              <text x={cx - 74} y={y + 3.5} fill="#a49a83" style={{ font: "400 9px 'JetBrains Mono',monospace" }}>
                L{l}
              </text>
            )}
          </g>
        )
      })}

      {[0, 1, 2, 3, 4, 5].map((i) => (
        <circle
          key={i}
          cx={cx}
          cy={yBot}
          r={2.6}
          fill={dangerOn ? cRise : cRiseCold}
          style={{ animation: `omRise 3.2s linear ${(i * 0.53).toFixed(2)}s infinite` }}
        />
      ))}

      <ellipse
        cx={cx}
        cy={y20}
        rx={48}
        ry={17}
        fill="none"
        stroke={cObs}
        strokeWidth={1.6}
        strokeDasharray="5 5"
        opacity={0.9}
        style={{ animation: 'omDash 6s linear infinite' }}
      />
      <text x={cx + 58} y={y20 - 5} fill={cObs} style={{ font: "600 10px 'JetBrains Mono',monospace", letterSpacing: '.06em' }}>
        OBSERVE
      </text>
      <text x={cx + 58} y={y20 + 8} fill={cObs2} style={{ font: "400 9px 'JetBrains Mono',monospace" }}>
        SAE @ L20
      </text>

      {feats.map((f, i) => {
        const ang = -Math.PI / 2 + (i - (feats.length - 1) / 2) * 0.64
        const x = cx + Math.cos(ang) * 54
        const y = y20 + Math.sin(ang) * 21
        const r = 10 * f.s + 4
        return (
          <g key={f.k}>
            <circle cx={x} cy={y} r={r} fill={f.c} opacity={f.s * 0.13} />
            <circle cx={x} cy={y} r={r * 0.62} fill={f.c} opacity={f.s * 0.2} />
            <circle cx={x} cy={y} r={r * 0.34} fill={f.c} opacity={f.s * 0.42} />
            <circle cx={x} cy={y} r={3.4} fill={cDot} />
            <circle
              cx={x}
              cy={y}
              r={6 * f.s + 3}
              fill="none"
              stroke={f.c}
              strokeWidth={1.4}
              opacity={0.9}
              style={{ transformOrigin: `${x}px ${y}px`, animation: `omFlare 1.9s ease-in-out ${(i * 0.2).toFixed(2)}s infinite` }}
            />
          </g>
        )
      })}

      <text x={cx - 96} y={mid - 4} fill={severed ? '#8a949f' : cObs} style={{ font: "600 10px 'JetBrains Mono',monospace", letterSpacing: '.04em' }}>
        INTERVENE
      </text>
      <text x={cx - 96} y={mid + 9} fill={cObs2} style={{ font: "400 9px 'JetBrains Mono',monospace" }}>
        refusal dir
      </text>
      <text x={cx - 96} y={mid + 21} fill={cObs2} style={{ font: "400 9px 'JetBrains Mono',monospace" }}>
        @ L8–12
      </text>

      {!severed ? (
        <>
          <rect
            x={cx - 17}
            y={top}
            width={34}
            height={bh}
            rx={9}
            fill={hexA(cRefusal, 0.16)}
            stroke={cRefusal}
            strokeWidth={1.8}
            opacity={refusalOn ? 1 : 0.4}
            style={{ transition: 'all .5s', animation: refusalOn ? 'omPulseGlow 2.2s ease-in-out infinite' : 'none' }}
          />
          <text x={cx + 26} y={mid + 4} fill={refusalOn ? cRefusal : '#a49a83'} style={{ font: "600 10px 'JetBrains Mono',monospace" }}>
            {refusalOn ? 'ENGAGED' : 'IDLE'}
          </text>
        </>
      ) : (
        <>
          <g style={{ transform: 'translate(-9px,-11px)', transition: 'transform .55s cubic-bezier(.6,0,.3,1)' }}>
            <rect x={cx - 17} y={top} width={34} height={bh / 2 - 3} rx={8} fill="rgba(120,130,145,0.16)" stroke="#6b7683" strokeWidth={1.4} />
          </g>
          <g style={{ transform: 'translate(9px,11px)', transition: 'transform .55s cubic-bezier(.6,0,.3,1)' }}>
            <rect x={cx - 17} y={top + bh / 2 + 3} width={34} height={bh / 2 - 3} rx={8} fill="rgba(120,130,145,0.16)" stroke="#6b7683" strokeWidth={1.4} />
          </g>
          {[0, 1, 2, 3, 4].map((i) => (
            <circle
              key={i}
              cx={cx - 8 + i * 4}
              cy={mid}
              r={1.9}
              fill="#a86a12"
              style={{ transformOrigin: `${cx - 8 + i * 4}px ${mid}px`, animation: `omSpark 1.1s ease-out ${(i * 0.12).toFixed(2)}s infinite` }}
            />
          ))}
          <text x={cx + 26} y={mid + 4} fill="#a86a12" style={{ font: "700 10px 'JetBrains Mono',monospace" }}>
            SEVERED
          </text>
        </>
      )}
    </svg>
  )
}
