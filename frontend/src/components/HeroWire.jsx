import { PALETTE } from '../constants'

// Ported from the design's heroWire(kind, S) — the small SVG under "THE GAP" showing the
// refusal-direction wire: slack (benign) / taut (harmful) / strained (jailbreak) / severed (cut).
export default function HeroWire({ kind }) {
  const W = 260
  const H = 48
  const y = 24
  const x1 = 12
  const x2 = 248
  const cRef = PALETTE.refusal
  const gray = PALETTE.gray
  const gold = PALETTE.gold

  if (kind === 'slack') {
    return (
      <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} style={{ display: 'block', overflow: 'visible' }}>
        <path
          d={`M${x1} ${y - 3} C 80 ${y + 16}, 180 ${y + 16}, ${x2} ${y - 3}`}
          fill="none"
          stroke={gray}
          strokeWidth={2}
          strokeLinecap="round"
          strokeDasharray="2 6"
          opacity={0.5}
        />
        <circle cx={x1} cy={y - 3} r={3} fill={gray} opacity={0.6} />
        <circle cx={x2} cy={y - 3} r={3} fill={gray} opacity={0.6} />
        <text x={W / 2} y={y + 26} textAnchor="middle" fill={gray} style={{ font: "500 9px 'JetBrains Mono',monospace", letterSpacing: '.08em' }}>
          no current
        </text>
      </svg>
    )
  }

  if (kind === 'taut') {
    return (
      <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} style={{ display: 'block', overflow: 'visible' }}>
        <path d={`M${x1} ${y} H${x2}`} fill="none" stroke={cRef} strokeWidth={8} strokeLinecap="round" opacity={0.14} />
        <path d={`M${x1} ${y} H${x2}`} fill="none" stroke={cRef} strokeWidth={3} strokeLinecap="round" />
        <path
          d={`M${x1} ${y} H${x2}`}
          fill="none"
          stroke="#fff"
          strokeWidth={3}
          strokeLinecap="round"
          strokeDasharray="3 9"
          opacity={0.85}
          style={{ animation: 'omFlow 1.1s linear infinite' }}
        />
        <circle cx={x1} cy={y} r={4} fill={cRef} />
        <circle cx={x2} cy={y} r={4} fill={cRef} />
        <text x={W / 2} y={y + 26} textAnchor="middle" fill={cRef} style={{ font: "600 9px 'JetBrains Mono',monospace", letterSpacing: '.06em' }}>
          current flowing · brake wired
        </text>
      </svg>
    )
  }

  if (kind === 'strained') {
    return (
      <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} style={{ display: 'block', overflow: 'visible' }}>
        <g style={{ animation: 'omStrain 0.35s ease-in-out infinite' }}>
          <path d={`M${x1} ${y} Q130 ${y + 9} ${x2} ${y}`} fill="none" stroke={cRef} strokeWidth={8} strokeLinecap="round" opacity={0.14} />
          <path d={`M${x1} ${y} Q130 ${y + 9} ${x2} ${y}`} fill="none" stroke={cRef} strokeWidth={3} strokeLinecap="round" />
          <path
            d={`M${x1} ${y} Q130 ${y + 9} ${x2} ${y}`}
            fill="none"
            stroke="#fff"
            strokeWidth={3}
            strokeLinecap="round"
            strokeDasharray="3 9"
            opacity={0.85}
            style={{ animation: 'omFlow 0.7s linear infinite' }}
          />
        </g>
        <path
          d={`M130 ${y + 12} v9 m-4 -4 l4 4 l4 -4`}
          fill="none"
          stroke={gray}
          strokeWidth={1.4}
          strokeLinecap="round"
          strokeLinejoin="round"
        />
        <circle cx={x1} cy={y} r={4} fill={cRef} />
        <circle cx={x2} cy={y} r={4} fill={cRef} />
        <text x={W / 2} y={y + 30} textAnchor="middle" fill={cRef} style={{ font: "600 9px 'JetBrains Mono',monospace", letterSpacing: '.06em' }}>
          strains — but holds
        </text>
      </svg>
    )
  }

  // severed
  return (
    <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} style={{ display: 'block', overflow: 'visible' }}>
      <path d={`M${x1} ${y} H108`} fill="none" stroke={gray} strokeWidth={3} strokeLinecap="round" opacity={0.7} />
      <path d={`M152 ${y} H${x2}`} fill="none" stroke={gray} strokeWidth={3} strokeLinecap="round" opacity={0.7} />
      {[0, 1, 2, 3, 4].map((i) => (
        <circle
          key={i}
          cx={120 + i * 5}
          cy={y}
          r={1.8}
          fill={gold}
          style={{ transformOrigin: `${120 + i * 5}px ${y}px`, animation: `omSpark 1.1s ease-out ${(i * 0.12).toFixed(2)}s infinite` }}
        />
      ))}
      <circle cx={x1} cy={y} r={4} fill={gray} />
      <circle cx={x2} cy={y} r={4} fill={gray} />
      <text x={W / 2} y={y + 26} textAnchor="middle" fill={gold} style={{ font: "700 9px 'JetBrains Mono',monospace", letterSpacing: '.06em' }}>
        severed by us — restorable
      </text>
    </svg>
  )
}
