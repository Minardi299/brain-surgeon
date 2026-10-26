import HeroWire from './HeroWire'
import { BEAT_LABELS, BEAT_ORDER, PALETTE } from '../constants'
import { fmt2, gapStr, segCount, statusColors, wireKindFor } from '../helpers'

function Segments({ value, on }) {
  const filled = segCount(value)
  return (
    <div style={{ display: 'flex', flexDirection: 'column-reverse', gap: 3, height: 190, width: 44 }}>
      {Array.from({ length: 18 }, (_, i) => (
        <div
          key={i}
          style={{ flex: 1, borderRadius: 2, background: i < filled ? on : PALETTE.segOff, transition: 'background .5s' }}
        />
      ))}
    </div>
  )
}

// The GAP hero — the centrepiece of the board: two 18-segment meters (danger recognized /
// refusal engaged), the wire SVG between them, the status pill, the beat stepper, and play.
export default function GapHero({ beat, activeKind, severed, playing, loading, onSelectBeat, onTogglePlay }) {
  const danger = beat?.danger_norm ?? 0
  const refusal = beat?.refusal_norm ?? 0
  const gap = beat?.gap ?? danger - refusal
  const status = beat?.status ?? ''
  const { bg: statusBg, fg: statusFg } = statusColors(danger, status, PALETTE)
  const wireKind = wireKindFor(beat, severed)

  return (
    <div>
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: '150px 1fr 150px',
          gap: 22,
          alignItems: 'stretch',
          marginTop: 6,
          background: '#fcfaf4',
          border: '1px solid #d9d2c2',
          borderRadius: 10,
          padding: '22px 30px',
          position: 'relative',
        }}
      >
        {loading && (
          <div
            style={{
              position: 'absolute',
              top: 8,
              right: 14,
              font: "600 10px 'JetBrains Mono',monospace",
              letterSpacing: '.08em',
              color: PALETTE.dim,
              animation: 'runPulse 1.1s ease-in-out infinite',
            }}
          >
            ● running model…
          </div>
        )}
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 10 }}>
          <Segments value={danger} on={PALETTE.danger} />
          <div style={{ font: "700 28px 'JetBrains Mono',monospace", color: PALETTE.danger }}>{fmt2(danger)}</div>
          <div
            style={{
              font: "600 13px 'Barlow Condensed',sans-serif",
              letterSpacing: '.16em',
              color: PALETTE.danger,
              textAlign: 'center',
            }}
          >
            DANGER
            <br />
            RECOGNIZED
          </div>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', textAlign: 'center' }}>
          <div style={{ font: "600 14px 'Barlow Condensed',sans-serif", letterSpacing: '.3em', color: PALETTE.dim }}>THE GAP</div>
          <div style={{ font: "700 96px/1 'JetBrains Mono',monospace", color: PALETTE.ink, marginTop: 4 }}>{gapStr(gap)}</div>
          <div
            style={{
              font: "600 18px 'Barlow Condensed',sans-serif",
              letterSpacing: '.18em',
              textTransform: 'uppercase',
              color: PALETTE.danger,
              marginTop: 8,
            }}
          >
            {beat?.gap_label ?? ''}
          </div>
          <div style={{ marginTop: 10 }}>
            <HeroWire kind={wireKind} />
          </div>
          
          <div
            style={{
              marginTop: 12,
              display: 'inline-block',
              padding: '6px 16px',
              borderRadius: 4,
              font: "700 12px 'JetBrains Mono',monospace",
              letterSpacing: '.1em',
              background: statusBg,
              color: statusFg,
              transition: 'all .4s',
            }}
          >
            {status}
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginTop: 16 }}>
            {BEAT_ORDER.map((kind) => {
              const on = activeKind === kind
              return (
                <button
                  key={kind}
                  onClick={() => onSelectBeat(kind)}
                  style={{
                    cursor: 'pointer',
                    font: "600 15px 'Barlow Condensed',sans-serif",
                    letterSpacing: '.1em',
                    textTransform: 'uppercase',
                    padding: '9px 16px',
                    borderRadius: 4,
                    border: `1px solid ${on ? PALETTE.ink : PALETTE.border}`,
                    background: on ? PALETTE.ink : 'transparent',
                    color: on ? '#f5f1e8' : PALETTE.dim,
                    transition: 'all .25s',
                  }}
                >
                  {BEAT_LABELS[kind]}
                </button>
              )
            })}
            <button
              onClick={onTogglePlay}
              style={{
                cursor: 'pointer',
                font: "600 12px 'JetBrains Mono',monospace",
                padding: '9px 14px',
                borderRadius: 4,
                border: '1px dashed #cfc7b4',
                background: 'transparent',
                color: PALETTE.dim,
              }}
            >
              {playing ? '❚❚ pause' : '▶ play'}
            </button>
          </div>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 10 }}>
          <Segments value={refusal} on={PALETTE.refusal} />
          <div style={{ font: "700 28px 'JetBrains Mono',monospace", color: PALETTE.refusal }}>{fmt2(refusal)}</div>
          <div
            style={{
              font: "600 13px 'Barlow Condensed',sans-serif",
              letterSpacing: '.16em',
              color: PALETTE.refusal,
              textAlign: 'center',
            }}
          >
            REFUSAL
            <br />
            ENGAGED
          </div>
        </div>
      </div>
      {beat?.caption && (
        <div style={{ marginTop: 12, font: "italic 400 15px/1.5 'Newsreader',serif", color: '#544c3d', maxWidth: 940 }}>
          {beat.caption}
        </div>
      )}
    </div>
  )
}
