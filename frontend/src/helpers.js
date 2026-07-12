import { ACTIVATION_SCALE } from './constants'

// 18-segment bar-graph fill count for a 0..1 normalized value (GAP hero meters).
export function segCount(v) {
  return Math.round((v ?? 0) * 18)
}

// Token-cell background/foreground for a category heatmap row, matching the design's
// per-cell alpha blend (rows[i].rgb is "r,g,b" as returned by the backend TokenRow).
export function cellStyle(rgb, activation, alphaMax = 0.85, strongFg = '#fbf7ef', fg = '#211d16') {
  const a = Math.min(1, (activation ?? 0) / ACTIVATION_SCALE)
  return {
    bg: `rgba(${rgb},${(a * alphaMax).toFixed(2)})`,
    fg: a > 0.55 ? strongFg : fg,
  }
}

export function pct(value, scale = ACTIVATION_SCALE) {
  return Math.round(((value ?? 0) / scale) * 100) + '%'
}

export function gapStr(gap) {
  const g = gap ?? 0
  return (g >= 0 ? '+' : '') + g.toFixed(2)
}

export function fmt2(v) {
  return (v ?? 0).toFixed(2)
}

// Status pill colour classification — mirrors the design's build(): ok when nothing is
// firing, refusal-blue when the status text mentions REFUSE, danger-red otherwise (complies).
export function statusColors(dangerNorm, status, palette) {
  if ((dangerNorm ?? 0) < 0.15) return { bg: palette.okBg, fg: palette.ok }
  if (/REFUSE/.test(status || '')) return { bg: palette.refusalBg, fg: palette.refusal }
  return { bg: palette.alarmBg, fg: palette.danger }
}

// Wire state for the GAP hero + panel 00 spine: slack (benign) / taut (harmful) /
// strained (jailbreak) / severed (cut). Derived from the active beat rather than a fixed
// scenario index, since any beat (including a custom analyzed prompt) can be active.
export function wireKindFor(beat, severed) {
  if (severed || beat?.kind === 'cut') return 'severed'
  const danger = beat?.danger_norm ?? 0
  if (danger < 0.15) return 'slack'
  if (beat?.kind === 'jailbreak') return 'strained'
  return 'taut'
}
