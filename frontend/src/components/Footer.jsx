import { PALETTE } from '../constants'

export default function Footer() {
  return (
    <div
      style={{
        marginTop: 22,
        borderTop: `1px solid ${PALETTE.border}`,
        paddingTop: 12,
        font: "400 11px 'JetBrains Mono',monospace",
        color: PALETTE.faint,
      }}
    >
      A ringing alarm is not a wired brake — the SAE features report the danger; the refusal direction acts on it. Numbers measured on
      Gemma-2-2b-it + Gemma Scope (L20); the danger and refusal meters are each normalized to their own firing range.
    </div>
  )
}
