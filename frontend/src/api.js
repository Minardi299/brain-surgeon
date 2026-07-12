// API client for the FastAPI backend (src/danger_tracker/api.py). Base path is relative
// "/api" — in dev, vite.config.js proxies that to http://localhost:8000; in production
// FastAPI serves this SPA from frontend/dist and answers /api/* itself, same origin.
//
// MOCK mode: when import.meta.env.VITE_MOCK === "1", every call resolves from canned data
// (src/mockData.js) instead of hitting the network, so `VITE_MOCK=1 npm run dev` runs the
// whole board offline. A small artificial delay simulates the real "running model…" beat.

import { MOCK_BEATS, MOCK_EXPERIMENTS, MOCK_LAYER_SWEEP, mockAnalyze } from './mockData'

const MOCK = import.meta.env.VITE_MOCK === '1'
const MOCK_DELAY_MS = 500

function delay(value) {
  return new Promise((resolve) => setTimeout(() => resolve(value), MOCK_DELAY_MS))
}

async function post(path, body) {
  const res = await fetch(`/api${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!res.ok) {
    throw new Error(`${path} failed: ${res.status} ${res.statusText}`)
  }
  return res.json()
}

async function get(path) {
  const res = await fetch(`/api${path}`)
  if (!res.ok) {
    throw new Error(`${path} failed: ${res.status} ${res.statusText}`)
  }
  return res.json()
}

export async function getHealth() {
  if (MOCK) return delay({ ok: true, cached_beats: Object.keys(MOCK_BEATS) })
  return get('/health')
}

export async function getLayerSweep() {
  if (MOCK) return delay({ layers: MOCK_LAYER_SWEEP })
  return get('/layer_sweep')
}

export async function postBeat(kind) {
  if (MOCK) {
    const beat = MOCK_BEATS[kind]
    if (!beat) throw new Error(`unknown mock beat kind ${kind}`)
    return delay(beat)
  }
  return post('/beat', { kind })
}

export async function postCut() {
  if (MOCK) return delay(MOCK_BEATS.cut)
  return post('/cut', {})
}

export async function postAnalyze(prompt) {
  if (MOCK) return delay(mockAnalyze(prompt))
  return post('/analyze', { prompt })
}

export async function postExperiment(which) {
  if (MOCK) {
    const exp = MOCK_EXPERIMENTS[which]
    if (!exp) throw new Error(`unknown mock experiment ${which}`)
    return delay(exp)
  }
  return post('/experiment', { which })
}

export const isMock = MOCK
