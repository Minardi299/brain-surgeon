import { useCallback, useEffect, useRef, useState } from 'react'
import Header from './components/Header'
import AlarmBoard from './components/AlarmBoard'
import GapHero from './components/GapHero'
import TryYourOwn from './components/TryYourOwn'
import PromptOutput from './components/PromptOutput'
import Panel00 from './components/Panel00'
import Panel01 from './components/Panel01'
import Panel02 from './components/Panel02'
import Panel03 from './components/Panel03'
import Panel04 from './components/Panel04'
import Panel05 from './components/Panel05'
import Footer from './components/Footer'
import { getLayerSweep, isMock, postAnalyze, postBeat, postCut, postExperiment } from './api'
import { BEAT_ORDER, LAYER_SWEEP_FALLBACK } from './constants'

const PLAY_MS = 2200
const SPOT_FULL = { opacity: 1, filter: 'none' }
const SPOT_DIM = { opacity: 0.6, filter: 'saturate(.72)' }

export default function App() {
  // cache of the four preset beats + the on-demand "cut" beat — kind -> BeatDict
  const [cache, setCache] = useState({})
  const [activeKind, setActiveKind] = useState('benign')
  const [activeBeat, setActiveBeat] = useState(null)
  const [cutOn, setCutOn] = useState(false)
  const [playing, setPlaying] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [expA, setExpA] = useState(null)
  const [expB, setExpB] = useState(null)
  const [layerSweep, setLayerSweep] = useState(LAYER_SWEEP_FALLBACK)

  const cacheRef = useRef(cache)
  cacheRef.current = cache

  const fetchBeat = useCallback(async (kind) => {
    if (cacheRef.current[kind]) return cacheRef.current[kind]
    const beat = await postBeat(kind)
    setCache((c) => ({ ...c, [kind]: beat }))
    return beat
  }, [])

  const selectBeat = useCallback(
    async (kind) => {
      setLoading(true)
      setError(null)
      try {
        const beat = await fetchBeat(kind)
        setActiveKind(kind)
        setActiveBeat(beat)
        setCutOn(false)
      } catch (e) {
        setError(String(e.message || e))
      } finally {
        setLoading(false)
      }
    },
    [fetchBeat]
  )

  // Initial load: default to the benign beat so the board isn't empty, and fetch the
  // (static/measured) layer sweep once.
  useEffect(() => {
    selectBeat('benign')
    getLayerSweep()
      .then((r) => setLayerSweep(r.layers ?? LAYER_SWEEP_FALLBACK))
      .catch(() => setLayerSweep(LAYER_SWEEP_FALLBACK))
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // Play — advances through the three named beats on a timer; pause toggles it off.
  useEffect(() => {
    if (!playing) return undefined
    const id = setInterval(() => {
      setActiveKind((prevKind) => {
        const i = BEAT_ORDER.indexOf(prevKind)
        const next = BEAT_ORDER[(i + 1) % BEAT_ORDER.length]
        selectBeat(next)
        return prevKind // selectBeat's own setActiveKind applies the real transition
      })
    }, PLAY_MS)
    return () => clearInterval(id)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [playing])

  const togglePlay = () => setPlaying((p) => !p)

  const toggleCut = useCallback(async () => {
    setLoading(true)
    setError(null)
    setPlaying(false)
    try {
      if (cutOn) {
        const harmful = await fetchBeat('harmful')
        setActiveKind('harmful')
        setActiveBeat(harmful)
        setCutOn(false)
      } else {
        await fetchBeat('harmful')
        let cut = cacheRef.current.cut
        if (!cut) {
          cut = await postCut()
          setCache((c) => ({ ...c, cut }))
        }
        setActiveKind('cut')
        setActiveBeat(cut)
        setCutOn(true)
      }
    } catch (e) {
      setError(String(e.message || e))
    } finally {
      setLoading(false)
    }
  }, [cutOn, fetchBeat])

  const runAnalyze = useCallback(async (prompt) => {
    setLoading(true)
    setError(null)
    setPlaying(false)
    try {
      const result = await postAnalyze(prompt)
      if (result.error) {
        setError(result.error)
        return
      }
      setActiveKind('custom')
      setActiveBeat(result)
      setCutOn(false)
    } catch (e) {
      setError(String(e.message || e))
    } finally {
      setLoading(false)
    }
  }, [])

  const toggleExperiment = useCallback(
    async (which) => {
      const setter = which === 'a' ? setExpA : setExpB
      const current = which === 'a' ? expA : expB
      if (current) {
        setter(null)
        return
      }
      setLoading(true)
      setError(null)
      try {
        const result = await postExperiment(which)
        setter(result)
      } catch (e) {
        setError(String(e.message || e))
      } finally {
        setLoading(false)
      }
    },
    [expA, expB]
  )

  const runCutTestFromPanel03 = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      await fetchBeat('harmful')
      if (!cacheRef.current.cut) {
        const cut = await postCut()
        setCache((c) => ({ ...c, cut }))
      }
    } catch (e) {
      setError(String(e.message || e))
    } finally {
      setLoading(false)
    }
  }, [fetchBeat])

  const severed = cutOn || activeKind === 'cut'
  const spotE1 = severed ? SPOT_DIM : SPOT_FULL // top block: hero + 01 + 02
  const spotE2 = severed ? SPOT_FULL : SPOT_DIM // bottom block: 00 + 03 + 04 + 05

  return (
    <div style={{ minHeight: '100vh', background: '#f5f1e8', display: 'flex', justifyContent: 'center', padding: '28px 20px', overflowX: 'auto' }}>
      <div style={{ width: 1280, flex: 'none', background: '#f5f1e8' }}>
        <div style={{ padding: '26px 30px 30px', color: '#211d16', fontFamily: "'Barlow Condensed',system-ui,sans-serif" }}>
          <Header />

          <AlarmBoard chips={activeBeat?.alarm_chips ?? []} />

          <GapHero
            beat={activeBeat}
            activeKind={activeKind}
            severed={severed}
            playing={playing}
            loading={loading}
            onSelectBeat={selectBeat}
            onTogglePlay={togglePlay}
          />

          <TryYourOwn onAnalyze={runAnalyze} loading={loading} mock={isMock} />

          {error && (
            <div
              style={{
                marginTop: 8,
                font: "600 11px 'JetBrains Mono',monospace",
                color: '#bf3b21',
                background: 'rgba(191,59,33,.08)',
                border: '1px solid rgba(191,59,33,.3)',
                borderRadius: 6,
                padding: '8px 12px',
              }}
            >
              {error}
            </div>
          )}

          <PromptOutput beat={activeBeat} mock={isMock} />

          <Panel00 beat={activeBeat} severed={severed} loading={loading} onToggleCut={toggleCut} spotStyle={spotE2} />

          <Panel01 rows={activeBeat?.token_rows ?? []} spotStyle={spotE1} />

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, marginTop: 24 }}>
            <div>
              <Panel02 features={activeBeat?.top_features ?? []} spotStyle={spotE1} />
              <div style={{ marginTop: 20 }}>
                <Panel03
                  harmfulBeat={cache.harmful}
                  cutBeat={cache.cut}
                  onRun={runCutTestFromPanel03}
                  loading={loading}
                  spotStyle={spotE2}
                />
              </div>
            </div>
            <div style={{ ...spotE2, transition: 'opacity .45s ease,filter .45s ease' }}>
              <Panel04 expA={expA} expB={expB} onRunA={() => toggleExperiment('a')} onRunB={() => toggleExperiment('b')} loading={loading} />
              <Panel05 layers={layerSweep} />
            </div>
          </div>

          <Footer />
        </div>
      </div>
    </div>
  )
}
