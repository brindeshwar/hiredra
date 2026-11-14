// Pipeline: one row per stage with a Run button and its live log; the job counts and today's model use.

import { useEffect, useState } from 'react'
import { STAGES, type Run, type Stage } from '../api'
import { Busy, ErrorLine, fmtWhen, useLoad } from '../ui'
import type { ScreenProps } from './index'

const ABOUT: Record<Stage, string> = {
  filter: 'Location, seniority, role and liveness checks on fetched jobs (no model)',
  extraction: 'Skills from each job description (Gemini, then the NVIDIA fallback)',
  sync: 'Embed new skills into the graph (local model)',
  describe: 'Describe new skills and fold in other names for known ones',
  scoring: 'Score every extracted job against your profile (code)',
  audit: 'Graph health checks',
  packets: 'Tailored resume and cover letter for the best scored jobs'
}

const STATUS_ORDER = ['fetched', 'filtered', 'extracted', 'scored', 'tailored', 'queued', 'applied', 'outcome', 'dropped']

export function PipelineScreen({ api }: ScreenProps) {
  const status = useLoad(() => api.status(), [])
  const [runs, setRuns] = useState<Record<string, Run>>({})
  const [top, setTop] = useState(5)

  // Follow the runs that are still going.
  useEffect(() => {
    const live = Object.values(runs).filter((r) => r.state === 'running')
    if (live.length === 0) return
    const t = setTimeout(async () => {
      const next = { ...runs }
      for (const r of live) next[r.stage] = await api.run(r.id).catch(() => r)
      setRuns(next)
      if (Object.values(next).every((r) => r.state !== 'running')) status.reload()
    }, 2000)
    return () => clearTimeout(t)
  }, [runs, api, status])

  const start = async (stage: Stage) => {
    const run = await api.startRun(stage, stage === 'packets' ? top : undefined)
    setRuns((r) => ({ ...r, [stage]: run }))
  }

  const jobs = status.data?.jobs ?? {}
  return (
    <section>
      <header className="screen-head">
        <h1>Pipeline</h1>
        <button onClick={status.reload}>Refresh</button>
      </header>
      <ErrorLine error={status.error} />
      <div className="cards">
        {STATUS_ORDER.filter((s) => jobs[s] !== undefined).map((s) => (
          <div className="card" key={s}>
            <div className="card-num">{jobs[s].toLocaleString()}</div>
            <div className="card-label">{s}</div>
          </div>
        ))}
      </div>
      <h2>Stages</h2>
      <table className="table">
        <thead>
          <tr>
            <th>Stage</th>
            <th>What it does</th>
            <th>Last run here</th>
            <th />
          </tr>
        </thead>
        <tbody>
          {STAGES.map((stage) => {
            const run = runs[stage]
            return (
              <tr key={stage}>
                <td className="mono">{stage}</td>
                <td>{ABOUT[stage]}</td>
                <td>
                  {run ? (
                    <>
                      <span className={`pill ${run.state}`}>{run.state}</span> <small>{fmtWhen(run.started_at)}</small>
                      {run.log.length > 0 && <pre className="log">{run.log.slice(-6).join('\n')}</pre>}
                    </>
                  ) : (
                    <small>-</small>
                  )}
                </td>
                <td className="right">
                  {stage === 'packets' && (
                    <input
                      className="num"
                      type="number"
                      min={1}
                      max={50}
                      value={top}
                      onChange={(e) => setTop(Math.max(1, Math.min(50, Number(e.target.value) || 1)))}
                      title="How many jobs"
                    />
                  )}
                  <Busy kind="primary" onClick={() => start(stage)} disabled={run?.state === 'running'}>
                    Run
                  </Busy>
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
      <h2>Model use today</h2>
      <table className="table compact">
        <thead>
          <tr>
            <th>Provider</th>
            <th>Model</th>
            <th className="right">Requests</th>
          </tr>
        </thead>
        <tbody>
          {(status.data?.usage ?? []).map((u) => (
            <tr key={`${u.provider}/${u.model}`}>
              <td>{u.provider}</td>
              <td className="mono">{u.model}</td>
              <td className="right">
                {u.calls}
                {u.exhausted ? ' (daily quota reached)' : ''}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="muted">Latest graph audit: {status.data?.latest_audit ?? 'none yet'}</p>
    </section>
  )
}
