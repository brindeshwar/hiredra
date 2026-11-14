// Jobs: the lists (ranked, review, tailored, queued, applied, all) and one job's detail with the triage buttons.

import { useState } from 'react'
import { VIEWS, type JobDetail, type View } from '../api'
import { Busy, ErrorLine, fmtScore, useLoad } from '../ui'
import type { ScreenProps } from './index'

const LABEL: Record<View, string> = {
  ranked: 'Ranked',
  review: 'Review',
  tailored: 'Tailored',
  queued: 'Queued',
  applied: 'Applied',
  all: 'All'
}

export function JobsScreen({ api, go }: ScreenProps) {
  const [view, setView] = useState<View>('ranked')
  const [picked, setPicked] = useState<string | null>(null)
  const list = useLoad(() => api.jobs(view, 300), [view])
  return (
    <section className="split">
      <div className="split-main">
        <header className="screen-head">
          <h1>Jobs</h1>
          <div className="tabs">
            {VIEWS.map((v) => (
              <button key={v} className={v === view ? 'tab active' : 'tab'} onClick={() => setView(v)}>
                {LABEL[v]}
              </button>
            ))}
          </div>
        </header>
        <ErrorLine error={list.error} />
        <p className="muted">
          {list.data ? `${list.data.total.toLocaleString()} jobs` : 'Loading…'}
          {list.data && list.data.total > list.data.items.length ? ` (showing the best ${list.data.items.length})` : ''}
          {view === 'ranked' ? ' · order is provisional until the weights are calibrated' : ''}
        </p>
        <table className="table hover">
          <thead>
            <tr>
              <th>Title</th>
              <th>Company</th>
              <th className="right">Score</th>
              <th>Status</th>
            </tr>
          </thead>
          <tbody>
            {(list.data?.items ?? []).map((j) => (
              <tr key={j.job_id} className={j.job_id === picked ? 'picked' : ''} onClick={() => setPicked(j.job_id)}>
                <td>{j.title}</td>
                <td>{j.company}</td>
                <td className="right mono">{fmtScore(j.score)}</td>
                <td>
                  {j.status}
                  {j.review_reason?.length ? <span className="pill warn">{j.review_reason.join(', ')}</span> : null}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <aside className="split-side">
        {picked ? (
          <JobDetailPanel key={picked} api={api} jobKey={picked} onChanged={list.reload} go={go} />
        ) : (
          <p className="muted">Pick a job to see its details.</p>
        )}
      </aside>
    </section>
  )
}

function JobDetailPanel({ api, jobKey, onChanged, go }: {
  api: ScreenProps['api']
  jobKey: string
  onChanged: () => void
  go: ScreenProps['go']
}) {
  const job = useLoad(() => api.job(jobKey), [jobKey])
  const j = job.data
  if (job.error) return <ErrorLine error={job.error} />
  if (!j) return <p className="muted">Loading…</p>
  const bands = groupBands(j)
  return (
    <div className="detail">
      <h2>{j.title}</h2>
      <p>
        <strong>{j.company}</strong> · {j.location ?? 'location not given'} · score {fmtScore(j.score)} · {j.status}
      </p>
      <div className="actions">
        {(j.apply_url || j.url) && <button onClick={() => api.openExternal((j.apply_url || j.url)!)}>Open job page</button>}
        {j.status === 'scored' && (
          <>
            <Busy onClick={async () => { await api.release(jobKey); onChanged(); job.reload() }}>Release</Busy>
            <Busy kind="danger" onClick={async () => { await api.reject(jobKey); onChanged(); job.reload() }}>Reject</Busy>
          </>
        )}
        {j.status === 'tailored' && <button onClick={() => go('packets', jobKey)}>Open packet</button>}
      </div>
      {bands.length > 0 && (
        <>
          <h3>Skills</h3>
          {bands.map(([band, items]) => (
            <p key={band}>
              <span className={`pill ${band}`}>{band}</span>{' '}
              {items.map((s) => (s.nearest && s.nearest !== s.skill ? `${s.skill} (via ${s.nearest})` : s.skill)).join(', ')}
            </p>
          ))}
        </>
      )}
      <h3>Description</h3>
      <div className="description">{j.description ?? 'No description stored.'}</div>
    </div>
  )
}

function groupBands(j: JobDetail): [string, NonNullable<JobDetail['score_features']>['skills'] & object][] {
  const rows = j.score_features?.skills ?? []
  const order = ['have', 'adjacent', 'missing']
  return order
    .map((b) => [b, rows.filter((r) => r.band === b)] as [string, typeof rows])
    .filter(([, items]) => items.length > 0)
}
