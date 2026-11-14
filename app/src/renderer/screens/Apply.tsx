// Apply: your turn. For one queued (or tailored) job: the packet, every prepared answer with a Copy button, the
// questions only you can answer, the job page, and your decision. The app never submits anything: you do, in the
// job's own page, and then record what happened here.

import { useState } from 'react'
import { keyPath } from '../api'
import { Busy, ErrorLine, useLoad } from '../ui'
import type { ScreenProps } from './index'

type Field = { label: string; type: string; required: boolean; class: 'FACT' | 'FILE' | 'DRAFT' | 'PAUSE'; value?: string | null; category?: string }
type ApplyView = { job_id: string; via: string; fields: Field[] }

export function ApplyScreen({ api, arg }: ScreenProps) {
  const queued = useLoad(() => api.jobs('queued', 200), [])
  const tailored = useLoad(() => api.jobs('tailored', 200), [])
  const [picked, setPicked] = useState<string | null>(arg ?? null)
  const rows = [...(queued.data?.items ?? []), ...(tailored.data?.items ?? [])]
  return (
    <section className="split">
      <div className="split-main">
        <header className="screen-head">
          <h1>Apply</h1>
        </header>
        <p className="note">Hiredra prepares; you submit. Nothing here sends an application.</p>
        <ErrorLine error={queued.error ?? tailored.error} />
        <ul className="pick-list">
          {rows.map((j) => (
            <li key={j.job_id} className={j.job_id === picked ? 'picked' : ''} onClick={() => setPicked(j.job_id)}>
              <strong>{j.title}</strong> · {j.company} <span className="pill">{j.status}</span>
            </li>
          ))}
        </ul>
        {rows.length === 0 && <p className="muted">Nothing tailored or queued yet.</p>}
      </div>
      <aside className="split-side">
        {picked ? <ApplyPanel key={picked} api={api} jobKey={picked} /> : <p className="muted">Pick a job.</p>}
      </aside>
    </section>
  )
}

function ApplyPanel({ api, jobKey }: { api: ScreenProps['api']; jobKey: string }) {
  const job = useLoad(() => api.job(jobKey), [jobKey])
  const view = useLoad(() => api.raw.get<ApplyView>(`/apply/${keyPath(jobKey)}`), [jobKey])
  const [confirmation, setConfirmation] = useState('')
  const j = job.data
  if (!j) return <ErrorLine error={job.error} />
  const fields = view.data?.fields ?? []
  const copy = (text: string) => navigator.clipboard.writeText(text)
  return (
    <div className="detail">
      <h2>{j.title}</h2>
      <p>
        <strong>{j.company}</strong> · {j.status}
      </p>
      <div className="actions">
        {(j.apply_url || j.url) && (
          <button className="primary" onClick={() => api.openExternal((j.apply_url || j.url)!)}>
            Open job page
          </button>
        )}
        <Busy onClick={() => api.openPacketFile(jobKey, 'cv.pdf')}>Resume PDF</Busy>
        <Busy onClick={() => api.openPacketFile(jobKey, 'cover.pdf')}>Cover letter PDF</Busy>
      </div>
      {view.error ? (
        <p className="note">
          No form answers yet{view.error.includes('Not Found') ? ' (the form reader comes with the application filler)' : `: ${view.error}`}.
          Use the packet files and type the rest yourself.
        </p>
      ) : (
        <>
          <h3>Prepared answers</h3>
          {fields.filter((f) => f.class !== 'PAUSE').map((f) => (
            <div key={f.label} className="answer">
              <div className="answer-label">
                {f.label} <span className="pill">{f.class}</span>
              </div>
              <div className="answer-value">{f.value ?? '-'}</div>
              {f.value && <button onClick={() => copy(f.value!)}>Copy</button>}
            </div>
          ))}
          <h3>Only you can answer these</h3>
          <ul>
            {fields.filter((f) => f.class === 'PAUSE').map((f) => (
              <li key={f.label}>
                {f.label} {f.category ? <span className="pill warn">{f.category}</span> : null}
              </li>
            ))}
          </ul>
        </>
      )}
      <h3>Your decision</h3>
      <p className="muted small">Record what you did after you have acted in the job's own page.</p>
      <textarea
        placeholder="Paste the confirmation text the site showed after you submitted"
        value={confirmation}
        onChange={(e) => setConfirmation(e.target.value)}
      />
      <div className="actions">
        <Busy kind="primary" disabled={confirmation.trim().length < 5}
          onClick={() => api.raw.post(`/apply/${keyPath(jobKey)}/decision`, { decision: 'applied', confirmation })}>
          I applied
        </Busy>
        <Busy onClick={() => api.raw.post(`/apply/${keyPath(jobKey)}/decision`, { decision: 'not_yet' })}>Not yet</Busy>
        <Busy onClick={() => api.raw.post(`/apply/${keyPath(jobKey)}/decision`, { decision: 'deferred' })}>Later</Busy>
        <Busy kind="danger" onClick={() => api.raw.post(`/apply/${keyPath(jobKey)}/decision`, { decision: 'skipped' })}>Skip</Busy>
      </div>
    </div>
  )
}
