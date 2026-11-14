// Packets: the tailored jobs, each with its resume and cover letter to open, and a button to build more.

import { useState } from 'react'
import { Busy, ErrorLine, fmtScore, useLoad } from '../ui'
import type { ScreenProps } from './index'

const FILES: [string, string][] = [
  ['cv.pdf', 'Resume (PDF)'],
  ['cv.docx', 'Resume (Word)'],
  ['cover.pdf', 'Cover letter (PDF)'],
  ['cover.docx', 'Cover letter (Word)']
]

export function PacketsScreen({ api, arg }: ScreenProps) {
  const list = useLoad(() => api.jobs('tailored', 500), [])
  const [picked, setPicked] = useState<string | null>(arg ?? null)
  const [top, setTop] = useState(5)
  const [message, setMessage] = useState<string | null>(null)
  return (
    <section className="split">
      <div className="split-main">
        <header className="screen-head">
          <h1>Packets</h1>
          <span>
            <input className="num" type="number" min={1} max={50} value={top}
              onChange={(e) => setTop(Math.max(1, Math.min(50, Number(e.target.value) || 1)))} />{' '}
            <Busy kind="primary" onClick={async () => {
              const run = await api.startRun('packets', top)
              setMessage(`Building packets for the best ${top} scored jobs (run ${run.id}); follow it on the Pipeline screen.`)
            }}>
              Build packets
            </Busy>
          </span>
        </header>
        {message && <p className="note">{message}</p>}
        <ErrorLine error={list.error} />
        <table className="table hover">
          <thead>
            <tr>
              <th>Title</th>
              <th>Company</th>
              <th className="right">Score</th>
            </tr>
          </thead>
          <tbody>
            {(list.data?.items ?? []).map((j) => (
              <tr key={j.job_id} className={j.job_id === picked ? 'picked' : ''} onClick={() => setPicked(j.job_id)}>
                <td>{j.title}</td>
                <td>{j.company}</td>
                <td className="right mono">{fmtScore(j.score)}</td>
              </tr>
            ))}
          </tbody>
        </table>
        {list.data && list.data.items.length === 0 && <p className="muted">No packets yet: build some above.</p>}
      </div>
      <aside className="split-side">
        {picked ? <PacketPanel key={picked} api={api} jobKey={picked} /> : <p className="muted">Pick a packet.</p>}
      </aside>
    </section>
  )
}

function PacketPanel({ api, jobKey }: { api: ScreenProps['api']; jobKey: string }) {
  const packet = useLoad(() => api.packet(jobKey), [jobKey])
  const p = packet.data
  if (packet.error) return <ErrorLine error={packet.error} />
  if (!p) return <p className="muted">Loading…</p>
  return (
    <div className="detail">
      <h2>Packet</h2>
      <p className="mono small">{p.folder}</p>
      {p.flags.length > 0 && (
        <p>
          {p.flags.map((f) => (
            <span key={f} className="pill warn" title={FLAG_HELP[f.replace('cover:', '')] ?? ''}>
              {f}
            </span>
          ))}
        </p>
      )}
      <div className="file-buttons">
        {FILES.filter(([name]) => p.files.includes(name)).map(([name, label]) => (
          <Busy key={name} onClick={() => api.openPacketFile(jobKey, name)}>
            {label}
          </Busy>
        ))}
      </div>
      <p className="muted small">
        Files open in your own apps. Read every line before you use them: nothing is ever sent or submitted for you.
      </p>
    </div>
  )
}

const FLAG_HELP: Record<string, string> = {
  critic_unresolved: 'Some generated text broke the claim rules twice and was replaced by your profile’s own words',
  page_overflow: 'The resume is still longer than one page',
  no_pdf: 'No PDF converter was found; only the Word file was made',
  model_gave_nothing: 'The model gave no usable answer; the packet uses your profile’s own words'
}
