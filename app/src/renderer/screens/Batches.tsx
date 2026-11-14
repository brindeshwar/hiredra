// Batches: groups of 10 queued jobs to review together. The batch endpoints arrive with the application filler
// (piece 5); until then this screen says so instead of failing.

import { Busy, ErrorLine, fmtWhen, useLoad } from '../ui'
import type { ScreenProps } from './index'

type Batch = { batch_id: string; created_at: string; status: string; items?: { job_id: string; position: number; decision: string | null }[] }

export function BatchesScreen({ api, go }: ScreenProps) {
  const batches = useLoad(() => api.raw.get<Batch[]>('/batches'), [])
  const missing = batches.error?.includes('Not Found')
  return (
    <section>
      <header className="screen-head">
        <h1>Batches</h1>
        {!missing && (
          <Busy kind="primary" onClick={async () => { await api.raw.post('/batches', { min: 1 }); batches.reload() }}>
            Make batch
          </Busy>
        )}
      </header>
      {missing ? (
        <p className="note">Batches come with the application filler. Tailored jobs are on the Packets screen.</p>
      ) : (
        <ErrorLine error={batches.error} />
      )}
      {(batches.data ?? []).map((b) => (
        <div key={b.batch_id} className="batch">
          <h2>
            {b.batch_id} <span className={`pill ${b.status}`}>{b.status}</span> <small>{fmtWhen(b.created_at)}</small>
          </h2>
          <ol>
            {(b.items ?? []).map((i) => (
              <li key={i.job_id}>
                <button className="link" onClick={() => go('apply', i.job_id)}>
                  {i.job_id}
                </button>{' '}
                {i.decision ? <span className="pill">{i.decision}</span> : null}
              </li>
            ))}
          </ol>
        </div>
      ))}
      {batches.data && batches.data.length === 0 && <p className="muted">No batches yet.</p>}
    </section>
  )
}
