// The typed client for the sidecar API. Every screen goes through it; it only ever talks to the preload bridge.

export type JobRow = {
  job_id: string
  title: string | null
  company: string | null
  location: string | null
  score: number | null
  status: string
  review_reason: string[] | null
  released_at: string | null
  apply_url: string | null
  url: string | null
}

export type JobList = { view: string; total: number; items: JobRow[] }

export type SkillFeature = { skill: string; req: string; nearest: string | null; s: number; fit: number; band: string }

export type JobDetail = JobRow & {
  description: string | null
  skills: { name: string; requirement: string; source: string }[] | null
  score_features: { skills?: SkillFeature[]; coverage?: number; gap_distance?: number; domain_bucket?: string } | null
  domain_bucket: string | null
  legitimacy: string | null
  date_posted: string | null
}

export type Run = {
  id: string
  stage: string
  state: 'running' | 'ok' | 'failed'
  exit_code: number | null
  started_at: string
  finished_at: string | null
  log: string[]
}

export type Status = {
  jobs: Record<string, number>
  usage: { provider: string; model: string; calls: number; exhausted: boolean }[]
  latest_audit: string | null
  runs: Run[]
}

export type Packet = { job_id: string; folder: string; files: string[]; flags: string[]; artifacts: unknown[] }

export type Bridge = {
  call: (method: 'GET' | 'POST', path: string, body?: unknown) => Promise<unknown>
  sidecarStatus: () => Promise<{ state: string; detail: string }>
  onSidecarStatus: (listener: (s: { state: string; detail: string }) => void) => () => void
  openPacketFile: (jobKey: string, name: string) => Promise<void>
  openExternal: (url: string) => Promise<void>
}

export const STAGES = ['filter', 'extraction', 'sync', 'describe', 'scoring', 'audit', 'packets', 'forms'] as const
export type Stage = (typeof STAGES)[number]
export const VIEWS = ['ranked', 'review', 'tailored', 'queued', 'applied', 'all'] as const
export type View = (typeof VIEWS)[number]

/** A job key goes into the path as it is ("workday:acme:/job/x" keeps its slashes); only "?" and "#" are escaped. */
export function keyPath(jobKey: string): string {
  return jobKey.replace(/[?#%]/g, encodeURIComponent)
}

export function makeApi(bridge: Bridge) {
  const get = <T>(path: string) => bridge.call('GET', path) as Promise<T>
  const post = <T>(path: string, body?: unknown) => bridge.call('POST', path, body) as Promise<T>
  return {
    status: () => get<Status>('/status'),
    jobs: (view: View, limit = 200, offset = 0) => get<JobList>(`/jobs?view=${view}&limit=${limit}&offset=${offset}`),
    job: (key: string) => get<JobDetail>(`/jobs/${keyPath(key)}`),
    events: (key: string) => get<{ kind: string; at: string; detail: string | null }[]>(`/jobs/${keyPath(key)}/events`),
    release: (key: string) => post<{ ok: boolean }>(`/jobs/${keyPath(key)}/release`),
    reject: (key: string, reason = 'rejected') => post<{ ok: boolean }>(`/jobs/${keyPath(key)}/reject`, { reason }),
    runs: () => get<Run[]>('/runs'),
    run: (id: string) => get<Run>(`/runs/${id}`),
    startRun: (stage: Stage, top?: number) => post<Run>(`/runs/${stage}`, top === undefined ? undefined : { top }),
    packet: (key: string) => get<Packet>(`/packets/${keyPath(key)}`),
    openPacketFile: (key: string, name: string) => bridge.openPacketFile(key, name),
    openExternal: (url: string) => bridge.openExternal(url),
    raw: { get, post }
  }
}

export type Api = ReturnType<typeof makeApi>

declare global {
  interface Window {
    hiredra: Bridge
  }
}
