// Small shared pieces: async loading with an error line, a button that shows it is busy, and number formatting.

import { useCallback, useEffect, useState, type ReactNode } from 'react'

export function useLoad<T>(load: () => Promise<T>, deps: unknown[] = []) {
  const [data, setData] = useState<T | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)
  const reload = useCallback(() => {
    setLoading(true)
    load()
      .then((d) => {
        setData(d)
        setError(null)
      })
      .catch((e: unknown) => setError(e instanceof Error ? e.message : String(e)))
      .finally(() => setLoading(false))
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps)
  useEffect(reload, [reload])
  return { data, error, loading, reload }
}

export function ErrorLine({ error }: { error: string | null }) {
  return error ? <p className="error">{error}</p> : null
}

export function Busy({ onClick, children, kind = 'plain', disabled = false }: {
  onClick: () => Promise<unknown>
  children: ReactNode
  kind?: 'plain' | 'primary' | 'danger'
  disabled?: boolean
}) {
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  return (
    <span className="busy">
      <button
        className={kind}
        disabled={busy || disabled}
        onClick={() => {
          setBusy(true)
          setError(null)
          onClick()
            .catch((e: unknown) => setError(e instanceof Error ? e.message : String(e)))
            .finally(() => setBusy(false))
        }}
      >
        {busy ? 'Working…' : children}
      </button>
      {error && <span className="error inline">{error}</span>}
    </span>
  )
}

export const fmtScore = (s: number | null) => (s === null || s === undefined ? '-' : s.toFixed(2))
export const fmtWhen = (iso: string | null) => (iso ? new Date(iso).toLocaleString() : '-')
