// The shell: a sidebar built from the screen registry, the sidecar's state, and the current screen.

import { useEffect, useMemo, useState } from 'react'
import { makeApi } from './api'
import { SCREENS } from './screens'

export function App() {
  const api = useMemo(() => makeApi(window.hiredra), [])
  const [screen, setScreen] = useState<{ id: string; arg?: string }>({ id: SCREENS[0].id })
  const [sidecar, setSidecar] = useState<{ state: string; detail: string }>({ state: 'starting', detail: '' })

  useEffect(() => {
    window.hiredra.sidecarStatus().then(setSidecar).catch(() => {})
    return window.hiredra.onSidecarStatus(setSidecar)
  }, [])

  const current = SCREENS.find((s) => s.id === screen.id) ?? SCREENS[0]
  const Screen = current.component
  const go = (id: string, arg?: string) => setScreen({ id, arg })

  return (
    <div className="app">
      <nav className="sidebar">
        <div className="brand">Hiredra</div>
        {SCREENS.map((s) => (
          <button key={s.id} className={s.id === current.id ? 'nav active' : 'nav'} onClick={() => go(s.id)} title={s.hint}>
            {s.title}
          </button>
        ))}
        <div className={`sidecar ${sidecar.state}`} title={sidecar.detail}>
          <span className="dot" /> service {sidecar.state}
        </div>
        <div className="rule">Prepares only. You submit.</div>
      </nav>
      <main className="content">
        {sidecar.state === 'ready' ? (
          <Screen key={`${current.id}:${screen.arg ?? ''}`} api={api} go={go} arg={screen.arg} />
        ) : (
          <div className="waiting">
            <h1>{sidecar.state === 'failed' ? 'The Python service did not start' : 'Starting the Python service…'}</h1>
            <p className="muted">{sidecar.detail}</p>
            {sidecar.state === 'failed' && (
              <p>
                Check that Docker (the database) is running and that <code>.venv</code> exists in the repo, then restart
                the app.
              </p>
            )}
          </div>
        )}
      </main>
    </div>
  )
}
