// Starts and watches the Python sidecar (python -m hiredra.api). The main process owns it: it makes a fresh token
// for each launch, reads the one {"ready": true, "port": N} line the sidecar prints, waits for /health, restarts
// it once if it dies, and stops it when the app quits. The renderer never sees the token or the port.

import { spawn, type ChildProcess } from 'node:child_process'
import { randomBytes } from 'node:crypto'
import { existsSync } from 'node:fs'
import { join } from 'node:path'
import { createInterface } from 'node:readline'

export type SidecarState = { state: 'starting' | 'ready' | 'failed'; detail: string }

export function pythonPath(repoRoot: string): string {
  if (process.env.HIREDRA_PYTHON) return process.env.HIREDRA_PYTHON
  const win = join(repoRoot, '.venv', 'Scripts', 'python.exe')
  const nix = join(repoRoot, '.venv', 'bin', 'python')
  if (process.platform === 'win32' && existsSync(win)) return win
  if (existsSync(nix)) return nix
  return process.platform === 'win32' ? 'python' : 'python3'
}

export class Sidecar {
  readonly token = randomBytes(32).toString('hex')
  port = 0
  status: SidecarState = { state: 'starting', detail: 'starting the Python service' }
  private child: ChildProcess | null = null
  private restarts = 0
  private stopping = false
  private log: string[] = []

  constructor(private readonly repoRoot: string, private readonly onChange: (s: SidecarState) => void = () => {}) {}

  start(): Promise<void> {
    this.set({ state: 'starting', detail: 'starting the Python service' })
    const child = spawn(pythonPath(this.repoRoot), ['-m', 'hiredra.api', '--port', '0'], {
      cwd: this.repoRoot,
      env: { ...process.env, HIREDRA_API_TOKEN: this.token, PYTHONIOENCODING: 'utf-8', PYTHONUNBUFFERED: '1' },
      windowsHide: true
    })
    this.child = child
    child.stderr?.on('data', (d: Buffer) => this.keep(d.toString()))
    return new Promise((resolveReady, reject) => {
      const lines = createInterface({ input: child.stdout! })
      lines.once('line', async (line) => {
        try {
          this.port = JSON.parse(line).port
          await this.waitHealthy()
          this.restarts = 0
          this.set({ state: 'ready', detail: `listening on 127.0.0.1:${this.port}` })
          resolveReady()
        } catch (err) {
          this.set({ state: 'failed', detail: `the Python service did not start: ${String(err)}` })
          reject(err)
        }
      })
      child.once('exit', (code) => {
        if (this.stopping) return
        const detail = `the Python service stopped (exit ${code}). ${this.log.slice(-3).join(' ')}`.trim()
        if (this.restarts < 1) {
          this.restarts += 1
          this.set({ state: 'starting', detail: `${detail} Restarting once.` })
          this.start().catch(() => {})
        } else {
          this.set({ state: 'failed', detail })
        }
        reject(new Error(detail))
      })
    })
  }

  private async waitHealthy(): Promise<void> {
    for (let i = 0; i < 50; i += 1) {
      try {
        const r = await fetch(`http://127.0.0.1:${this.port}/health`)
        if (r.ok) return
      } catch {
        /* not listening yet */
      }
      await new Promise((r) => setTimeout(r, 200))
    }
    throw new Error('no answer from /health')
  }

  /** One request to the sidecar, with the token. Returns the JSON body, or throws with the server's message. */
  async call(method: string, path: string, body?: unknown): Promise<unknown> {
    if (this.status.state !== 'ready') throw new Error(this.status.detail)
    const r = await fetch(`http://127.0.0.1:${this.port}${path}`, {
      method,
      headers: { Authorization: `Bearer ${this.token}`, 'Content-Type': 'application/json' },
      body: body === undefined ? undefined : JSON.stringify(body)
    })
    const text = await r.text()
    const data = text ? JSON.parse(text) : null
    if (!r.ok) throw new Error((data && (data as { detail?: string }).detail) || `HTTP ${r.status}`)
    return data
  }

  stop(): void {
    this.stopping = true
    this.child?.kill()
  }

  private keep(text: string): void {
    this.log.push(...text.split(/\r?\n/).filter(Boolean))
    this.log = this.log.slice(-50)
  }

  private set(s: SidecarState): void {
    this.status = s
    this.onChange(s)
  }
}
