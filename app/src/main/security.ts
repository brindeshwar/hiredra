// The app's security rules, kept free of Electron imports so they can be tested on their own.
// Electron's own checklist, all on from day one: isolated, sandboxed renderers with no Node access, a strict
// content policy, no permissions, no navigation away from the app, links out only through a checked handler,
// files opened only from inside the packet folder, and requests to the sidecar only on a short list of routes.

import { resolve, sep } from 'node:path'

/** The options every app window is created with. A test reads these and checks each flag. */
export function windowOptions(preloadPath: string) {
  return {
    width: 1360,
    height: 860,
    minWidth: 1000,
    minHeight: 640,
    show: false,
    title: 'Hiredra',
    autoHideMenuBar: true,
    webPreferences: {
      preload: preloadPath,
      contextIsolation: true,
      sandbox: true,
      nodeIntegration: false,
      nodeIntegrationInWorker: false,
      nodeIntegrationInSubFrames: false,
      webSecurity: true,
      allowRunningInsecureContent: false,
      webviewTag: false,
      spellcheck: false
    }
  } as const
}

/** The content policy for the built app: its own scripts only; it never talks to the network directly. */
export const CONTENT_SECURITY_POLICY = [
  "default-src 'self'",
  "script-src 'self'",
  "style-src 'self' 'unsafe-inline'",
  "img-src 'self' data:",
  "font-src 'self'",
  "connect-src 'self'",
  "object-src 'none'",
  "base-uri 'none'",
  "form-action 'none'",
  "frame-ancestors 'none'"
].join('; ')

/** The policy while developing (`npm run dev` only): React's live reload needs one inline script and a websocket to
 * the local dev server. The built app never uses this. */
export const DEV_CONTENT_SECURITY_POLICY = CONTENT_SECURITY_POLICY.replace(
  "script-src 'self'",
  "script-src 'self' 'unsafe-inline'"
).replace("connect-src 'self'", "connect-src 'self' ws://localhost:*")

/** The only permission a page may have: writing to the clipboard, for the Copy buttons. */
export function isAllowedPermission(permission: string): boolean {
  return permission === 'clipboard-sanitized-write'
}

/** A link the app may hand to the system browser: plain http(s) only. */
export function isSafeExternal(url: string): boolean {
  try {
    const u = new URL(url)
    return (u.protocol === 'https:' || u.protocol === 'http:') && u.username === '' && u.password === ''
  } catch {
    return false
  }
}

/** The files a packet may contain; nothing else is ever opened. */
export const PACKET_FILES = new Set([
  'cv.docx',
  'cv.pdf',
  'cover.docx',
  'cover.pdf',
  'payload.json',
  'claims.json',
  'form_scan.json',
  'answers.json'
])

/** The packet folder name for a job key: the same rule as hiredra/resume/packet.py's folder_name. */
export function packetFolder(jobKey: string): string {
  return jobKey.replace(/[^A-Za-z0-9._-]/g, '_').slice(0, 80)
}

/** The absolute path of a packet file, or null when the name or the result would leave the packets folder. */
export function packetFilePath(packetsRoot: string, jobKey: string, name: string): string | null {
  if (!PACKET_FILES.has(name)) return null
  const root = resolve(packetsRoot)
  const full = resolve(root, packetFolder(jobKey), name)
  return full.startsWith(root + sep) ? full : null
}

/** The sidecar routes the renderer may call, by method. Anything else is refused in the main process. */
const ROUTES: Record<string, RegExp[]> = {
  GET: [/^\/status$/, /^\/jobs(\?.*)?$/, /^\/jobs\/.+$/, /^\/runs(\/[a-f0-9]+)?$/, /^\/packets\/.+$/, /^\/batches(\/.+)?$/, /^\/apply\/.+$/],
  POST: [/^\/jobs\/.+\/(release|reject)$/, /^\/runs\/[a-z]+$/, /^\/batches$/, /^\/apply\/.+\/(scan|decision)$/]
}

export function isAllowedCall(method: string, path: string): boolean {
  if (!path.startsWith('/') || path.includes('..') || path.includes('\\')) return false
  return (ROUTES[method] ?? []).some((re) => re.test(path))
}
