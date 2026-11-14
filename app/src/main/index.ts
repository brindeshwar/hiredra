// The main process: one window, the sidecar, and the only bridge between the two.

import { app, BrowserWindow, ipcMain, session, shell, type IpcMainInvokeEvent } from 'electron'
import { join, resolve } from 'node:path'
import { Sidecar } from './sidecar'
import {
  CONTENT_SECURITY_POLICY,
  isAllowedCall,
  isAllowedPermission,
  isSafeExternal,
  packetFilePath,
  windowOptions
} from './security'

// The app lives in <repo>/app; the pipeline and its data are one level up.
const repoRoot = resolve(app.getAppPath(), '..')
const packetsRoot = join(repoRoot, 'data', 'packets')
let win: BrowserWindow | null = null

const sidecar = new Sidecar(repoRoot, (s) => win?.webContents.send('hiredra:sidecar', s))

/** Only the app's own page may use the bridge. */
function fromApp(event: IpcMainInvokeEvent): boolean {
  const url = event.senderFrame?.url ?? ''
  const dev = process.env.ELECTRON_RENDERER_URL
  return (dev !== undefined && url.startsWith(dev)) || url.startsWith('file://')
}

function createWindow(): void {
  win = new BrowserWindow(windowOptions(join(__dirname, '../preload/index.cjs')))
  win.once('ready-to-show', () => win?.show())
  // Never navigate away from the app, never open a second window: links go to the system browser, checked.
  win.webContents.on('will-navigate', (e) => e.preventDefault())
  win.webContents.setWindowOpenHandler(({ url }) => {
    if (isSafeExternal(url)) void shell.openExternal(url)
    return { action: 'deny' }
  })
  if (process.env.ELECTRON_RENDERER_URL) void win.loadURL(process.env.ELECTRON_RENDERER_URL)
  else void win.loadFile(join(__dirname, '../renderer/index.html'))
}

app.whenReady().then(() => {
  // Deny every permission except writing to the clipboard (the Copy buttons on the Apply screen).
  session.defaultSession.setPermissionRequestHandler((_wc, permission, done) => done(isAllowedPermission(permission)))
  session.defaultSession.setPermissionCheckHandler((_wc, permission) => isAllowedPermission(permission))
  session.defaultSession.webRequest.onHeadersReceived((details, done) => {
    done({ responseHeaders: { ...details.responseHeaders, 'Content-Security-Policy': [CONTENT_SECURITY_POLICY] } })
  })

  ipcMain.handle('hiredra:call', async (event, method: string, path: string, body?: unknown) => {
    if (!fromApp(event)) throw new Error('refused: not the app')
    if (!isAllowedCall(method, path)) throw new Error(`refused: ${method} ${path}`)
    return sidecar.call(method, path, body)
  })
  ipcMain.handle('hiredra:sidecar-status', (event) => {
    if (!fromApp(event)) throw new Error('refused: not the app')
    return sidecar.status
  })
  ipcMain.handle('hiredra:open-packet-file', async (event, jobKey: string, name: string) => {
    if (!fromApp(event)) throw new Error('refused: not the app')
    const file = packetFilePath(packetsRoot, jobKey, name)
    if (!file) throw new Error('refused: not a packet file')
    const error = await shell.openPath(file)
    if (error) throw new Error(error)
  })
  ipcMain.handle('hiredra:open-external', async (event, url: string) => {
    if (!fromApp(event)) throw new Error('refused: not the app')
    if (!isSafeExternal(url)) throw new Error('refused: not an http(s) link')
    await shell.openExternal(url)
  })

  createWindow()
  sidecar.start().catch(() => {
    /* the window shows the failure from the status */
  })
  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow()
  })
})

app.on('web-contents-created', (_e, contents) => {
  contents.on('will-attach-webview', (e) => e.preventDefault())
})

app.on('window-all-closed', () => {
  sidecar.stop()
  if (process.platform !== 'darwin') app.quit()
})

app.on('before-quit', () => sidecar.stop())
