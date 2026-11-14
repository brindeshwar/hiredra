// The only bridge from the page to the main process: four calls, nothing else. No raw ipcRenderer is exposed.

import { contextBridge, ipcRenderer } from 'electron'

const bridge = {
  call: (method: 'GET' | 'POST', path: string, body?: unknown): Promise<unknown> =>
    ipcRenderer.invoke('hiredra:call', method, path, body),
  sidecarStatus: (): Promise<{ state: string; detail: string }> => ipcRenderer.invoke('hiredra:sidecar-status'),
  onSidecarStatus: (listener: (s: { state: string; detail: string }) => void): (() => void) => {
    const handler = (_e: unknown, s: { state: string; detail: string }) => listener(s)
    ipcRenderer.on('hiredra:sidecar', handler)
    return () => ipcRenderer.removeListener('hiredra:sidecar', handler)
  },
  openPacketFile: (jobKey: string, name: string): Promise<void> =>
    ipcRenderer.invoke('hiredra:open-packet-file', jobKey, name),
  openExternal: (url: string): Promise<void> => ipcRenderer.invoke('hiredra:open-external', url)
}

contextBridge.exposeInMainWorld('hiredra', bridge)

export type Bridge = typeof bridge
