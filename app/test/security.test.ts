import { describe, expect, it } from 'vitest'
import { join, resolve } from 'node:path'
import {
  CONTENT_SECURITY_POLICY,
  isAllowedCall,
  isAllowedPermission,
  isSafeExternal,
  packetFilePath,
  packetFolder,
  windowOptions
} from '../src/main/security'

describe('window options', () => {
  it('turns on every security flag', () => {
    const w = windowOptions('/x/preload.cjs').webPreferences
    expect(w.contextIsolation).toBe(true)
    expect(w.sandbox).toBe(true)
    expect(w.nodeIntegration).toBe(false)
    expect(w.nodeIntegrationInWorker).toBe(false)
    expect(w.nodeIntegrationInSubFrames).toBe(false)
    expect(w.webSecurity).toBe(true)
    expect(w.allowRunningInsecureContent).toBe(false)
    expect(w.webviewTag).toBe(false)
    expect(w.preload).toBe('/x/preload.cjs')
  })

  it('has a strict content policy', () => {
    expect(CONTENT_SECURITY_POLICY).toContain("script-src 'self'")
    expect(CONTENT_SECURITY_POLICY).toContain("object-src 'none'")
    expect(CONTENT_SECURITY_POLICY).not.toContain('unsafe-eval')
  })
})

describe('what the page may do', () => {
  it('only gets clipboard writing', () => {
    expect(isAllowedPermission('clipboard-sanitized-write')).toBe(true)
    for (const p of ['media', 'geolocation', 'notifications', 'clipboard-read', 'openExternal']) {
      expect(isAllowedPermission(p)).toBe(false)
    }
  })

  it('only opens plain http(s) links outside', () => {
    expect(isSafeExternal('https://boards.greenhouse.io/x/jobs/1')).toBe(true)
    expect(isSafeExternal('http://example.com')).toBe(true)
    for (const bad of ['file:///C:/x', 'javascript:alert(1)', 'mailto:a@b.c', 'https://u:p@x.com', 'not a url']) {
      expect(isSafeExternal(bad)).toBe(false)
    }
  })

  it('only calls the listed sidecar routes', () => {
    expect(isAllowedCall('GET', '/status')).toBe(true)
    expect(isAllowedCall('GET', '/jobs?view=ranked&limit=10&offset=0')).toBe(true)
    expect(isAllowedCall('GET', '/jobs/workday:acme:/job/x')).toBe(true)
    expect(isAllowedCall('POST', '/jobs/greenhouse:1/release')).toBe(true)
    expect(isAllowedCall('POST', '/runs/packets')).toBe(true)
    expect(isAllowedCall('POST', '/apply/greenhouse:1/decision')).toBe(true)
    expect(isAllowedCall('POST', '/jobs/greenhouse:1')).toBe(false)
    expect(isAllowedCall('DELETE', '/runs/abc')).toBe(false)
    expect(isAllowedCall('GET', '/packets/../../.env')).toBe(false)
    expect(isAllowedCall('GET', 'status')).toBe(false)
    expect(isAllowedCall('GET', '/health')).toBe(false)
  })
})

describe('packet files', () => {
  const root = resolve('/tmp/packets')
  it('names folders like the Python side', () => {
    expect(packetFolder('workday:acme:/job/Pune/AI_R-1')).toBe('workday_acme__job_Pune_AI_R-1')
    expect(packetFolder('x'.repeat(100))).toHaveLength(80)
  })
  it('opens only known files inside the packet folder', () => {
    expect(packetFilePath(root, 'greenhouse:1', 'cv.pdf')).toBe(join(root, 'greenhouse_1', 'cv.pdf'))
    expect(packetFilePath(root, 'greenhouse:1', '../../.env')).toBeNull()
    expect(packetFilePath(root, 'greenhouse:1', 'notes.txt')).toBeNull()
    expect(packetFilePath(root, '../..', 'cv.pdf')).toBe(join(root, '.._..', 'cv.pdf'))
  })
})
