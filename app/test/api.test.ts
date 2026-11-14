import { describe, expect, it } from 'vitest'
import { keyPath, makeApi, type Bridge } from '../src/renderer/api'

function fakeBridge() {
  const calls: [string, string, unknown][] = []
  const bridge: Bridge = {
    call: async (method, path, body) => {
      calls.push([method, path, body])
      return { ok: true }
    },
    sidecarStatus: async () => ({ state: 'ready', detail: '' }),
    onSidecarStatus: () => () => {},
    openPacketFile: async () => {},
    openExternal: async () => {}
  }
  return { bridge, calls }
}

describe('the API client', () => {
  it('builds each request', async () => {
    const { bridge, calls } = fakeBridge()
    const api = makeApi(bridge)
    await api.status()
    await api.jobs('review', 50, 100)
    await api.job('workday:acme:/job/Pune/AI')
    await api.release('greenhouse:1')
    await api.reject('greenhouse:2', 'not for me')
    await api.startRun('packets', 5)
    await api.startRun('scoring')
    await api.packet('ashby:abc')
    expect(calls).toEqual([
      ['GET', '/status', undefined],
      ['GET', '/jobs?view=review&limit=50&offset=100', undefined],
      ['GET', '/jobs/workday:acme:/job/Pune/AI', undefined],
      ['POST', '/jobs/greenhouse:1/release', undefined],
      ['POST', '/jobs/greenhouse:2/reject', { reason: 'not for me' }],
      ['POST', '/runs/packets', { top: 5 }],
      ['POST', '/runs/scoring', undefined],
      ['GET', '/packets/ashby:abc', undefined]
    ])
  })

  it('escapes only what would break a path', () => {
    expect(keyPath('greenhouse:1')).toBe('greenhouse:1')
    expect(keyPath('x?y#z%')).toBe('x%3Fy%23z%25')
  })
})
