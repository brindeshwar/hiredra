import { describe, expect, it } from 'vitest'
import { SCREENS } from '../src/renderer/screens'

describe('the screen registry', () => {
  it('has the five screens with unique ids, titles and components', () => {
    expect(SCREENS.map((s) => s.id)).toEqual(['pipeline', 'jobs', 'packets', 'batches', 'apply'])
    expect(new Set(SCREENS.map((s) => s.id)).size).toBe(SCREENS.length)
    for (const s of SCREENS) {
      expect(s.title.length).toBeGreaterThan(0)
      expect(typeof s.component).toBe('function')
    }
  })
})
