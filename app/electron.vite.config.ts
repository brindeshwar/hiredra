import { resolve } from 'node:path'
import { defineConfig } from 'electron-vite'
import react from '@vitejs/plugin-react'
import type { Plugin } from 'vite'
import { CONTENT_SECURITY_POLICY } from './src/main/security'

// The built page carries the strict content policy in its HTML (a file:// page gets no response headers). In
// development the main process sets a looser one instead, because React's live reload needs an inline script.
const strictPolicyInBuild: Plugin = {
  name: 'hiredra-csp',
  apply: 'build',
  transformIndexHtml: (html) =>
    html.replace('<head>', `<head>\n    <meta http-equiv="Content-Security-Policy" content="${CONTENT_SECURITY_POLICY}" />`)
}

// Three builds: the main process (window, sidecar, IPC), the preload bridge, and the React renderer.
export default defineConfig({
  main: {
    build: { rollupOptions: { input: { index: resolve(__dirname, 'src/main/index.ts') } } }
  },
  preload: {
    build: {
      rollupOptions: {
        input: { index: resolve(__dirname, 'src/preload/index.ts') },
        // A sandboxed preload must be a CommonJS script, not an ES module.
        output: { format: 'cjs', entryFileNames: '[name].cjs' }
      }
    }
  },
  renderer: {
    root: resolve(__dirname, 'src/renderer'),
    build: { rollupOptions: { input: { index: resolve(__dirname, 'src/renderer/index.html') } } },
    plugins: [react(), strictPolicyInBuild]
  }
})
