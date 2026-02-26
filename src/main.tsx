// Copyright (c) 2025 hotflow2024
// Licensed under AGPL-3.0-or-later. See LICENSE for details.
// Commercial licensing available. See COMMERCIAL_LICENSE.md.
import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App.tsx'
import './index.css'

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
)

// Use contextBridge or Mock for Web Preview
if (!window.ipcRenderer) {
  console.warn('[WebPreview] ipcRenderer is not available in browser. Mocking for preview...');
  (window as unknown as { ipcRenderer: unknown }).ipcRenderer = {
    on: (channel: string) => {
      console.log(`[MockIPC] Register listener for: ${channel}`);
      return () => {};
    },
    removeListener: () => {},
    send: (channel: string, ...args: unknown[]) => {
      console.log(`[MockIPC] send (ignored): ${channel}`, ...args);
    },
    invoke: async (channel: string, ...args: unknown[]) => {
      console.log(`[MockIPC] invoke (returning null): ${channel}`, ...args);
      if (channel === 'storage-get-app-data-path') return '/tmp/moyin-web-preview';
      if (channel.startsWith('storage-')) return { success: true, data: null };
      return null;
    },
    sendSync: (channel: string) => {
      console.log(`[MockIPC] sendSync: ${channel}`);
      return null;
    }
  };
}

window.ipcRenderer.on('main-process-message', (_event, message) => {
  console.log(message)
})
