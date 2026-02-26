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
  (window as any).ipcRenderer = {
    on: (_channel: string, _listener: (...args: any[]) => void) => {
      console.log(`[MockIPC] Register listener for: ${_channel}`);
      return () => {};
    },
    removeListener: (_channel: string, _listener: (...args: any[]) => void) => {},
    send: (_channel: string, ..._args: any[]) => {
      console.log(`[MockIPC] send (ignored): ${_channel}`, ..._args);
    },
    invoke: async (_channel: string, ..._args: any[]) => {
      console.log(`[MockIPC] invoke (returning null): ${_channel}`, ..._args);
      if (_channel === 'storage-get-app-data-path') return '/tmp/moyin-web-preview';
      if (_channel.startsWith('storage-')) return { success: true, data: null };
      return null;
    },
    sendSync: (_channel: string, ..._args: any[]) => {
      console.log(`[MockIPC] sendSync: ${_channel}`);
      return null;
    }
  };
}

window.ipcRenderer.on('main-process-message', (_event, message) => {
  console.log(message)
})
