import { useState } from 'react'
import './index.css'
import LiveRestorePage from './pages/LiveRestorePage'
import BenchmarkExplorerPage from './pages/BenchmarkExplorerPage'

const TABS = [
  { id: 'restore',   label: 'Live Restore' },
  { id: 'benchmark', label: 'Benchmark Explorer' },
]

function App() {
  const [activeTab, setActiveTab] = useState('restore')
  return (
    <div className="app">
      <header className="header">
        <div className="container header-content">
          <div className="logo">
            <div className="logo-icon">🎨</div>
            <span>Artifex</span>
          </div>
          <nav style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
            {TABS.map((tab) => (
              <button key={tab.id} onClick={() => setActiveTab(tab.id)}
                style={{
                  background: activeTab === tab.id ? 'rgba(212,175,55,0.15)' : 'transparent',
                  border: activeTab === tab.id ? '1px solid rgba(212,175,55,0.4)' : '1px solid rgba(255,255,255,0.1)',
                  borderRadius: 'var(--radius-md)', padding: '8px 16px',
                  color: activeTab === tab.id ? 'var(--color-gold)' : 'var(--color-gray-300)',
                  cursor: 'pointer', fontSize: '0.875rem',
                  fontWeight: activeTab === tab.id ? '600' : '400',
                }}>
                {tab.id === 'restore' ? '✨ ' : '📊 '}{tab.label}
              </button>
            ))}
          </nav>
        </div>
      </header>
      <main>
        {activeTab === 'restore'   && <LiveRestorePage />}
        {activeTab === 'benchmark' && <BenchmarkExplorerPage />}
      </main>
      <footer className="footer">
        <div className="container">
          <p className="footer-text">Artifex · Brushstroke-Aware Van Gogh Restoration · IPD Thesis 2026</p>
        </div>
      </footer>
    </div>
  )
}

export default App
