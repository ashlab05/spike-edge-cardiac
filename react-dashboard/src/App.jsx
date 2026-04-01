import { useState } from 'react'
import './index.css'
import data from './data/results.json'
import Overview from './pages/Overview'
import Tradeoff from './pages/Tradeoff'
import EdgeDeploy from './pages/EdgeDeploy'
import Architecture from './pages/Architecture'
import { Zap, BarChart3, Trophy, Cpu, Brain } from 'lucide-react'

const PAGES = [
  { id: 'overview', label: 'Overview', icon: BarChart3 },
  { id: 'tradeoff', label: 'Why SNN Wins', icon: Trophy },
  { id: 'edge', label: 'Edge Deployment', icon: Cpu },
  { id: 'arch', label: 'SNN Architecture', icon: Brain },
]

export default function App() {
  const [page, setPage] = useState('overview')

  const renderPage = () => {
    switch (page) {
      case 'overview': return <Overview data={data} />
      case 'tradeoff': return <Tradeoff data={data} />
      case 'edge': return <EdgeDeploy data={data} />
      case 'arch': return <Architecture data={data} />
      default: return <Overview data={data} />
    }
  }

  return (
    <div className="app">
      <aside className="sidebar">
        <div className="sidebar-logo">
          <Zap size={28} color="#f39c12" style={{ filter: 'drop-shadow(0 0 8px rgba(243,156,18,0.5))' }} />
          <div>
            <h1>Spike-Edge</h1>
            <span className="subtitle">Cardiac Anomaly Detection</span>
          </div>
        </div>

        <nav className="sidebar-nav">
          {PAGES.map(p => (
            <button
              key={p.id}
              className={`nav-item ${page === p.id ? 'active' : ''}`}
              onClick={() => setPage(p.id)}
            >
              <span className="nav-icon"><p.icon size={18} /></span>
              {p.label}
            </button>
          ))}
        </nav>

        <div className="sidebar-footer">
          <div className="badge">Edge Ready</div>
        </div>
      </aside>

      <main className="main-content">
        {renderPage()}
      </main>
    </div>
  )
}
