import {
    BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, RadarChart,
    PolarGrid, PolarAngleAxis, Radar, Legend, Cell
} from 'recharts'
import { Activity, ShieldCheck, Zap, Timer, HardDrive, Heart } from 'lucide-react'

const CustomTooltip = ({ active, payload, label }) => {
    if (!active || !payload?.length) return null
    return (
        <div style={{
            background: '#1e293b', border: '1px solid rgba(255,255,255,0.1)',
            borderRadius: 10, padding: '12px 16px', fontSize: 13
        }}>
            <p style={{ fontWeight: 600, marginBottom: 6 }}>{label}</p>
            {payload.map((p, i) => (
                <p key={i} style={{ color: p.color }}>
                    {p.name}: {typeof p.value === 'number' ? p.value.toFixed(3) : p.value}
                </p>
            ))}
        </div>
    )
}

const CLASS_COLORS = ['#2ecc71', '#e74c3c', '#e67e22', '#9b59b6']

export default function Overview({ data }) {
    const snn = data.models.find(m => m.name === 'SNN (LIF)')
    const models = data.models
    const classNames = data.class_names || ['Normal', 'Arrhythmia', 'Hypotensive', 'Hypertensive']

    const perfData = models.map(m => ({
        name: m.name.replace(' (TinyML)', '').replace(' Regression', ''),
        'F1 Macro': m.f1_macro,
        color: m.color,
        isSNN: m.category === 'snn'
    })).sort((a, b) => b['F1 Macro'] - a['F1 Macro'])

    const radarModels = ['SNN (LIF)', 'MLP (TinyML)', 'XGBoost', 'Random Forest', 'Threshold Baseline']
    const radarData = ['F1-Macro', 'Precision', 'Recall', 'Mem Eff', 'Energy Eff', 'Latency'].map((metric, i) => {
        const entry = { metric }
        radarModels.forEach(name => {
            const m = models.find(x => x.name === name)
            if (!m) return
            if (i === 0) entry[name] = m.f1_macro
            else if (i === 1) entry[name] = m.precision_macro
            else if (i === 2) entry[name] = m.recall_macro
            else if (i === 3) entry[name] = Math.max(0, 1 - Math.log10(Math.max(m.deploy.ram_kb, 0.1)) / Math.log10(300))
            else if (i === 4) entry[name] = Math.max(0, 1 - Math.log10(Math.max(m.deploy.uj, 0.1)) / Math.log10(400))
            else entry[name] = Math.max(0, 1 - Math.log10(Math.max(m.deploy.us, 1)) / Math.log10(3000))
        })
        return entry
    })

    const statCards = [
        { label: 'SNN F1-Macro', value: snn.f1_macro.toFixed(3), cls: 'accent', unit: '4-class cardiac event classification', icon: Activity, highlight: true },
        { label: 'SNN Accuracy', value: (snn.accuracy * 100).toFixed(1) + '%', cls: 'accent', unit: 'Distinguishes Normal, Arrhythmia, Hypotensive, Hypertensive', icon: Heart, highlight: true },
        { label: 'SNN Energy', value: snn.deploy.uj, cls: 'green', unit: 'µJ per inference — 225× less than RF', icon: Zap, highlight: false },
        { label: 'SNN Latency', value: snn.deploy.us, cls: 'cyan', unit: 'µs deterministic — real-time capable', icon: Timer, highlight: false },
        { label: 'Model Size', value: snn.deploy.size_kb, cls: 'blue', unit: 'KB — fits anywhere, 375× smaller than RF', icon: HardDrive, highlight: false },
    ]

    // Per-class F1 data for SNN
    const perClassData = classNames.map((name, i) => ({
        name,
        f1: snn.per_class_f1[i],
        color: CLASS_COLORS[i],
    }))

    return (
        <div>
            <div className="page-header">
                <h2>Cardiac Event Classification — Model Performance</h2>
                <p>4-class classification on {data.dataset.total.toLocaleString()} readings — Normal, Arrhythmia, Hypotensive, Hypertensive</p>
            </div>

            <div className="stats-grid">
                {statCards.map((s, i) => (
                    <div className={`stat-card ${s.highlight ? 'highlight' : ''}`} key={i}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                            <div className="stat-label">{s.label}</div>
                            <s.icon size={16} style={{ color: 'var(--text-muted)', opacity: 0.5 }} />
                        </div>
                        <div className={`stat-value ${s.cls}`}>{s.value}</div>
                        <div className="stat-unit">{s.unit}</div>
                    </div>
                ))}
            </div>

            {/* SNN Per-Class F1 */}
            <div className="card" style={{ marginBottom: 24 }}>
                <div className="card-title">SNN Per-Class Detection F1-Score</div>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr 1fr', gap: 12 }}>
                    {perClassData.map((c, i) => (
                        <div key={i} style={{
                            padding: 16, borderRadius: 10, textAlign: 'center',
                            background: `${c.color}10`, border: `1px solid ${c.color}30`
                        }}>
                            <div style={{ fontSize: 12, color: '#94a3b8', marginBottom: 6 }}>{c.name}</div>
                            <div style={{ fontSize: 28, fontWeight: 700, color: c.color }}>{c.f1.toFixed(3)}</div>
                            <div style={{
                                marginTop: 8, height: 4, borderRadius: 2, background: 'rgba(255,255,255,0.05)',
                                overflow: 'hidden'
                            }}>
                                <div style={{
                                    width: `${c.f1 * 100}%`, height: '100%', borderRadius: 2,
                                    background: c.color
                                }} />
                            </div>
                        </div>
                    ))}
                </div>
            </div>

            <div className="charts-grid">
                <div className="chart-card">
                    <h3>F1-Macro Score (4-Class, Higher = Better)</h3>
                    <ResponsiveContainer width="100%" height={340}>
                        <BarChart data={perfData} layout="vertical" margin={{ left: 80, right: 30 }}>
                            <XAxis type="number" domain={[0, 1]} tick={{ fill: '#94a3b8', fontSize: 11 }}
                                axisLine={{ stroke: 'rgba(255,255,255,0.1)' }} />
                            <YAxis type="category" dataKey="name" tick={{ fill: '#94a3b8', fontSize: 12 }}
                                axisLine={false} tickLine={false} width={90} />
                            <Tooltip content={<CustomTooltip />} />
                            <Bar dataKey="F1 Macro" radius={[0, 6, 6, 0]} barSize={20}>
                                {perfData.map((entry, i) => (
                                    <Cell key={i} fill={entry.isSNN ? '#f39c12' : entry.color}
                                        fillOpacity={entry.isSNN ? 1 : 0.7} />
                                ))}
                            </Bar>
                        </BarChart>
                    </ResponsiveContainer>
                </div>

                <div className="chart-card">
                    <h3>Multi-Dimensional Comparison (SNN excels in efficiency)</h3>
                    <ResponsiveContainer width="100%" height={340}>
                        <RadarChart data={radarData}>
                            <PolarGrid stroke="rgba(255,255,255,0.1)" />
                            <PolarAngleAxis dataKey="metric" tick={{ fill: '#94a3b8', fontSize: 11 }} />
                            {radarModels.map(name => {
                                const m = models.find(x => x.name === name)
                                return (
                                    <Radar key={name} name={name} dataKey={name} stroke={m?.color || '#fff'}
                                        fill={m?.color || '#fff'} fillOpacity={name === 'SNN (LIF)' ? 0.3 : 0.08}
                                        strokeWidth={name === 'SNN (LIF)' ? 3 : 1.5} />
                                )
                            })}
                            <Legend wrapperStyle={{ fontSize: 11, color: '#94a3b8' }} />
                        </RadarChart>
                    </ResponsiveContainer>
                </div>
            </div>

            <div className="card" style={{ marginBottom: 24 }}>
                <div className="card-title">Complete Results Table (4-Class Classification)</div>
                <div style={{ overflowX: 'auto' }}>
                    <table className="data-table">
                        <thead>
                            <tr>
                                <th>Model</th>
                                <th>Accuracy</th>
                                <th>F1-Macro</th>
                                <th>Precision</th>
                                <th>Recall</th>
                                <th>Size</th>
                                <th>Energy</th>
                                <th>Latency</th>
                            </tr>
                        </thead>
                        <tbody>
                            {models.map(m => (
                                <tr key={m.name} className={m.category === 'snn' ? 'highlight' : ''}>
                                    <td>
                                        <span className="model-dot" style={{ background: m.color }} />
                                        {m.name}
                                    </td>
                                    <td>{m.accuracy.toFixed(3)}</td>
                                    <td style={{ fontWeight: 600 }}>{m.f1_macro.toFixed(3)}</td>
                                    <td>{m.precision_macro.toFixed(3)}</td>
                                    <td>{m.recall_macro.toFixed(3)}</td>
                                    <td>{m.deploy.size_kb} KB</td>
                                    <td>{m.deploy.uj} µJ</td>
                                    <td>{m.deploy.us} µs</td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    )
}
