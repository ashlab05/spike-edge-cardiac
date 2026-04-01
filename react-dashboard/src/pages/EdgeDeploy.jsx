import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell, LabelList } from 'recharts'
import { HardDrive, MemoryStick, Timer, Zap, Lightbulb } from 'lucide-react'

export default function EdgeDeploy({ data }) {
    const models = data.models

    const metrics = [
        { key: 'size_kb', label: 'Model Size (KB)', unit: 'KB', title: 'Model Size — Smaller = Better' },
        { key: 'ram_kb', label: 'RAM Usage (KB)', unit: 'KB', title: 'RAM Usage — 512 KB available' },
        { key: 'us', label: 'Latency (µs)', unit: 'µs', title: 'Inference Latency — Lower = Better' },
        { key: 'uj', label: 'Energy (µJ)', unit: 'µJ', title: 'Energy per Inference — Lower = Better' },
    ]

    const feasibility = [
        { name: 'SNN (LIF)', deployable: 'Yes', reason: '0.8 KB model, 0.4 KB RAM, native C, 4-class classifier', color: '#f39c12', best: true },
        { name: 'MLP (TinyML)', deployable: 'Yes', reason: 'Small network, TFLite compatible', color: '#3498db', best: false },
        { name: 'Decision Tree', deployable: 'Yes', reason: 'Small, interpretable', color: '#e74c3c', best: false },
        { name: 'Logistic Regression', deployable: 'Yes', reason: 'Minimal footprint', color: '#34495e', best: false },
        { name: 'Random Forest', deployable: 'Marginal', reason: '150 trees = 300 KB model, high RAM', color: '#1abc9c', best: false },
        { name: 'XGBoost', deployable: 'Marginal', reason: 'Requires runtime library, 220 KB', color: '#e67e22', best: false },
        { name: 'LightGBM', deployable: 'Marginal', reason: 'Similar to XGBoost', color: '#2ecc71', best: false },
        { name: 'k-NN', deployable: 'No', reason: 'Stores full training set (240 KB+)', color: '#9b59b6', best: false },
    ]

    const snn = models.find(m => m.name === 'SNN (LIF)')
    const espSpecs = { flash_kb: 8192, sram_kb: 512 }
    const gauges = [
        { label: 'Flash Usage', used: snn.deploy.size_kb + 2, total: espSpecs.flash_kb, unit: 'KB' },
        { label: 'SRAM Usage', used: snn.deploy.ram_kb, total: espSpecs.sram_kb, unit: 'KB' },
    ]

    const statCards = [
        { label: 'SNN Model Size', value: `${snn.deploy.size_kb} KB`, unit: `of ${espSpecs.flash_kb.toLocaleString()} KB Flash (${(snn.deploy.size_kb / espSpecs.flash_kb * 100).toFixed(3)}%)`, Icon: HardDrive, highlight: true },
        { label: 'SNN RAM Usage', value: `${snn.deploy.ram_kb} KB`, unit: `of ${espSpecs.sram_kb} KB SRAM (${(snn.deploy.ram_kb / espSpecs.sram_kb * 100).toFixed(3)}%)`, Icon: MemoryStick, highlight: true },
        { label: 'Inference Latency', value: `${snn.deploy.us} µs`, unit: 'Deterministic — hard real-time', Icon: Timer, highlight: false },
        { label: 'Energy / Inference', value: `${snn.deploy.uj} µJ`, unit: `@ 10 Hz = ${(snn.deploy.uj * 10 * 3600 / 1e6).toFixed(4)} J/hour`, Icon: Zap, highlight: false },
    ]

    return (
        <div>
            <div className="page-header">
                <h2>Edge Deployment Analysis</h2>
                <p>Edge MCU (Xtensa LX7, 240 MHz, 8 MB Flash, 512 KB SRAM) — SNN leaves 99.9% resources free</p>
            </div>

            <div className="stats-grid">
                {statCards.map((s, i) => (
                    <div className={`stat-card ${s.highlight ? 'highlight' : ''}`} key={i}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                            <div className="stat-label">{s.label}</div>
                            <s.Icon size={16} style={{ color: 'var(--text-muted)', opacity: 0.5 }} />
                        </div>
                        <div className={`stat-value ${s.highlight ? 'accent' : 'green'}`}>{s.value}</div>
                        <div className="stat-unit">{s.unit}</div>
                    </div>
                ))}
            </div>

            <div className="card" style={{ marginBottom: 24 }}>
                <div className="card-title">Edge MCU Resource Utilization (SNN)</div>
                <div className="gauge-row">
                    {gauges.map((g, i) => (
                        <div className="gauge-item" key={i}>
                            <span className="gauge-label">{g.label}</span>
                            <div className="gauge-bar">
                                <div className="gauge-fill" style={{
                                    width: `${Math.max(1, (g.used / g.total) * 100)}%`,
                                    background: 'linear-gradient(90deg, #f39c12, #e67e22)'
                                }} />
                            </div>
                            <span className="gauge-value">
                                {g.used} / {g.total} {g.unit}
                            </span>
                        </div>
                    ))}
                    <div className="gauge-item">
                        <span className="gauge-label">CPU Usage</span>
                        <div className="gauge-bar">
                            <div className="gauge-fill" style={{ width: '0.3%', background: 'linear-gradient(90deg, #2ecc71, #1abc9c)' }} />
                        </div>
                        <span className="gauge-value">8 µs / 100,000 µs cycle</span>
                    </div>
                </div>
                <p style={{ marginTop: 16, fontSize: 13, color: '#94a3b8', display: 'flex', alignItems: 'center', gap: 8 }}>
                    <Lightbulb size={14} color="#f39c12" />
                    The SNN uses less than 0.1% of all MCU resources, leaving room for sensor drivers,
                    communication stack, display rendering, and other firmware tasks.
                </p>
            </div>

            <div className="charts-grid">
                {metrics.map((metric, idx) => {
                    const chartData = models
                        .map(m => ({
                            name: m.name.replace(' (TinyML)', '').replace(' Regression', ''),
                            value: m.deploy[metric.key],
                            color: m.color,
                            isSNN: m.category === 'snn',
                        }))
                        .sort((a, b) => a.value - b.value)

                    return (
                        <div className="chart-card" key={idx}>
                            <h3>{metric.title}</h3>
                            <ResponsiveContainer width="100%" height={280}>
                                <BarChart data={chartData} layout="vertical" margin={{ left: 80, right: 60 }}>
                                    <XAxis type="number" scale="log" domain={['auto', 'auto']}
                                        tick={{ fill: '#94a3b8', fontSize: 11 }}
                                        axisLine={{ stroke: 'rgba(255,255,255,0.1)' }} />
                                    <YAxis type="category" dataKey="name"
                                        tick={{ fill: '#94a3b8', fontSize: 11 }}
                                        axisLine={false} tickLine={false} width={90} />
                                    <Tooltip formatter={(v) => `${v} ${metric.unit}`}
                                        contentStyle={{
                                            background: '#1e293b', border: '1px solid rgba(255,255,255,0.1)',
                                            borderRadius: 10, fontSize: 13
                                        }} />
                                    <Bar dataKey="value" radius={[0, 6, 6, 0]} barSize={18}>
                                        {chartData.map((entry, i) => (
                                            <Cell key={i} fill={entry.isSNN ? '#f39c12' : entry.color}
                                                fillOpacity={entry.isSNN ? 1 : 0.6} />
                                        ))}
                                        <LabelList dataKey="value" position="right" fill="#94a3b8"
                                            formatter={v => `${v} ${metric.unit}`}
                                            style={{ fontSize: 10 }} />
                                    </Bar>
                                </BarChart>
                            </ResponsiveContainer>
                        </div>
                    )
                })}
            </div>

            <div className="card">
                <div className="card-title">Edge Deployment Feasibility</div>
                <table className="data-table">
                    <thead>
                        <tr><th>Model</th><th>Deployable?</th><th>Reason</th></tr>
                    </thead>
                    <tbody>
                        {feasibility.map((f, i) => (
                            <tr key={i} className={f.best ? 'highlight' : ''}>
                                <td>
                                    <span className="model-dot" style={{ background: f.color }} />
                                    {f.name}
                                </td>
                                <td>
                                    <span className={`comparison-badge ${f.deployable === 'Yes' ? 'best' : f.deployable === 'No' ? 'neutral' : 'good'}`}>
                                        {f.deployable}
                                    </span>
                                </td>
                                <td style={{ color: '#94a3b8' }}>{f.reason}</td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
        </div>
    )
}
