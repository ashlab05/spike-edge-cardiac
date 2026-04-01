import {
    ScatterChart, Scatter, XAxis, YAxis, Tooltip, ResponsiveContainer,
    BarChart, Bar, Cell, LabelList
} from 'recharts'
import {
    Zap, Gauge, HardDrive, ShieldCheck, Brain, Rocket, Award,
    Timer, Layers, Heart
} from 'lucide-react'

const CustomTooltip = ({ active, payload }) => {
    if (!active || !payload?.length) return null
    const d = payload[0].payload
    return (
        <div style={{
            background: '#1e293b', border: '1px solid rgba(255,255,255,0.1)',
            borderRadius: 10, padding: '12px 16px', fontSize: 13
        }}>
            <p style={{ fontWeight: 700, marginBottom: 6, color: d.color }}>{d.name}</p>
            <p>F1-Macro: {d.f1?.toFixed(3)}</p>
            <p>Energy: {d.energy} µJ</p>
            <p>Size: {d.size} KB</p>
            <p>Latency: {d.latency} µs</p>
        </div>
    )
}

export default function Tradeoff({ data }) {
    const models = data.models
    const snn = models.find(m => m.name === 'SNN (LIF)')

    const scatterData = models.map(m => ({
        name: m.name,
        f1: m.f1_macro,
        energy: m.deploy.uj,
        size: m.deploy.size_kb,
        latency: m.deploy.us,
        color: m.color,
        isSNN: m.category === 'snn',
    }))

    const energyComparison = models
        .filter(m => m.category === 'ml')
        .map(m => ({
            name: m.name.replace(' (TinyML)', '').replace(' Regression', ''),
            ratio: Math.round(m.deploy.uj / snn.deploy.uj),
            color: m.color,
        }))
        .sort((a, b) => b.ratio - a.ratio)

    const tradeoffWins = [
        {
            Icon: Zap, title: '225× More Energy Efficient',
            desc: 'SNN uses only 0.6 µJ per inference vs Random Forest\'s 135 µJ. At 10 Hz monitoring, that\'s 0.022 J/hour — enabling months-long battery life on wearables.',
            stat: '0.6 µJ'
        },
        {
            Icon: Gauge, title: 'Sub-5µs Deterministic Latency',
            desc: 'With 4µs worst-case latency, the SNN provides hard real-time guarantees critical for cardiac monitoring. Random Forest takes 900µs and is non-deterministic.',
            stat: '4 µs'
        },
        {
            Icon: HardDrive, title: '375× Smaller Model',
            desc: 'At 0.8 KB, the SNN model is 375× smaller than Random Forest (300 KB). It leaves >99.9% of edge MCU resources free for other firmware tasks.',
            stat: '0.8 KB'
        },
        {
            Icon: Heart, title: '4-Class Cardiac Classification',
            desc: 'SNN doesn\'t just detect anomalies — it classifies Normal, Arrhythmia, Hypotensive, and Hypertensive events with competitive F1-macro. Per-class F1 is strongest on Normal and Hypertensive classes, with room for improvement on minority classes.',
            stat: '4 Classes'
        },
        {
            Icon: Brain, title: 'Native Temporal Awareness',
            desc: 'The LIF membrane potential maintains state across 10 timesteps, naturally detecting gradual HR drift and intermittent SpO2 drops — impossible with static classifiers.',
            stat: 'Built-in'
        },
        {
            Icon: Rocket, title: 'Surrogate Gradient Training',
            desc: 'SNN trains via fast sigmoid surrogate gradient with BPTT over T=10 timesteps. Only 130 weights to learn. Spike encoding uses medical thresholds — domain knowledge built-in.',
            stat: '0.3 ms'
        },
    ]

    return (
        <div>
            <div className="page-header">
                <h2>Why SNN is the Optimal Choice</h2>
                <p>4-class cardiac event classification at the edge — the right tradeoff for always-on wearable monitoring</p>
            </div>

            <div className="winner-banner">
                <h2><Zap size={24} style={{ display: 'inline', verticalAlign: 'middle', marginRight: 8 }} />
                    SNN (LIF) — Best Edge AI Tradeoff
                    <Zap size={24} style={{ display: 'inline', verticalAlign: 'middle', marginLeft: 8 }} /></h2>
                <p>Competitive 4-class F1-macro = {snn.f1_macro.toFixed(3)} with 225× energy reduction, 4µs latency, and 0.8 KB model size</p>
            </div>

            <div className="advantages-grid">
                {tradeoffWins.map((adv, i) => (
                    <div className="advantage-card" key={i}>
                        <adv.Icon size={28} color="#f39c12" style={{ marginBottom: 12 }} />
                        <h4>{adv.title}</h4>
                        <p>{adv.desc}</p>
                        <div className="adv-stat">{adv.stat}</div>
                    </div>
                ))}
            </div>

            <div className="charts-grid">
                <div className="chart-card">
                    <h3>F1-Macro vs Energy — The Edge AI Tradeoff</h3>
                    <ResponsiveContainer width="100%" height={360}>
                        <ScatterChart margin={{ top: 20, right: 30, bottom: 20, left: 20 }}>
                            <XAxis type="number" dataKey="energy" name="Energy (µJ)" scale="log" domain={[0.1, 500]}
                                tick={{ fill: '#94a3b8', fontSize: 11 }}
                                axisLine={{ stroke: 'rgba(255,255,255,0.1)' }}
                                label={{ value: 'Energy per Inference (µJ) — Log Scale', position: 'bottom', fill: '#64748b', fontSize: 12 }} />
                            <YAxis type="number" dataKey="f1" name="F1-Macro" domain={[0.6, 1.0]}
                                tick={{ fill: '#94a3b8', fontSize: 11 }}
                                axisLine={{ stroke: 'rgba(255,255,255,0.1)' }}
                                label={{ value: 'F1-Macro (4-Class)', angle: -90, position: 'insideLeft', fill: '#64748b', fontSize: 12 }} />
                            <Tooltip content={<CustomTooltip />} />
                            <Scatter data={scatterData}>
                                {scatterData.map((entry, i) => (
                                    <Cell key={i} fill={entry.color}
                                        r={entry.isSNN ? 12 : 7}
                                        stroke={entry.isSNN ? '#fff' : 'none'}
                                        strokeWidth={entry.isSNN ? 2 : 0}
                                        fillOpacity={entry.isSNN ? 1 : 0.7} />
                                ))}
                            </Scatter>
                        </ScatterChart>
                    </ResponsiveContainer>
                    <p style={{ textAlign: 'center', color: '#64748b', fontSize: 12, marginTop: 8 }}>
                        Lower energy = better · Higher F1 = better · SNN achieves best energy-accuracy tradeoff
                    </p>
                </div>

                <div className="chart-card">
                    <h3>SNN Energy Advantage (× times more efficient)</h3>
                    <ResponsiveContainer width="100%" height={360}>
                        <BarChart data={energyComparison} layout="vertical" margin={{ left: 80, right: 60 }}>
                            <XAxis type="number" tick={{ fill: '#94a3b8', fontSize: 11 }}
                                axisLine={{ stroke: 'rgba(255,255,255,0.1)' }} />
                            <YAxis type="category" dataKey="name" tick={{ fill: '#94a3b8', fontSize: 12 }}
                                axisLine={false} tickLine={false} width={90} />
                            <Bar dataKey="ratio" radius={[0, 6, 6, 0]} barSize={24}>
                                {energyComparison.map((entry, i) => (
                                    <Cell key={i} fill={entry.color} fillOpacity={0.8} />
                                ))}
                                <LabelList dataKey="ratio" position="right" fill="#f39c12"
                                    formatter={v => `${v}×`} style={{ fontSize: 12, fontWeight: 700 }} />
                            </Bar>
                        </BarChart>
                    </ResponsiveContainer>
                </div>
            </div>

            {/* Training Time */}
            <div className="charts-grid" style={{ marginBottom: 24 }}>
                <div className="chart-card">
                    <h3><Timer size={16} style={{ display: 'inline', verticalAlign: 'middle', marginRight: 8 }} />
                        Training Time Comparison</h3>
                    <ResponsiveContainer width="100%" height={340}>
                        <BarChart data={data.training_comparison} layout="vertical" margin={{ left: 90, right: 80 }}>
                            <XAxis type="number" tick={{ fill: '#94a3b8', fontSize: 11 }}
                                axisLine={{ stroke: 'rgba(255,255,255,0.1)' }}
                                label={{ value: 'Training Time (seconds)', position: 'bottom', fill: '#64748b', fontSize: 12 }} />
                            <YAxis type="category" dataKey="name" tick={{ fill: '#94a3b8', fontSize: 12 }}
                                axisLine={false} tickLine={false} width={90} />
                            <Tooltip formatter={(v) => `${v === 0 ? '0 — rule-based' : (v * 1000).toFixed(1) + ' ms'}`}
                                contentStyle={{
                                    background: '#1e293b', border: '1px solid rgba(255,255,255,0.1)',
                                    borderRadius: 10, fontSize: 13
                                }} />
                            <Bar dataKey="train_s" radius={[0, 6, 6, 0]} barSize={20}>
                                {data.training_comparison.map((entry, i) => (
                                    <Cell key={i} fill={entry.color}
                                        fillOpacity={entry.name === 'SNN (LIF)' ? 1 : 0.7} />
                                ))}
                                <LabelList dataKey="train_label" position="right" fill="#94a3b8"
                                    style={{ fontSize: 11, fontWeight: 500 }} />
                            </Bar>
                        </BarChart>
                    </ResponsiveContainer>
                    <div style={{
                        marginTop: 12, background: 'rgba(243,156,18,0.08)', border: '1px solid rgba(243,156,18,0.2)',
                        borderRadius: 10, padding: 14
                    }}>
                        <p style={{ color: '#f39c12', fontSize: 13, fontWeight: 600, marginBottom: 4 }}>
                            SNN trains via surrogate gradient descent with BPTT
                        </p>
                        <p style={{ color: '#94a3b8', fontSize: 12, lineHeight: 1.6 }}>
                            Weights are learned using <strong style={{ color: '#f1f5f9' }}>backpropagation through time</strong> with a
                            fast sigmoid surrogate for the non-differentiable spike function.
                            Training completes in <strong style={{ color: '#f1f5f9' }}>0.3 ms</strong> for all 130 weights.
                            Spike encoding uses medical thresholds, so the model benefits from clinical domain knowledge.
                        </p>
                    </div>
                </div>

                {/* New: per-class F1 chart */}
                <div className="chart-card">
                    <h3><Heart size={16} style={{ display: 'inline', verticalAlign: 'middle', marginRight: 8 }} />
                        SNN Per-Class Detection Performance</h3>
                    <div style={{ padding: '20px 0' }}>
                        {['Normal', 'Arrhythmia', 'Hypotensive', 'Hypertensive'].map((cls, i) => {
                            const colors = ['#2ecc71', '#e74c3c', '#e67e22', '#9b59b6']
                            const f1 = snn.per_class_f1[i]
                            return (
                                <div key={cls} style={{ marginBottom: 16 }}>
                                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
                                        <span style={{ color: colors[i], fontSize: 13, fontWeight: 600 }}>{cls}</span>
                                        <span style={{ color: '#f1f5f9', fontSize: 13, fontWeight: 700 }}>{f1.toFixed(3)}</span>
                                    </div>
                                    <div style={{ height: 8, borderRadius: 4, background: 'rgba(255,255,255,0.05)' }}>
                                        <div style={{
                                            width: `${f1 * 100}%`, height: '100%', borderRadius: 4,
                                            background: `linear-gradient(90deg, ${colors[i]}88, ${colors[i]})`
                                        }} />
                                    </div>
                                </div>
                            )
                        })}
                    </div>
                    <div style={{
                        marginTop: 12, background: 'rgba(46,204,113,0.08)', border: '1px solid rgba(46,204,113,0.2)',
                        borderRadius: 10, padding: 14
                    }}>
                        <p style={{ color: '#2ecc71', fontSize: 13, fontWeight: 600, marginBottom: 4 }}>
                            Clinically Meaningful Classification
                        </p>
                        <p style={{ color: '#94a3b8', fontSize: 12, lineHeight: 1.6 }}>
                            SNN classifies <strong style={{ color: '#f1f5f9' }}>4 distinct cardiac states</strong> — not just normal vs abnormal.
                            This enables targeted clinical responses: arrhythmia alerts trigger ECG review,
                            hypertensive events trigger BP management, and hypotensive events trigger fluid/vasopressor protocols.
                        </p>
                    </div>
                </div>
            </div>

            {/* Model Complexity Table */}
            <div className="card" style={{ marginBottom: 24 }}>
                <div className="card-title"><Layers size={14} style={{ display: 'inline', verticalAlign: 'middle', marginRight: 6 }} />
                    Model Complexity Comparison (4-Class)</div>
                <table className="data-table">
                    <thead>
                        <tr><th>Model</th><th>Architecture</th><th>Parameters</th><th>Training</th><th>Weight Source</th></tr>
                    </thead>
                    <tbody>
                        {[
                            ['SNN (LIF)', '12→10→4 LIF', '130 weights', '0.3 ms', 'Surrogate gradient', true],
                            ['Threshold', '6 rules', '6 thresholds', 'Not needed', 'Manual rules', false],
                            ['Logistic Reg.', '6→4 linear', '28 params', '11.5 ms', 'Gradient descent', false],
                            ['Decision Tree', 'Depth-8 tree', '~100 nodes', '7.5 ms', 'CART splitting', false],
                            ['MLP (TinyML)', '6→32→16→4', '692 params', '208.0 ms', 'Backpropagation', false],
                            ['LightGBM', '150 trees', '~40K', '14.2 ms', 'Histogram gradient', false],
                            ['XGBoost', '150 trees', '~48K', '28.3 ms', 'Gradient boosting', false],
                            ['k-NN', 'k=5 nearest', 'N/A', '~0 ms', 'Stores all data', false],
                            ['Random Forest', '150 trees', '~65K', '167.3 ms', 'Bagging + splits', false],
                        ].map(([model, arch, params, train, source, isSNN], i) => (
                            <tr key={i} className={isSNN ? 'highlight' : ''}>
                                <td style={{ fontWeight: 600 }}>{model}</td>
                                <td>{arch}</td>
                                <td>{isSNN ? <span style={{ color: '#f39c12', fontWeight: 700 }}>{params}</span> : params}</td>
                                <td>{train === 'Not needed' ? <span className="comparison-badge best">{train}</span> : train}</td>
                                <td style={{ color: isSNN ? '#f39c12' : '#64748b' }}>{source}</td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>

            {/* Criterion Winner */}
            <div className="card">
                <div className="card-title"><Award size={14} style={{ display: 'inline', verticalAlign: 'middle', marginRight: 6 }} />
                    Criterion-by-Criterion Winner</div>
                <table className="data-table">
                    <thead>
                        <tr><th>Criterion</th><th>Winner</th><th>Runner-Up</th><th>Why It Matters</th></tr>
                    </thead>
                    <tbody>
                        {[
                            ['F1-Macro (4-Class)', 'XGBoost', 'Random Forest', 'Best multi-class F1-macro = 0.901'],
                            ['Energy Efficiency', 'SNN (LIF)', 'Threshold', 'Battery life for wearables'],
                            ['Memory Footprint', 'SNN (LIF)', 'Threshold', 'Fits on any microcontroller'],
                            ['Inference Latency', 'SNN (LIF)', 'Threshold', 'Real-time cardiac monitoring'],
                            ['Training Speed', 'SNN (LIF)', 'k-NN', 'Instant retraining on edge'],
                            ['Temporal Awareness', 'SNN (LIF)', '—', 'Detects gradual anomalies'],
                            ['Deployment Simplicity', 'SNN (LIF)', 'Decision Tree', 'Pure C, no runtime'],
                            ['Normal Detection', 'XGBoost', 'Random Forest', 'F1=0.983 — best normal detection'],
                            ['Hypertensive Detection', 'Random Forest', 'SNN (LIF)', 'F1=0.961 vs SNN 0.948'],
                            ['Edge AI Tradeoff', 'SNN (LIF)', 'MLP (TinyML)', 'Best F1-per-microjoule ratio'],
                        ].map(([criterion, winner, runner, why], i) => (
                            <tr key={i} className={winner === 'SNN (LIF)' ? 'highlight' : ''}>
                                <td style={{ fontWeight: 600 }}>{criterion}</td>
                                <td>
                                    {winner === 'SNN (LIF)' && <span className="comparison-badge best">BEST</span>}
                                    {' '}{winner}
                                </td>
                                <td>{runner}</td>
                                <td style={{ color: '#64748b' }}>{why}</td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
        </div>
    )
}
