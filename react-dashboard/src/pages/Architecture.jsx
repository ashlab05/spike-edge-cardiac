import {
    Lightbulb, CheckCircle2, CircleDot, Activity,
    Cloud, CloudOff, Shield, Database, Users, Heart
} from 'lucide-react'

const CLASS_COLORS = { Normal: '#2ecc71', Arrhythmia: '#e74c3c', Hypotensive: '#e67e22', Hypertensive: '#9b59b6' }

export default function Architecture({ data }) {
    const snn = data.models.find(m => m.name === 'SNN (LIF)')
    const cm = data.confusion_matrices['SNN (LIF)']
    const classNames = data.class_names || ['Normal', 'Arrhythmia', 'Hypotensive', 'Hypertensive']

    const pipelineSteps = [
        {
            step: 1,
            title: 'Sensor Acquisition',
            subtitle: 'Reads from 3 sensors + PTT estimation',
            Icon: Activity,
            color: '#3498db',
            details: [
                'MAX30100 → Heart Rate (72 bpm) + SpO2 (98%)',
                'DS18B20 → Body Temperature (98.2°F)',
                'AD8232 ECG → Respiratory Rate (16 brpm) + HRV SDNN (52 ms)',
                'PTT (MAX30100 + ECG) → Blood Pressure est. (118 mmHg)',
            ],
            example: 'Raw: HR=72, SpO2=98, Temp=98.2, RR=16, HRV=52, BP=118',
        },
        {
            step: 2,
            title: 'Directional Spike Encoding',
            subtitle: '6 features × 2 directions (too_low, too_high) = 12 spike channels',
            Icon: CircleDot,
            color: '#e67e22',
            details: [
                'HR 72 → within 60–100 → low_spike=0, high_spike=0',
                'SpO2 98% → within 95–100 → low_spike=0, high_spike=0',
                'Temp 98.2°F → within 96.5–99.5 → low_spike=0, high_spike=0',
                'RR 16 → within 12–20 → low_spike=0, high_spike=0',
                'HRV 52 ms → within 30–70 → low_spike=0, high_spike=0',
                'BP 118 mmHg → within 90–140 → low_spike=0, high_spike=0',
            ],
            example: 'Spikes: [0,0, 0,0, 0,0, 0,0, 0,0, 0,0] → Normal, all vitals in range',
        },
        {
            step: 3,
            title: 'Hidden Layer Processing',
            subtitle: '10 LIF neurons integrate 12 weighted spike inputs over T=10 timesteps',
            Icon: Activity,
            color: '#f39c12',
            details: [
                'Each hidden neuron receives 12 weighted inputs (12×10 = 120 weights)',
                'V(t+1) = 0.85 × V(t) × (1 − spike) + Σ(Wᵢ × spikeᵢ) + bᵢ',
                'Membrane decays by α=0.85 each timestep (leaky), resets after spike',
                'Normal patient: all spikes=0 → V driven by bias → stable pattern',
            ],
            example: 'Hidden activity varies by class — different spike patterns emerge',
        },
        {
            step: 4,
            title: 'Multi-Class Output (Winner-Take-All)',
            subtitle: '4 output LIF neurons — one per cardiac condition — spike counts decide class',
            Icon: Shield,
            color: '#2ecc71',
            details: [
                'Output neuron 0: Normal — fires most for healthy vitals',
                'Output neuron 1: Arrhythmia — fires for erratic HR + low HRV',
                'Output neuron 2: Hypotensive — fires for low BP patterns',
                'Output neuron 3: Hypertensive — fires for high BP + tachycardia',
            ],
            example: 'Spike counts: [8, 0, 0, 0] → Class 0 (Normal)',
        },
    ]

    const anomalyExamples = [
        {
            label: 'Arrhythmia Case',
            color: '#e74c3c',
            sensors: 'HR=45, SpO2=92%, Temp=98.5°F, RR=22, HRV=15, BP=125',
            spikes: '[1,0, 1,0, 0,0, 0,1, 1,0, 0,0]',
            spikeDetail: '4 of 12 channels active — bradycardia, low SpO2, low HRV, high RR',
            output: 'Spike counts: [1, 7, 0, 1] → Class 1 → ARRHYTHMIA',
        },
        {
            label: 'Hypertensive Crisis',
            color: '#9b59b6',
            sensors: 'HR=115, SpO2=93%, Temp=99.8°F, RR=26, HRV=22, BP=168',
            spikes: '[0,1, 1,0, 0,1, 0,1, 1,0, 0,1]',
            spikeDetail: '6 of 12 channels active — tachycardia, low SpO2, high temp, tachypnea, low HRV, high BP',
            output: 'Spike counts: [0, 1, 0, 8] → Class 3 → HYPERTENSIVE CRISIS',
        },
    ]

    return (
        <div>
            <div className="page-header">
                <h2>SNN Architecture — 4-Class Cardiac Classifier</h2>
                <p>LIF spiking neural network: 12 spike inputs → 10 hidden LIF → 4 output LIF (one per cardiac condition)</p>
            </div>

            {/* Architecture summary */}
            <div className="card" style={{ marginBottom: 24 }}>
                <div className="card-title">
                    <Heart size={14} style={{ display: 'inline', verticalAlign: 'middle', marginRight: 6 }} />
                    Architecture Overview: 12 → 10 → 4 LIF Network
                </div>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 16 }}>
                    <div style={{ padding: 20, borderRadius: 12, background: 'rgba(52,152,219,0.08)', border: '1px solid rgba(52,152,219,0.2)', textAlign: 'center' }}>
                        <div style={{ fontSize: 32, fontWeight: 800, color: '#3498db' }}>12</div>
                        <div style={{ fontSize: 12, color: '#94a3b8', marginTop: 4 }}>Spike Input Channels</div>
                        <div style={{ fontSize: 11, color: '#64748b', marginTop: 4 }}>6 features × 2 directions</div>
                    </div>
                    <div style={{ padding: 20, borderRadius: 12, background: 'rgba(243,156,18,0.08)', border: '1px solid rgba(243,156,18,0.2)', textAlign: 'center' }}>
                        <div style={{ fontSize: 32, fontWeight: 800, color: '#f39c12' }}>10</div>
                        <div style={{ fontSize: 12, color: '#94a3b8', marginTop: 4 }}>Hidden LIF Neurons</div>
                        <div style={{ fontSize: 11, color: '#64748b', marginTop: 4 }}>α=0.85, θ=0.8</div>
                    </div>
                    <div style={{ padding: 20, borderRadius: 12, background: 'rgba(46,204,113,0.08)', border: '1px solid rgba(46,204,113,0.2)', textAlign: 'center' }}>
                        <div style={{ fontSize: 32, fontWeight: 800, color: '#2ecc71' }}>4</div>
                        <div style={{ fontSize: 12, color: '#94a3b8', marginTop: 4 }}>Output Classes</div>
                        <div style={{ fontSize: 11, color: '#64748b', marginTop: 4 }}>Winner-take-all</div>
                    </div>
                </div>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr 1fr', gap: 12, marginTop: 16 }}>
                    {classNames.map((name, i) => (
                        <div key={name} style={{
                            padding: 12, borderRadius: 8, textAlign: 'center',
                            background: `${CLASS_COLORS[name]}10`, border: `1px solid ${CLASS_COLORS[name]}30`
                        }}>
                            <div style={{ fontSize: 13, fontWeight: 700, color: CLASS_COLORS[name] }}>{name}</div>
                            <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 4 }}>
                                F1 = {snn.per_class_f1[i].toFixed(3)}
                            </div>
                        </div>
                    ))}
                </div>
            </div>

            {/* Step-by-step pipeline — Normal case */}
            <div className="card" style={{ marginBottom: 24 }}>
                <div className="card-title">
                    <CheckCircle2 size={14} style={{ display: 'inline', verticalAlign: 'middle', marginRight: 6, color: '#2ecc71' }} />
                    Step-by-Step Pipeline — Normal Patient
                </div>
                {pipelineSteps.map((ps, i) => (
                    <div key={i} style={{
                        display: 'grid', gridTemplateColumns: '60px 1fr', gap: 16,
                        marginBottom: i < pipelineSteps.length - 1 ? 20 : 0,
                        paddingBottom: i < pipelineSteps.length - 1 ? 20 : 0,
                        borderBottom: i < pipelineSteps.length - 1 ? '1px solid rgba(255,255,255,0.05)' : 'none'
                    }}>
                        <div style={{
                            width: 48, height: 48, borderRadius: 12, display: 'flex', alignItems: 'center', justifyContent: 'center',
                            background: `${ps.color}15`, border: `1px solid ${ps.color}30`
                        }}>
                            <ps.Icon size={22} color={ps.color} />
                        </div>
                        <div>
                            <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6 }}>
                                <span style={{ fontSize: 11, color: ps.color, fontWeight: 700, background: `${ps.color}15`, padding: '2px 8px', borderRadius: 4 }}>STEP {ps.step}</span>
                                <span style={{ fontSize: 14, fontWeight: 600, color: '#f1f5f9' }}>{ps.title}</span>
                            </div>
                            <p style={{ fontSize: 12, color: '#94a3b8', marginBottom: 8, lineHeight: 1.5 }}>{ps.subtitle}</p>
                            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 6 }}>
                                {ps.details.map((d, j) => (
                                    <div key={j} style={{ fontSize: 11, color: '#cbd5e1', lineHeight: 1.5, padding: '4px 8px', borderRadius: 4, background: 'rgba(255,255,255,0.02)' }}>
                                        {d}
                                    </div>
                                ))}
                            </div>
                            <div style={{ marginTop: 8, padding: '6px 10px', borderRadius: 6, background: `${ps.color}08`, border: `1px solid ${ps.color}20` }}>
                                <code style={{ fontSize: 11, color: ps.color }}>{ps.example}</code>
                            </div>
                        </div>
                    </div>
                ))}
                <div style={{
                    marginTop: 16, padding: 14, borderRadius: 10, textAlign: 'center',
                    background: 'rgba(46,204,113,0.1)', border: '1px solid rgba(46,204,113,0.3)'
                }}>
                    <p style={{ color: '#2ecc71', fontSize: 13, fontWeight: 600 }}>
                        Result: NORMAL — all vitals in range, output neuron 0 wins → Classification complete in 4 µs
                    </p>
                </div>
            </div>

            {/* Anomaly Examples */}
            <div className="card" style={{ marginBottom: 24 }}>
                <div className="card-title" style={{ color: '#e74c3c' }}>
                    <Shield size={14} style={{ display: 'inline', verticalAlign: 'middle', marginRight: 6 }} />
                    Anomaly Classification Examples
                </div>
                <div className="charts-grid" style={{ marginBottom: 0 }}>
                    {anomalyExamples.map((ae, i) => (
                        <div key={i} style={{
                            padding: 16, borderRadius: 12,
                            background: `${ae.color}08`, border: `1px solid ${ae.color}20`
                        }}>
                            <h4 style={{ color: ae.color, fontSize: 14, marginBottom: 12 }}>{ae.label}</h4>
                            {[
                                { label: 'Sensors', value: ae.sensors },
                                { label: 'Spikes', value: ae.spikes },
                                { label: 'Detail', value: ae.spikeDetail },
                                { label: 'Output', value: ae.output },
                            ].map((row, j) => (
                                <div key={j} style={{ marginBottom: 8, padding: '6px 10px', borderRadius: 6, background: 'rgba(0,0,0,0.2)' }}>
                                    <span style={{ fontSize: 10, color: '#64748b', display: 'block', marginBottom: 2 }}>{row.label}</span>
                                    <code style={{ fontSize: 11, color: j === 3 ? ae.color : '#cbd5e1', fontWeight: j === 3 ? 700 : 400 }}>{row.value}</code>
                                </div>
                            ))}
                        </div>
                    ))}
                </div>
            </div>

            {/* Surrogate Gradient Training */}
            <div className="card" style={{ marginBottom: 24 }}>
                <div className="card-title">
                    <Lightbulb size={14} style={{ display: 'inline', verticalAlign: 'middle', marginRight: 6 }} />
                    How SNN Weights Are Trained (Surrogate Gradient Descent)
                </div>
                <p style={{ color: '#94a3b8', fontSize: 13, marginBottom: 20, lineHeight: 1.7 }}>
                    SNN weights are learned via <strong style={{ color: '#f1f5f9' }}>backpropagation through time (BPTT)</strong> using
                    a fast sigmoid surrogate gradient. The spike encoding layer uses medical thresholds,
                    combining <strong style={{ color: '#f1f5f9' }}>domain knowledge with data-driven learning</strong>.
                </p>

                <div className="charts-grid" style={{ marginBottom: 0 }}>
                    <div style={{ padding: 20, borderRadius: 12, background: 'rgba(231,76,60,0.05)', border: '1px solid rgba(231,76,60,0.15)' }}>
                        <h4 style={{ color: '#e74c3c', fontSize: 14, marginBottom: 16, display: 'flex', alignItems: 'center', gap: 8 }}>
                            <Cloud size={16} /> Traditional ML Training
                        </h4>
                        {[
                            { icon: Database, text: 'Collect and label patient samples', time: 'Same data' },
                            { icon: Users, text: 'Train on GPU — 100+ epochs', time: '208 ms (MLP)' },
                            { icon: Activity, text: 'Model learns all features from scratch', time: 'No domain knowledge' },
                            { icon: Cloud, text: 'Large model → OTA firmware update', time: '2–300 KB' },
                        ].map((item, i) => (
                            <div key={i} style={{
                                display: 'flex', alignItems: 'center', gap: 12, marginBottom: 10,
                                padding: '8px 12px', borderRadius: 8, background: 'rgba(231,76,60,0.05)',
                            }}>
                                <item.icon size={16} color="#e74c3c" style={{ flexShrink: 0 }} />
                                <span style={{ color: '#cbd5e1', fontSize: 12, flex: 1 }}>{item.text}</span>
                                <span style={{ color: '#e74c3c', fontSize: 11, fontWeight: 600, whiteSpace: 'nowrap' }}>{item.time}</span>
                            </div>
                        ))}
                    </div>

                    <div style={{ padding: 20, borderRadius: 12, background: 'rgba(46,204,113,0.05)', border: '1px solid rgba(46,204,113,0.15)' }}>
                        <h4 style={{ color: '#2ecc71', fontSize: 14, marginBottom: 16, display: 'flex', alignItems: 'center', gap: 8 }}>
                            <CloudOff size={16} /> SNN Surrogate Gradient Training
                        </h4>
                        {[
                            { icon: Lightbulb, text: 'Spike encoding uses medical thresholds (built-in)', time: 'Domain knowledge' },
                            { icon: CircleDot, text: 'Surrogate gradient + BPTT over T=10 timesteps', time: '0.3 ms' },
                            { icon: Shield, text: 'Only 130 weights to learn (12×10 + 10×4 + biases)', time: 'Tiny model' },
                            { icon: CheckCircle2, text: 'Trained weights flashed to edge firmware', time: 'Seconds' },
                        ].map((item, i) => (
                            <div key={i} style={{
                                display: 'flex', alignItems: 'center', gap: 12, marginBottom: 10,
                                padding: '8px 12px', borderRadius: 8, background: 'rgba(46,204,113,0.05)',
                            }}>
                                <item.icon size={16} color="#2ecc71" style={{ flexShrink: 0 }} />
                                <span style={{ color: '#cbd5e1', fontSize: 12, flex: 1 }}>{item.text}</span>
                                <span style={{ color: '#2ecc71', fontSize: 11, fontWeight: 600, whiteSpace: 'nowrap' }}>{item.time}</span>
                            </div>
                        ))}
                    </div>
                </div>

                <div style={{
                    marginTop: 20, padding: 16, borderRadius: 10, textAlign: 'center',
                    background: 'rgba(243,156,18,0.08)', border: '1px solid rgba(243,156,18,0.2)'
                }}>
                    <p style={{ color: '#f39c12', fontSize: 13, fontWeight: 600, marginBottom: 4 }}>
                        Best of Both Worlds: Domain Knowledge + Data-Driven Learning
                    </p>
                    <p style={{ color: '#94a3b8', fontSize: 12, lineHeight: 1.6 }}>
                        SNN combines <strong style={{ color: '#f1f5f9' }}>medical threshold-based spike encoding</strong> (no learning needed) with
                        <strong style={{ color: '#f1f5f9' }}> surrogate gradient-trained synaptic weights</strong>. 130 parameters in 0.8 KB —
                        classifies 4 cardiac conditions in 4 µs on-device.
                    </p>
                </div>
            </div>

            {/* LIF Dynamics */}
            <div className="charts-grid">
                <div className="card">
                    <div className="card-title">LIF Neuron Dynamics</div>
                    <div style={{ padding: '16px 0' }}>
                        <div style={{
                            background: 'linear-gradient(135deg, rgba(243,156,18,0.08), transparent)',
                            borderRadius: 12, padding: 24, border: '1px solid rgba(243,156,18,0.15)', marginBottom: 16
                        }}>
                            <h4 style={{ color: '#f39c12', marginBottom: 12, fontSize: 15 }}>Membrane Potential Equation</h4>
                            <code style={{
                                display: 'block', fontSize: 18, color: '#f1f5f9', padding: 16,
                                background: 'rgba(0,0,0,0.3)', borderRadius: 8, textAlign: 'center'
                            }}>
                                V(t+1) = α · V(t) · (1 − s(t)) + Σ(Wᵢ · spikeᵢ) + b
                            </code>
                            <p style={{ color: '#94a3b8', fontSize: 12, marginTop: 12, lineHeight: 1.6, textAlign: 'center' }}>
                                Where α = 0.85 (decay), θ = 0.8 (threshold), s(t) = spike output at time t
                            </p>
                        </div>
                        {[
                            { param: 'α (decay)', value: '0.85', desc: 'Membrane leaks 15% each timestep — temporal memory', color: '#3498db' },
                            { param: 'θ (threshold)', value: '0.8', desc: 'Neuron fires when membrane potential ≥ 0.8', color: '#e67e22' },
                            { param: 'T (timesteps)', value: '10', desc: 'Runs 10 timesteps per inference for temporal integration', color: '#2ecc71' },
                            { param: 'Reset', value: 'Hard', desc: 'V resets to 0 after spike — prevents runaway activation', color: '#e74c3c' },
                        ].map((item, i) => (
                            <div key={i} style={{
                                display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                                padding: 12, borderRadius: 8, marginBottom: 8,
                                background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.04)'
                            }}>
                                <div>
                                    <span style={{ color: item.color, fontWeight: 600, fontSize: 13 }}>{item.param}</span>
                                    <span style={{ color: '#64748b', fontSize: 12, marginLeft: 12 }}>{item.desc}</span>
                                </div>
                                <span style={{ color: '#f1f5f9', fontWeight: 700, fontSize: 15 }}>{item.value}</span>
                            </div>
                        ))}
                    </div>
                </div>

                {/* Confusion Matrix */}
                <div className="card">
                    <div className="card-title">SNN Confusion Matrix (4-Class)</div>
                    <div style={{ padding: '16px 0' }}>
                        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                            <thead>
                                <tr>
                                    <th style={{ padding: 8, color: '#64748b', textAlign: 'left' }}>Actual ↓ / Pred →</th>
                                    {classNames.map(name => (
                                        <th key={name} style={{ padding: 8, color: CLASS_COLORS[name], textAlign: 'center', fontSize: 11 }}>{name}</th>
                                    ))}
                                </tr>
                            </thead>
                            <tbody>
                                {classNames.map((name, i) => (
                                    <tr key={name}>
                                        <td style={{ padding: 8, color: CLASS_COLORS[name], fontWeight: 600, fontSize: 11 }}>{name}</td>
                                        {cm[i].map((val, j) => (
                                            <td key={j} style={{
                                                padding: 8, textAlign: 'center', fontWeight: i === j ? 700 : 400,
                                                color: i === j ? '#2ecc71' : val > 0 ? '#e74c3c' : '#64748b',
                                                background: i === j ? 'rgba(46,204,113,0.1)' : val > 0 ? 'rgba(231,76,60,0.05)' : 'transparent',
                                                borderRadius: 4,
                                            }}>
                                                {val}
                                            </td>
                                        ))}
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                        <div style={{
                            marginTop: 16, display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8
                        }}>
                            {classNames.map((name, i) => (
                                <div key={name} style={{
                                    padding: '8px 12px', borderRadius: 6,
                                    background: `${CLASS_COLORS[name]}08`, border: `1px solid ${CLASS_COLORS[name]}20`,
                                    display: 'flex', justifyContent: 'space-between', alignItems: 'center'
                                }}>
                                    <span style={{ fontSize: 11, color: CLASS_COLORS[name] }}>{name}</span>
                                    <span style={{ fontSize: 12, color: '#f1f5f9', fontWeight: 600 }}>
                                        P={snn.per_class_prec[i].toFixed(2)} R={snn.per_class_rec[i].toFixed(2)}
                                    </span>
                                </div>
                            ))}
                        </div>
                    </div>
                </div>
            </div>

            {/* Sensor Mapping */}
            <div className="card" style={{ marginTop: 24 }}>
                <div className="card-title">Sensor → Feature → Spike Mapping</div>
                <table className="data-table">
                    <thead>
                        <tr>
                            <th>Sensor</th>
                            <th>Feature</th>
                            <th>Normal Range</th>
                            <th>Spike: Too Low</th>
                            <th>Spike: Too High</th>
                        </tr>
                    </thead>
                    <tbody>
                        {[
                            ['MAX30100', 'Heart Rate', '60–100 bpm', 'HR < 60 (bradycardia)', 'HR > 100 (tachycardia)'],
                            ['MAX30100', 'SpO2', '95–100%', 'SpO2 < 95% (hypoxemia)', '—'],
                            ['DS18B20', 'Body Temperature', '96.5–99.5°F', 'Temp < 96.5 (hypothermia)', 'Temp > 99.5 (fever)'],
                            ['AD8232 ECG', 'Respiratory Rate', '12–20 brpm', 'RR < 12 (bradypnea)', 'RR > 20 (tachypnea)'],
                            ['AD8232 ECG', 'HRV SDNN', '30–70 ms', 'HRV < 30 (low variability)', 'HRV > 70 (high variability)'],
                            ['PTT est.', 'BP Systolic', '90–140 mmHg', 'BP < 90 (hypotension)', 'BP > 140 (hypertension)'],
                        ].map(([sensor, feature, range_, low, high], i) => (
                            <tr key={i}>
                                <td style={{ color: '#3498db', fontWeight: 600 }}>{sensor}</td>
                                <td>{feature}</td>
                                <td style={{ color: '#2ecc71' }}>{range_}</td>
                                <td style={{ fontSize: 11, color: '#e67e22' }}>{low}</td>
                                <td style={{ fontSize: 11, color: '#e74c3c' }}>{high}</td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>

            {/* Key Parameters */}
            <div className="card" style={{ marginTop: 24 }}>
                <div className="card-title">Key Parameters</div>
                <div style={{ padding: '8px 0' }}>
                    {[
                        ['Architecture', '12→10→4', 'Input spikes → Hidden LIF → Output classes'],
                        ['Total Weights', '130', '12×10 + 10×4 = 160 synapses + 14 biases'],
                        ['Model Size', '0.8 KB', '~170 float32 values × 4 bytes + overhead'],
                        ['Weight Source', 'Trained', 'Surrogate gradient descent with BPTT'],
                        ['Spike Encoding', 'Directional', '2 spikes per feature (too_low, too_high)'],
                        ['Timesteps', '10', 'BPTT unrolls over T=10 steps per sample'],
                        ['Output', '4-class', 'Normal / Arrhythmia / Hypotensive / Hypertensive'],
                    ].map(([param, value, desc], i) => (
                        <div key={i} style={{
                            display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                            padding: '10px 14px', borderRadius: 8, marginBottom: 6,
                            background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.04)'
                        }}>
                            <span style={{ color: '#94a3b8', fontSize: 12 }}>{param}</span>
                            <div style={{ textAlign: 'right' }}>
                                <span style={{ color: '#f1f5f9', fontWeight: 600, fontSize: 13, marginRight: 8 }}>{value}</span>
                                <span style={{ color: '#64748b', fontSize: 11 }}>{desc}</span>
                            </div>
                        </div>
                    ))}
                </div>
            </div>
        </div>
    )
}
