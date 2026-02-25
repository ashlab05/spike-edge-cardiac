/**
 * app.js — Spike-Edge Cardiac Monitor
 * WebSocket client, real-time Chart.js charts, simulator slider controls.
 */

// ── Constants ────────────────────────────────────────────────────────────────
const WS_URL       = `ws://${location.host}/ws`;
const MAX_POINTS   = 60;          // rolling window (60 ticks = ~6 seconds)
const DEBOUNCE_MS  = 180;

// ── DOM refs ─────────────────────────────────────────────────────────────────
const $ = id => document.getElementById(id);

const wsBadge    = $('ws-badge');
const wsLabel    = $('ws-label');
const valHr      = $('val-hr');
const valSpo2    = $('val-spo2');
const valTemp    = $('val-temp');
const valStatus  = $('val-status');
const valMode    = $('val-mode');
const cardStatus = $('card-status');

const sliderHr   = $('slider-hr');
const sliderSpo2 = $('slider-spo2');
const sliderTemp = $('slider-temp');
const dispHr     = $('disp-hr');
const dispSpo2   = $('disp-spo2');
const dispTemp   = $('disp-temp');
const prevHr     = $('prev-hr');
const prevSpo2   = $('prev-spo2');
const prevTemp   = $('prev-temp');
const injectStatus = $('inject-status');

const btnInject  = $('btn-inject');
const btnReset   = $('btn-reset');

// ── Chart.js setup ───────────────────────────────────────────────────────────
Chart.defaults.color = '#64748b';
Chart.defaults.font  = { family: "'Roboto Mono', monospace", size: 11 };

const chartDefaults = {
  type: 'line',
  options: {
    animation: false,
    responsive: true,
    maintainAspectRatio: false,
    elements: {
      point: { radius: 0 },
      line:  { tension: 0.35, borderWidth: 2 },
    },
    plugins: { legend: { display: false } },
    scales: {
      x: {
        grid:   { color: 'rgba(255,255,255,0.04)' },
        border: { color: 'rgba(255,255,255,0.08)' },
        ticks:  { maxTicksLimit: 6 },
      },
      y: {
        grid:   { color: 'rgba(255,255,255,0.04)' },
        border: { color: 'rgba(255,255,255,0.08)' },
      },
    },
  },
};

function makeDataset(color, label) {
  return {
    label,
    data: [],
    borderColor: color,
    backgroundColor: color.replace(')', ', 0.08)').replace('rgb(', 'rgba('),
    fill: true,
  };
}

// HR chart
const hrChart = new Chart($('chart-hr'), {
  ...JSON.parse(JSON.stringify(chartDefaults)),
  data: {
    labels: [],
    datasets: [makeDataset('rgb(0,212,255)', 'HR')],
  },
});
hrChart.options.scales.y.min = 40;
hrChart.options.scales.y.max = 180;

// SpO2 chart
const spo2Chart = new Chart($('chart-spo2'), {
  ...JSON.parse(JSON.stringify(chartDefaults)),
  data: {
    labels: [],
    datasets: [makeDataset('rgb(46,213,115)', 'SpO2')],
  },
});
spo2Chart.options.scales.y.min = 80;
spo2Chart.options.scales.y.max = 100;

// Temp chart
const tempChart = new Chart($('chart-temp'), {
  ...JSON.parse(JSON.stringify(chartDefaults)),
  data: {
    labels: [],
    datasets: [makeDataset('rgb(255,165,2)', 'Temp')],
  },
});
tempChart.options.scales.y.min = 35;
tempChart.options.scales.y.max = 40;

function pushPoint(chart, label, value) {
  chart.data.labels.push(label);
  chart.data.datasets[0].data.push(value);
  if (chart.data.labels.length > MAX_POINTS) {
    chart.data.labels.shift();
    chart.data.datasets[0].data.shift();
  }
  chart.update('none');
}

// ── WebSocket ─────────────────────────────────────────────────────────────────
let ws = null;
let reconnectTimer = null;

function connect() {
  ws = new WebSocket(WS_URL);

  ws.onopen = () => {
    wsBadge.className = 'status-badge connected';
    wsLabel.textContent = 'Live';
    clearTimeout(reconnectTimer);
  };

  ws.onmessage = ({ data }) => {
    const d = JSON.parse(data);
    updateMonitor(d);
  };

  ws.onclose = () => {
    wsBadge.className = 'status-badge disconnected';
    wsLabel.textContent = 'Disconnected';
    reconnectTimer = setTimeout(connect, 2000);
  };

  ws.onerror = () => ws.close();
}

function updateMonitor(d) {
  // metric cards
  valHr.textContent   = d.hr;
  valSpo2.textContent = d.spo2;
  valTemp.textContent = d.temp;

  if (d.anomaly === 1) {
    valStatus.textContent  = 'ANOMALY';
    valStatus.className    = 'metric-value red';
    cardStatus.classList.add('anomaly-active');
  } else {
    valStatus.textContent  = 'NORMAL';
    valStatus.className    = 'metric-value cyan';
    cardStatus.classList.remove('anomaly-active');
  }

  valMode.textContent = d.mode === 'custom' ? 'Custom Override' : 'Simulator';

  // charts
  const label = String(Math.round(d.time));
  pushPoint(hrChart,   label, d.hr);
  pushPoint(spo2Chart, label, d.spo2);
  pushPoint(tempChart, label, d.temp);
}

connect();

// ── Tab switching ─────────────────────────────────────────────────────────────
document.querySelectorAll('.tab-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById(`tab-${btn.dataset.tab}`).classList.add('active');
  });
});

// ── Simulator sliders ─────────────────────────────────────────────────────────
let configDebounce = null;

function syncSliderDisplay() {
  const hr   = parseFloat(sliderHr.value);
  const spo2 = parseFloat(sliderSpo2.value);
  const temp = parseFloat(sliderTemp.value);

  dispHr.textContent   = `${hr} bpm`;
  dispSpo2.textContent = `${spo2} %`;
  dispTemp.textContent = `${temp} °C`;
  prevHr.textContent   = `${hr} bpm`;
  prevSpo2.textContent = `${spo2} %`;
  prevTemp.textContent = `${temp} °C`;
}

function sendConfig() {
  fetch('/simulator/config', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      hr:   parseFloat(sliderHr.value),
      spo2: parseFloat(sliderSpo2.value),
      temp: parseFloat(sliderTemp.value),
    }),
  });
}

[sliderHr, sliderSpo2, sliderTemp].forEach(s => {
  s.addEventListener('input', () => {
    syncSliderDisplay();
    clearTimeout(configDebounce);
    configDebounce = setTimeout(sendConfig, DEBOUNCE_MS);
  });
});

// Init display
syncSliderDisplay();

// ── Buttons ───────────────────────────────────────────────────────────────────
btnInject.addEventListener('click', () => {
  fetch('/simulator/inject', { method: 'POST' });
  injectStatus.classList.add('visible');
  setTimeout(() => injectStatus.classList.remove('visible'), 6000);
});

btnReset.addEventListener('click', () => {
  fetch('/simulator/reset', { method: 'POST' });
  sliderHr.value   = 72;
  sliderSpo2.value = 98;
  sliderTemp.value = 36.8;
  syncSliderDisplay();
  injectStatus.classList.remove('visible');
});
