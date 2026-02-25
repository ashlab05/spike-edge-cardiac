"""
FastAPI dashboard server for Spike-Edge Cardiac Anomaly Detection.

Endpoints:
  GET  /              → serve index.html
  WS   /ws            → real-time data stream at 10 Hz
  POST /simulator/config   → update slider values {hr, spo2, temp}
  POST /simulator/inject   → trigger 5-second anomaly burst
  POST /simulator/reset    → reset to baseline
  GET  /simulator/status   → current simulator state

Run:
  uvicorn main:app --reload --port 8000
"""

import asyncio
import json
from pathlib import Path
from typing import Set

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from simulator import SimulatorEngine

# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(title="Spike-Edge Cardiac Monitor")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ── Shared state ──────────────────────────────────────────────────────────────
simulator = SimulatorEngine()
active_connections: Set[WebSocket] = set()

# ── Background tick task ──────────────────────────────────────────────────────
async def broadcast_loop():
    """Tick the simulator and push data to all WebSocket clients at 10 Hz."""
    while True:
        await asyncio.sleep(0.1)
        simulator.tick()
        data = json.dumps(simulator.get_state())
        dead = set()
        for ws in active_connections:
            try:
                await ws.send_text(data)
            except Exception:
                dead.add(ws)
        active_connections.difference_update(dead)


@app.on_event("startup")
async def startup_event():
    asyncio.create_task(broadcast_loop())


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_connections.add(websocket)
    try:
        while True:
            # keep the connection alive; data is pushed by broadcast_loop
            await websocket.receive_text()
    except WebSocketDisconnect:
        active_connections.discard(websocket)


# ── Simulator REST API ────────────────────────────────────────────────────────
class SimConfig(BaseModel):
    hr: float
    spo2: float
    temp: float


@app.post("/simulator/config")
async def set_config(cfg: SimConfig):
    simulator.set_config(cfg.hr, cfg.spo2, cfg.temp)
    return {"status": "ok", "config": cfg.model_dump()}


@app.post("/simulator/inject")
async def inject_anomaly():
    simulator.inject_anomaly(duration=5.0)
    return {"status": "ok", "action": "anomaly injected for 5s"}


@app.post("/simulator/reset")
async def reset_simulator():
    simulator.reset()
    return {"status": "ok", "action": "reset to baseline"}


@app.get("/simulator/status")
async def get_status():
    return simulator.get_state()
