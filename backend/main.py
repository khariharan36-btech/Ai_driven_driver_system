"""
FASTAPI BACKEND — AI Driver Safety System
==========================================
Endpoints:
  POST /session/start           — start a new driving session
  POST /session/{id}/analyze    — send a frame, get fatigue + alert back
  POST /session/{id}/end        — end session, save summary
  GET  /session/{id}            — get session details
  GET  /sessions                — list recent sessions (ANALYST token required)
  GET  /analytics               — full analytics dashboard
  GET  /analytics/alerts        — alert effectiveness
  GET  /analytics/incidents     — incident correlation
  GET  /security/status         — security module status
  GET  /security/audit          — audit trail (ANALYST only)
  POST /auth/token              — issue access token
  GET  /health                  — API health check

Run with:
  uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""

import base64
import io
import os
import sys
import time
import uuid
from datetime import datetime
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Header, Depends, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── Local modules ──────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from dl_module     import FatigueDetector
from rl_agent      import RLAlertAgent
from security_module import SecurityManager
from database      import (
    init_db, seed_sample_data, create_driver, create_session, end_session,
    log_alert, log_fatigue_point, log_incident, mark_alert_effective,
    get_session, get_recent_sessions, get_session_alerts, get_fatigue_timeline,
    get_full_analytics_dashboard
)


# ─── App Setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="AI Driver Safety System API",
    description="Deep learning fatigue detection + RL alert system",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Module Initialization ────────────────────────────────────────────────────
print("[INIT] Initializing database...")
init_db()

print("[INIT] Seeding sample data (if empty)...")
try:
    seed_sample_data(n_drivers=3, n_sessions=12)
except Exception:
    pass    # already seeded

print("[INIT] Loading DL module...")
dl_detector = FatigueDetector()

print("[INIT] Loading RL agent...")
rl_agent = RLAlertAgent(q_table_path="models/q_table.pkl")
# Train if no Q-table exists
if not os.path.exists("models/q_table.pkl"):
    print("[INIT] No Q-table found — running quick training...")
    rl_agent.simulate_training(episodes=2000)

print("[INIT] Starting security manager...")
security = SecurityManager(
    key_path="security/encryption.key",
    audit_log_path="security/audit_trail.jsonl"
)

# Active session state (in-memory, per session_id)
_active_sessions: dict[str, dict] = {}

print("[INIT] All modules ready\n")


# ─── Pydantic Models ──────────────────────────────────────────────────────────
class StartSessionRequest(BaseModel):
    driver_name:  str = "demo_driver"
    route_label:  str = "Unknown Route"

class AnalyzeFrameRequest(BaseModel):
    frame_b64: str        # base64-encoded JPEG frame from webcam

class EndSessionRequest(BaseModel):
    session_id: str

class AuthRequest(BaseModel):
    user_id:  str
    password: str = "demo"
    role:     str = "ANALYST"

class IncidentReportRequest(BaseModel):
    session_id:  str
    incident_type: str = "NEAR_MISS"
    severity:    str = "MEDIUM"
    location:    str = "Unknown"


# ─── Auth Dependency ──────────────────────────────────────────────────────────
def get_token(authorization: Optional[str] = Header(None)) -> Optional[str]:
    if not authorization:
        return None
    parts = authorization.split()
    return parts[1] if len(parts) == 2 else None


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/health")
def health_check():
    return {
        "status":    "ok",
        "timestamp": datetime.utcnow().isoformat(),
        "modules": {
            "dl":       "ready",
            "rl":       "ready",
            "security": "ready",
            "database": "ready"
        }
    }


# ── Auth ───────────────────────────────────────────────────────────────────────
@app.post("/auth/token")
def get_access_token(req: AuthRequest):
    """Issue a role-based access token (demo: any password accepted)."""
    token = security.create_access_token(req.user_id, req.role.upper())
    security.log_data_access(req.user_id, req.role, "AUTH", "token_request")
    return {
        "access_token": token,
        "role":         req.role.upper(),
        "expires_in":   "8h"
    }


# ── Session Management ─────────────────────────────────────────────────────────
@app.post("/session/start")
def start_session(req: StartSessionRequest):
    """
    Begin a new driving session.
    Returns session_id and masked driver_id for the UI.
    """
    # Hash / mask driver identity
    driver_id = security.hash_driver_id(req.driver_name)
    masked_id = security.mask_driver_id(driver_id)

    # Register driver if new
    create_driver(driver_id, masked_id)

    # Generate session
    session_id = security.generate_session_id()
    create_session(session_id, driver_id, req.route_label)

    # In-memory session state
    _active_sessions[session_id] = {
        "driver_id":         driver_id,
        "masked_driver":     masked_id,
        "start_time":        time.time(),
        "fatigue_readings":  [],
        "total_alerts":      0,
        "ignored_alerts":    0,
        "yawn_count":        0,
        "last_alert_time":   None,
        "last_alert_idx":    0,
        "last_fatigue":      0.0,
        "pending_alert_id":  None,
    }

    security.log_session_start(session_id, masked_id)

    return {
        "session_id":  session_id,
        "driver_id":   masked_id,
        "started_at":  datetime.utcnow().isoformat(),
        "route_label": req.route_label,
        "message":     "Session started. Secure logging active."
    }


@app.post("/session/{session_id}/analyze")
def analyze_frame(session_id: str, req: AnalyzeFrameRequest):
    """
    Main inference endpoint.
    Accepts a base64-encoded webcam frame.
    Returns DL fatigue result + RL alert decision.

    React frontend calls this every ~1.5 seconds.
    """
    if session_id not in _active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    sess = _active_sessions[session_id]

    # ── Decode frame ──────────────────────────────────────────────────────────
    try:
        img_bytes = base64.b64decode(req.frame_b64)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        frame     = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Could not decode image")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid frame: {e}")

    # ── DL Inference ─────────────────────────────────────────────────────────
    dl_result = dl_detector.process_frame(frame)

    fatigue_score     = dl_result["fatigue_score"]
    distraction_score = dl_result["distraction_score"]
    yawn_detected     = dl_result["yawn_detected"]

    # Update session state
    sess["fatigue_readings"].append(fatigue_score)
    if len(sess["fatigue_readings"]) > 200:
        sess["fatigue_readings"] = sess["fatigue_readings"][-200:]
    if yawn_detected:
        sess["yawn_count"] += 1

    # ── RL Decision ───────────────────────────────────────────────────────────
    decision = rl_agent.select_action(
        fatigue_score, distraction_score, yawn_detected
    )
    alert_action = decision["action_name"]
    alert_index  = decision["action_index"]

    # ── Alert change detection ─────────────────────────────────────────────────
    alert_fired = alert_index != sess["last_alert_idx"] and alert_index > 0

    if alert_fired:
        sess["total_alerts"] += 1
        alert_id = log_alert(
            session_id=session_id,
            fatigue_score=fatigue_score,
            distraction_score=distraction_score,
            alert_type=alert_action,
            alert_index=alert_index,
            rl_state=decision["state"],
            rl_q_values=decision["q_values"]
        )
        sess["pending_alert_id"]  = alert_id
        sess["last_alert_time"]   = time.time()
        sess["last_alert_idx"]    = alert_index
        security.log_alert_event(session_id, alert_action, fatigue_score)

    # ── Driver response feedback to RL ─────────────────────────────────────────
    if sess["last_fatigue"] > 0:
        rl_agent.record_driver_response(fatigue_score, sess["last_fatigue"])

        # Mark previous alert as effective if fatigue dropped
        if (sess["pending_alert_id"] and
                fatigue_score < sess["last_fatigue"] - 4 and
                sess["last_alert_time"]):
            response_time = time.time() - sess["last_alert_time"]
            mark_alert_effective(sess["pending_alert_id"], response_time)
            sess["pending_alert_id"] = None

    sess["last_fatigue"] = fatigue_score

    # ── Log timeline point (every ~5 calls to save DB writes) ────────────────
    if len(sess["fatigue_readings"]) % 5 == 0:
        log_fatigue_point(
            session_id, fatigue_score, distraction_score,
            dl_result["eye_status"], dl_result["head_pose"], dl_result["blink_rate"]
        )

    # ── Build annotated frame for UI (optional, base64) ──────────────────────
    annotated = dl_detector.annotate_frame(frame, dl_result)
    _, buf = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 70])
    annotated_b64 = base64.b64encode(buf).decode()

    return {
        # Deep Learning outputs
        "face_detected":      dl_result["face_detected"],
        "fatigue_score":      fatigue_score,
        "distraction_score":  distraction_score,
        "risk_level":         dl_result["risk_level"],
        "eye_status":         dl_result["eye_status"],
        "yawn_detected":      yawn_detected,
        "yawn_count":         sess["yawn_count"],
        "blink_rate":         dl_result["blink_rate"],
        "head_pose":          dl_result["head_pose"],
        "features":           dl_result["features"],

        # RL outputs
        "alert_action":       alert_action,
        "alert_index":        alert_index,
        "alert_description":  decision["description"],
        "alert_fired":        alert_fired,
        "policy_confidence":  decision["policy_confidence"],
        "q_values":           decision["q_values"],
        "rl_state":           decision["state"],

        # Session summary
        "session_secs":       round(time.time() - sess["start_time"]),
        "total_alerts":       sess["total_alerts"],
        "avg_fatigue":        round(sum(sess["fatigue_readings"]) /
                                    max(1, len(sess["fatigue_readings"])), 1),

        # Security
        "driver_id":          sess["masked_driver"],
        "session_id":         session_id,

        # Annotated frame
        "annotated_frame_b64": annotated_b64,

        "timestamp": datetime.utcnow().isoformat()
    }


@app.post("/session/{session_id}/incident")
def report_incident(session_id: str, req: IncidentReportRequest):
    """Manually log a near-miss or incident during a session."""
    if session_id not in _active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    sess = _active_sessions[session_id]
    incident_id = log_incident(
        session_id=session_id,
        driver_id=sess["driver_id"],
        incident_type=req.incident_type,
        fatigue_before=sess["last_fatigue"],
        distraction_before=0.0,
        alerts_before=sess["total_alerts"],
        alerts_ignored=sess["ignored_alerts"],
        severity=req.severity,
        location=req.location
    )
    return {"incident_id": incident_id, "logged": True}


@app.post("/session/{session_id}/end")
def end_session_route(session_id: str):
    """End and summarize a session."""
    if session_id not in _active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    sess = _active_sessions[session_id]
    readings = sess["fatigue_readings"]
    max_f = max(readings) if readings else 0.0
    avg_f = sum(readings) / max(1, len(readings))

    end_session(
        session_id=session_id,
        max_fatigue=round(max_f, 1),
        avg_fatigue=round(avg_f, 1),
        total_alerts=sess["total_alerts"],
        ignored_alerts=sess["ignored_alerts"],
        yawn_count=sess["yawn_count"]
    )

    duration = round(time.time() - sess["start_time"])
    security.log_session_end(session_id, duration, max_f)
    del _active_sessions[session_id]

    return {
        "session_id":    session_id,
        "duration_secs": duration,
        "max_fatigue":   round(max_f, 1),
        "avg_fatigue":   round(avg_f, 1),
        "total_alerts":  sess["total_alerts"],
        "yawn_count":    sess["yawn_count"],
        "status":        "COMPLETED"
    }


# ── Analytics ─────────────────────────────────────────────────────────────────
@app.get("/analytics")
def get_analytics(token: str = Depends(get_token)):
    if token and not security.check_permission(token, "can_view_analytics"):
        raise HTTPException(status_code=403, detail="Analyst role required")
    return get_full_analytics_dashboard()


@app.get("/sessions")
def list_sessions(limit: int = 20, token: str = Depends(get_token)):
    if token and not security.check_permission(token, "can_view_session_logs"):
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    sessions = get_recent_sessions(limit)
    role = "OPERATOR"
    if token:
        sess_info = security.validate_token(token)
        role = sess_info["role"] if sess_info else "OPERATOR"
    return [security.redact_record(s, role) for s in sessions]


@app.get("/session/{session_id}")
def get_session_detail(session_id: str, token: str = Depends(get_token)):
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    alerts   = get_session_alerts(session_id)
    timeline = get_fatigue_timeline(session_id)
    return {
        "session": session,
        "alerts":  alerts,
        "fatigue_timeline": timeline[:100]
    }


# ── Security ──────────────────────────────────────────────────────────────────
@app.get("/security/status")
def security_status():
    return security.get_security_status()


@app.get("/security/audit")
def audit_trail(last_n: int = 50, token: str = Depends(get_token)):
    if token and not security.check_permission(token, "can_view_audit_trail"):
        raise HTTPException(status_code=403, detail="Admin role required")
    return {"audit_entries": security.get_audit_trail(last_n)}
