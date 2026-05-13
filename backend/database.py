"""
DATABASE MODULE — AI Driver Safety System
==========================================
Schema:
  drivers          — hashed driver profiles
  sessions         — driving sessions
  alert_log        — every alert fired per session
  health_records   — health/sleep data per driver
  incident_records — near-miss / accident records

Analytics views:
  - Alert effectiveness per type
  - Fatigue trend per session
  - Incident correlation with fatigue patterns
  - Driver recovery time statistics

Engine: SQLite (file-based, zero-setup for hackathon)
ORM: raw sqlite3 + dict-based results (no SQLAlchemy dependency)
"""

import sqlite3
import json
import os
import random
import hashlib
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
import statistics

DB_PATH = os.path.join(os.path.dirname(__file__), "data", "driver_safety.db")


# ─── Connection helper ─────────────────────────────────────────────────────────
def get_conn():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row      # dict-like row access
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


# ─── Schema creation ───────────────────────────────────────────────────────────
SCHEMA = """
CREATE TABLE IF NOT EXISTS drivers (
    driver_id       TEXT PRIMARY KEY,   -- hashed ID e.g. DRV-A1B2-C3D4
    masked_id       TEXT NOT NULL,      -- display version DRV-****-C3D4
    created_at      TEXT NOT NULL,
    total_sessions  INTEGER DEFAULT 0,
    avg_fatigue     REAL    DEFAULT 0.0,
    risk_profile    TEXT    DEFAULT 'NORMAL'
);

CREATE TABLE IF NOT EXISTS sessions (
    session_id      TEXT PRIMARY KEY,
    driver_id       TEXT NOT NULL REFERENCES drivers(driver_id),
    start_time      TEXT NOT NULL,
    end_time        TEXT,
    duration_secs   INTEGER,
    max_fatigue     REAL    DEFAULT 0.0,
    avg_fatigue     REAL    DEFAULT 0.0,
    total_alerts    INTEGER DEFAULT 0,
    ignored_alerts  INTEGER DEFAULT 0,
    yawn_count      INTEGER DEFAULT 0,
    near_miss       INTEGER DEFAULT 0,
    status          TEXT    DEFAULT 'ACTIVE',   -- ACTIVE | COMPLETED
    route_label     TEXT
);

CREATE TABLE IF NOT EXISTS alert_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      TEXT NOT NULL REFERENCES sessions(session_id),
    timestamp       TEXT NOT NULL,
    fatigue_score   REAL NOT NULL,
    distraction_score REAL DEFAULT 0.0,
    alert_type      TEXT NOT NULL,
    alert_index     INTEGER NOT NULL,
    was_effective   INTEGER DEFAULT 0,   -- 1 if fatigue decreased after alert
    response_time_secs REAL DEFAULT 0.0, -- seconds until fatigue dropped
    rl_state        TEXT,                -- JSON encoded state tuple
    rl_q_values     TEXT                 -- JSON encoded Q-values
);

CREATE TABLE IF NOT EXISTS health_records (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    driver_id       TEXT NOT NULL REFERENCES drivers(driver_id),
    recorded_at     TEXT NOT NULL,
    avg_fatigue_score REAL DEFAULT 0.0,
    sleep_hours     REAL DEFAULT 7.0,
    sleep_quality   TEXT DEFAULT 'GOOD',  -- POOR | FAIR | GOOD
    stress_level    INTEGER DEFAULT 3,    -- 1-10
    driving_hours_today REAL DEFAULT 0.0,
    prior_incidents INTEGER DEFAULT 0,
    recovery_time_avg REAL DEFAULT 0.0,
    notes           TEXT
);

CREATE TABLE IF NOT EXISTS incident_records (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      TEXT NOT NULL REFERENCES sessions(session_id),
    driver_id       TEXT NOT NULL REFERENCES drivers(driver_id),
    timestamp       TEXT NOT NULL,
    incident_type   TEXT NOT NULL,       -- NEAR_MISS | SUDDEN_BRAKE | COLLISION_MINOR
    fatigue_before  REAL DEFAULT 0.0,
    distraction_before REAL DEFAULT 0.0,
    alerts_before   INTEGER DEFAULT 0,
    alerts_ignored  INTEGER DEFAULT 0,
    severity        TEXT DEFAULT 'LOW',  -- LOW | MEDIUM | HIGH
    location_label  TEXT DEFAULT 'Unknown',
    resolved        INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS fatigue_timeline (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      TEXT NOT NULL,
    timestamp       TEXT NOT NULL,
    fatigue_score   REAL NOT NULL,
    distraction_score REAL DEFAULT 0.0,
    eye_status      TEXT,
    head_pose       TEXT,
    blink_rate      INTEGER DEFAULT 15
);

CREATE INDEX IF NOT EXISTS idx_alert_session   ON alert_log(session_id);
CREATE INDEX IF NOT EXISTS idx_health_driver   ON health_records(driver_id);
CREATE INDEX IF NOT EXISTS idx_incident_driver ON incident_records(driver_id);
CREATE INDEX IF NOT EXISTS idx_timeline_session ON fatigue_timeline(session_id);
"""


def init_db():
    """Create all tables if they don't exist."""
    with get_conn() as conn:
        conn.executescript(SCHEMA)
    print(f"[DB] Database initialised at {DB_PATH}")


# ─── Driver CRUD ───────────────────────────────────────────────────────────────
def create_driver(driver_id: str, masked_id: str) -> dict:
    with get_conn() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO drivers (driver_id, masked_id, created_at) VALUES (?,?,?)",
            (driver_id, masked_id, datetime.utcnow().isoformat())
        )
    return {"driver_id": driver_id, "masked_id": masked_id}


def get_driver(driver_id: str) -> Optional[dict]:
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM drivers WHERE driver_id=?", (driver_id,)).fetchone()
    return dict(row) if row else None


# ─── Session CRUD ──────────────────────────────────────────────────────────────
def create_session(session_id: str, driver_id: str, route_label: str = "Unknown") -> dict:
    now = datetime.utcnow().isoformat()
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO sessions (session_id, driver_id, start_time, route_label, status)
               VALUES (?,?,?,?,?)""",
            (session_id, driver_id, now, route_label, "ACTIVE")
        )
    return {"session_id": session_id, "driver_id": driver_id, "start_time": now}


def end_session(session_id: str, max_fatigue: float, avg_fatigue: float,
                total_alerts: int, ignored_alerts: int, yawn_count: int):
    now = datetime.utcnow().isoformat()
    with get_conn() as conn:
        start = conn.execute(
            "SELECT start_time FROM sessions WHERE session_id=?", (session_id,)
        ).fetchone()
        duration = 0
        if start:
            try:
                delta = datetime.utcnow() - datetime.fromisoformat(start["start_time"])
                duration = int(delta.total_seconds())
            except Exception:
                pass
        conn.execute(
            """UPDATE sessions SET end_time=?, duration_secs=?, max_fatigue=?,
               avg_fatigue=?, total_alerts=?, ignored_alerts=?, yawn_count=?, status='COMPLETED'
               WHERE session_id=?""",
            (now, duration, max_fatigue, avg_fatigue, total_alerts, ignored_alerts,
             yawn_count, session_id)
        )


def get_session(session_id: str) -> Optional[dict]:
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM sessions WHERE session_id=?", (session_id,)).fetchone()
    return dict(row) if row else None


def get_recent_sessions(limit: int = 20) -> List[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM sessions ORDER BY start_time DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


# ─── Alert Log ─────────────────────────────────────────────────────────────────
def log_alert(session_id: str, fatigue_score: float, distraction_score: float,
              alert_type: str, alert_index: int,
              rl_state: tuple = None, rl_q_values: list = None) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            """INSERT INTO alert_log
               (session_id, timestamp, fatigue_score, distraction_score,
                alert_type, alert_index, rl_state, rl_q_values)
               VALUES (?,?,?,?,?,?,?,?)""",
            (session_id, datetime.utcnow().isoformat(),
             fatigue_score, distraction_score, alert_type, alert_index,
             json.dumps(rl_state) if rl_state else None,
             json.dumps(rl_q_values) if rl_q_values else None)
        )
    return cur.lastrowid


def mark_alert_effective(alert_id: int, response_time: float):
    with get_conn() as conn:
        conn.execute(
            "UPDATE alert_log SET was_effective=1, response_time_secs=? WHERE id=?",
            (response_time, alert_id)
        )


def get_session_alerts(session_id: str) -> List[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM alert_log WHERE session_id=? ORDER BY timestamp ASC",
            (session_id,)
        ).fetchall()
    return [dict(r) for r in rows]


# ─── Fatigue Timeline ─────────────────────────────────────────────────────────
def log_fatigue_point(session_id: str, fatigue: float, distraction: float,
                      eye_status: str, head_pose: str, blink_rate: int):
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO fatigue_timeline
               (session_id, timestamp, fatigue_score, distraction_score,
                eye_status, head_pose, blink_rate)
               VALUES (?,?,?,?,?,?,?)""",
            (session_id, datetime.utcnow().isoformat(),
             fatigue, distraction, eye_status, head_pose, blink_rate)
        )


def get_fatigue_timeline(session_id: str) -> List[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM fatigue_timeline WHERE session_id=? ORDER BY timestamp ASC",
            (session_id,)
        ).fetchall()
    return [dict(r) for r in rows]


# ─── Health Records ────────────────────────────────────────────────────────────
def log_health_record(driver_id: str, sleep_hours: float, sleep_quality: str,
                      stress_level: int, driving_hours: float,
                      avg_fatigue: float, notes: str = "") -> int:
    with get_conn() as conn:
        cur = conn.execute(
            """INSERT INTO health_records
               (driver_id, recorded_at, avg_fatigue_score, sleep_hours,
                sleep_quality, stress_level, driving_hours_today, notes)
               VALUES (?,?,?,?,?,?,?,?)""",
            (driver_id, datetime.utcnow().isoformat(), avg_fatigue,
             sleep_hours, sleep_quality, stress_level, driving_hours, notes)
        )
    return cur.lastrowid


# ─── Incident Records ─────────────────────────────────────────────────────────
def log_incident(session_id: str, driver_id: str, incident_type: str,
                 fatigue_before: float, distraction_before: float,
                 alerts_before: int, alerts_ignored: int,
                 severity: str = "LOW", location: str = "Unknown") -> int:
    with get_conn() as conn:
        cur = conn.execute(
            """INSERT INTO incident_records
               (session_id, driver_id, timestamp, incident_type,
                fatigue_before, distraction_before, alerts_before,
                alerts_ignored, severity, location_label)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (session_id, driver_id, datetime.utcnow().isoformat(),
             incident_type, fatigue_before, distraction_before,
             alerts_before, alerts_ignored, severity, location)
        )
    return cur.lastrowid


def get_driver_incidents(driver_id: str) -> List[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM incident_records WHERE driver_id=? ORDER BY timestamp DESC",
            (driver_id,)
        ).fetchall()
    return [dict(r) for r in rows]


# ─── Analytics Queries ─────────────────────────────────────────────────────────
def analytics_alert_effectiveness() -> dict:
    """Which alert types are most effective at reducing fatigue?"""
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT alert_type,
                   COUNT(*) as total,
                   SUM(was_effective) as effective,
                   AVG(response_time_secs) as avg_response_time,
                   AVG(fatigue_score) as avg_fatigue_at_alert
            FROM alert_log
            WHERE alert_index > 0
            GROUP BY alert_type
            ORDER BY alert_index
        """).fetchall()
    result = []
    for r in rows:
        d = dict(r)
        d["effectiveness_rate"] = round((d["effective"] or 0) / max(1, d["total"]), 3)
        result.append(d)
    return {"alert_effectiveness": result}


def analytics_fatigue_patterns() -> dict:
    """Overall fatigue statistics and session-level trends."""
    with get_conn() as conn:
        stats = conn.execute("""
            SELECT
                COUNT(*) as total_sessions,
                AVG(avg_fatigue) as overall_avg_fatigue,
                MAX(max_fatigue) as peak_fatigue_ever,
                AVG(total_alerts) as avg_alerts_per_session,
                SUM(near_miss) as total_near_misses,
                AVG(duration_secs)/60.0 as avg_session_minutes
            FROM sessions WHERE status='COMPLETED'
        """).fetchone()

        high_risk = conn.execute("""
            SELECT COUNT(*) as count FROM sessions
            WHERE max_fatigue >= 75 AND status='COMPLETED'
        """).fetchone()

    return {
        "session_stats":  dict(stats) if stats else {},
        "high_risk_sessions": dict(high_risk)["count"] if high_risk else 0
    }


def analytics_incident_correlation() -> dict:
    """What fatigue levels and ignored alerts precede incidents?"""
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT incident_type, severity,
                   AVG(fatigue_before) as avg_fatigue_before,
                   AVG(alerts_ignored) as avg_ignored_alerts,
                   COUNT(*) as count
            FROM incident_records
            GROUP BY incident_type, severity
        """).fetchall()
    return {"incident_correlations": [dict(r) for r in rows]}


def get_full_analytics_dashboard() -> dict:
    """Single call returning everything the UI analytics panel needs."""
    return {
        "alert_effectiveness":   analytics_alert_effectiveness(),
        "fatigue_patterns":      analytics_fatigue_patterns(),
        "incident_correlation":  analytics_incident_correlation(),
        "recent_sessions":       get_recent_sessions(10)
    }


# ─── Seed Data ────────────────────────────────────────────────────────────────
def seed_sample_data(n_drivers: int = 3, n_sessions: int = 12):
    """
    Insert realistic sample data so the analytics dashboard
    has meaningful content to display from the first demo.
    """
    init_db()
    ALERT_TYPES = ["SOFT_VISUAL","VOICE_WARNING","AUDIO_ALERT","STRONG_ALARM","BREAK_REQUIRED"]
    ALERT_IDX   = [1, 2, 3, 4, 5]
    ROUTES = ["Highway 45N", "City Centre", "Ring Road", "Airport Route", "Industrial Zone"]

    driver_ids = []
    for i in range(n_drivers):
        raw = f"SampleDriver_{i+1:02d}"
        hashed = "DRV-" + hashlib.sha256(raw.encode()).hexdigest()[:4].upper() + \
                 "-" + hashlib.sha256(raw.encode()).hexdigest()[4:8].upper()
        masked = f"DRV-****-{hashed[-4:]}"
        create_driver(hashed, masked)
        driver_ids.append(hashed)

        # Health records
        for d in range(7):
            log_health_record(
                driver_id=hashed,
                sleep_hours=round(random.uniform(4.5, 8.5), 1),
                sleep_quality=random.choice(["POOR","FAIR","GOOD","GOOD"]),
                stress_level=random.randint(2, 8),
                driving_hours=round(random.uniform(1, 9), 1),
                avg_fatigue=round(random.uniform(15, 65), 1),
                notes=random.choice(["Normal commute","Long shift","Night drive","Post-break"])
            )

    session_num = 0
    for driver_id in driver_ids:
        for s in range(n_sessions // n_drivers):
            session_num += 1
            sid_raw = f"session_{session_num}"
            sid = "SES-" + hashlib.sha256(sid_raw.encode()).hexdigest()[:4].upper() + \
                  "-" + hashlib.sha256(sid_raw.encode()).hexdigest()[4:8].upper()

            start = datetime.utcnow() - timedelta(days=random.randint(0, 30),
                                                  hours=random.randint(0, 12))
            duration = random.randint(600, 7200)

            with get_conn() as conn:
                conn.execute(
                    """INSERT OR IGNORE INTO sessions
                       (session_id, driver_id, start_time, end_time, duration_secs,
                        max_fatigue, avg_fatigue, total_alerts, ignored_alerts,
                        yawn_count, near_miss, status, route_label)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (sid, driver_id,
                     start.isoformat(),
                     (start + timedelta(seconds=duration)).isoformat(),
                     duration,
                     round(random.uniform(30, 90), 1),
                     round(random.uniform(15, 60), 1),
                     random.randint(0, 15),
                     random.randint(0, 5),
                     random.randint(0, 8),
                     random.randint(0, 2),
                     "COMPLETED",
                     random.choice(ROUTES))
                )

            # Alert log entries
            n_alerts = random.randint(2, 10)
            for _ in range(n_alerts):
                alert_type = random.choice(ALERT_TYPES)
                alert_idx  = ALERT_IDX[ALERT_TYPES.index(alert_type)]
                effective  = random.random() < 0.65
                with get_conn() as conn:
                    conn.execute(
                        """INSERT INTO alert_log
                           (session_id, timestamp, fatigue_score, distraction_score,
                            alert_type, alert_index, was_effective, response_time_secs)
                           VALUES (?,?,?,?,?,?,?,?)""",
                        (sid,
                         (start + timedelta(seconds=random.randint(30, duration-30))).isoformat(),
                         round(random.uniform(25, 90), 1),
                         round(random.uniform(5, 70), 1),
                         alert_type, alert_idx, int(effective),
                         round(random.uniform(3, 45), 1) if effective else 0.0)
                    )

            # Incident (30% of sessions)
            if random.random() < 0.30:
                f_before = round(random.uniform(55, 90), 1)
                log_incident(
                    session_id=sid, driver_id=driver_id,
                    incident_type=random.choice(["NEAR_MISS","SUDDEN_BRAKE","NEAR_MISS"]),
                    fatigue_before=f_before,
                    distraction_before=round(random.uniform(20, 75), 1),
                    alerts_before=random.randint(1, 5),
                    alerts_ignored=random.randint(0, 3),
                    severity=random.choice(["LOW","MEDIUM","LOW","HIGH"]),
                    location=random.choice(ROUTES)
                )
                with get_conn() as conn:
                    conn.execute(
                        "UPDATE sessions SET near_miss=near_miss+1 WHERE session_id=?", (sid,)
                    )

    print(f"[DB] Seeded {n_drivers} drivers, {n_sessions} sessions with alerts and incidents")


# ─── Quick Test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    init_db()
    seed_sample_data(n_drivers=3, n_sessions=12)

    print("\n=== Analytics Dashboard Data ===")
    dash = get_full_analytics_dashboard()

    print("\nAlert Effectiveness:")
    for a in dash["alert_effectiveness"]["alert_effectiveness"]:
        print(f"  {a['alert_type']:18s} → effectiveness: {a['effectiveness_rate']:.0%}  "
              f"(n={a['total']})")

    print("\nFatigue Patterns:")
    fp = dash["fatigue_patterns"]["session_stats"]
    if fp:
        print(f"  Total sessions   : {fp.get('total_sessions', 0)}")
        print(f"  Avg fatigue      : {fp.get('overall_avg_fatigue', 0):.1f}")
        print(f"  Peak fatigue     : {fp.get('peak_fatigue_ever', 0):.1f}")
        print(f"  Near-misses      : {fp.get('total_near_misses', 0)}")

    print("\nRecent Sessions:")
    for s in dash["recent_sessions"][:3]:
        print(f"  {s['session_id']}  max_fatigue={s['max_fatigue']}  "
              f"alerts={s['total_alerts']}  status={s['status']}")
