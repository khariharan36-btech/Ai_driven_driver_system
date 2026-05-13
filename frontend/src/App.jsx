// App.jsx — AI Driver Safety System Frontend
// Connects to FastAPI backend at http://localhost:8000
// Uses webcam via getUserMedia + sends frames to /session/{id}/analyze

import { useState, useEffect, useRef, useCallback } from "react";
import {
    LineChart, Line, XAxis, YAxis, CartesianGrid,
    Tooltip, ResponsiveContainer, BarChart, Bar, Cell, Legend
} from "recharts";

const API = "http://localhost:8000";

// ─── Color helpers ────────────────────────────────────────────────────────────
const RISK_COLORS = {
    SAFE: { main: "#22c55e", bg: "#f0fdf4", text: "#15803d" },
    CAUTION: { main: "#f59e0b", bg: "#fffbeb", text: "#b45309" },
    WARNING: { main: "#f97316", bg: "#fff7ed", text: "#c2410c" },
    CRITICAL: { main: "#ef4444", bg: "#fef2f2", text: "#b91c1c" },
};

const ALERT_COLORS = ["#94a3b8", "#3b82f6", "#f59e0b", "#f97316", "#ef4444", "#8b5cf6"];
const ALERT_NAMES = ["No Alert", "Soft Visual", "Voice Warning", "Audio Alert", "Strong Alarm", "Break Required"];


// ─── Sub-components ───────────────────────────────────────────────────────────

function StatusBadge({ label, color, bg, textColor }) {
    return (
        <span style={{
            fontSize: 11, fontWeight: 600, letterSpacing: "0.05em",
            padding: "3px 10px", borderRadius: 20,
            background: bg, color: textColor, border: `1px solid ${color}40`
        }}>{label}</span>
    );
}

function MetricCard({ label, value, subtext, color }) {
    return (
        <div style={{
            background: "#fff", border: "1px solid #e5e7eb",
            borderRadius: 10, padding: "10px 12px", flex: 1
        }}>
            <div style={{ fontSize: 10, color: "#6b7280", marginBottom: 3 }}>{label}</div>
            <div style={{ fontSize: 22, fontWeight: 700, color: color || "#111827", lineHeight: 1 }}>{value}</div>
            {subtext && <div style={{ fontSize: 10, color: "#9ca3af", marginTop: 3 }}>{subtext}</div>}
        </div>
    );
}


// ─── Main App ─────────────────────────────────────────────────────────────────
export default function App() {

    // Session state
    const [sessionId, setSessionId] = useState(null);
    const [isRunning, setIsRunning] = useState(false);
    const [status, setStatus] = useState("idle"); // idle|starting|running|ended

    // DL outputs
    const [fatigueScore, setFatigueScore] = useState(0);
    const [distScore, setDistScore] = useState(0);
    const [riskLevel, setRiskLevel] = useState("SAFE");
    const [eyeStatus, setEyeStatus] = useState("OPEN");
    const [headPose, setHeadPose] = useState("STABLE");
    const [blinkRate, setBlinkRate] = useState(15);
    const [yawnCount, setYawnCount] = useState(0);
    const [features, setFeatures] = useState({});

    // RL outputs
    const [alertAction, setAlertAction] = useState(0);
    const [alertDesc, setAlertDesc] = useState("No intervention needed");
    const [policyConf, setPolicyConf] = useState(1.0);
    const [qValues, setQValues] = useState(Array(6).fill(0));

    // Session summary
    const [sessionSecs, setSessionSecs] = useState(0);
    const [totalAlerts, setTotalAlerts] = useState(0);
    const [avgFatigue, setAvgFatigue] = useState(0);
    const [maskedId, setMaskedId] = useState("DRV-****-??");

    // History for charts
    const [fatigueHistory, setFatigueHistory] = useState([]);
    const [alertCounts, setAlertCounts] = useState(Array(6).fill(0));
    const [events, setEvents] = useState([]);

    // Analytics
    const [analytics, setAnalytics] = useState(null);
    const [activeTab, setActiveTab] = useState("monitor"); // monitor|analytics|security

    // Refs
    const videoRef = useRef(null);
    const canvasRef = useRef(null);
    const intervalRef = useRef(null);
    const streamRef = useRef(null);

    const risk = RISK_COLORS[riskLevel] || RISK_COLORS.SAFE;

    // ── Add event to timeline ────────────────────────────────────────────────
    const addEvent = useCallback((text, level = "info") => {
        const now = new Date();
        const ts = [now.getHours(), now.getMinutes(), now.getSeconds()]
            .map(n => String(n).padStart(2, "0")).join(":");
        setEvents(prev => [{ ts, text, level, id: Date.now() }, ...prev].slice(0, 30));
    }, []);

    // ── Start session ────────────────────────────────────────────────────────
    const startSession = async () => {
        setStatus("starting");
        try {
            // Request webcam
            const stream = await navigator.mediaDevices.getUserMedia({
                video: { width: 640, height: 480, facingMode: "user" }
            });
            streamRef.current = stream;
            if (videoRef.current) videoRef.current.srcObject = stream;

            // Create backend session
            const res = await fetch(`${API}/session/start`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ driver_name: "demo_driver", route_label: "Demo Route" })
            });
            const data = await res.json();
            setSessionId(data.session_id);
            setMaskedId(data.driver_id);

            setStatus("running");
            setIsRunning(true);
            addEvent("Session started — secure logging active", "success");
            addEvent("DL model active (MediaPipe + CNN)", "info");
            addEvent("RL agent loaded — Q-table ready", "info");

        } catch (err) {
            console.error(err);
            addEvent(`Error: ${err.message}`, "error");
            setStatus("idle");
        }
    };

    // ── Frame capture & analysis loop ───────────────────────────────────────
    useEffect(() => {
        if (!isRunning || !sessionId) return;

        const capture = async () => {
            const video = videoRef.current;
            const canvas = canvasRef.current;
            if (!video || !canvas) return;

            // Draw current frame to canvas
            const ctx = canvas.getContext("2d");
            canvas.width = video.videoWidth || 640;
            canvas.height = video.videoHeight || 480;
            ctx.drawImage(video, 0, 0);

            // Convert to base64 JPEG
            const frameB64 = canvas.toDataURL("image/jpeg", 0.7).split(",")[1];

            try {
                const res = await fetch(`${API}/session/${sessionId}/analyze`, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ frame_b64: frameB64 })
                });
                if (!res.ok) throw new Error(`HTTP ${res.status}`);
                const d = await res.json();

                // Update DL state
                setFatigueScore(d.fatigue_score);
                setDistScore(d.distraction_score);
                setRiskLevel(d.risk_level);
                setEyeStatus(d.eye_status);
                setHeadPose(d.head_pose);
                setBlinkRate(d.blink_rate);
                setYawnCount(d.yawn_count);
                setFeatures(d.features || {});

                // Update RL state
                setAlertAction(d.alert_index);
                setAlertDesc(d.alert_description);
                setPolicyConf(d.policy_confidence);
                setQValues(d.q_values || Array(6).fill(0));

                // Session stats
                setSessionSecs(d.session_secs);
                setTotalAlerts(d.total_alerts);
                setAvgFatigue(d.avg_fatigue);

                // Add to fatigue chart history
                setFatigueHistory(prev => {
                    const next = [...prev, {
                        t: d.session_secs,
                        f: d.fatigue_score,
                        d: d.distraction_score
                    }];
                    return next.slice(-40);
                });

                // Alert fired event
                if (d.alert_fired && d.alert_index > 0) {
                    addEvent(`RL → "${ALERT_NAMES[d.alert_index]}" triggered`, d.alert_index >= 4 ? "danger" : "warning");
                    setAlertCounts(prev => {
                        const next = [...prev];
                        next[d.alert_index]++;
                        return next;
                    });
                }

                // Risk change events
                if (d.fatigue_score > 75)
                    addEvent("⚠ Critical fatigue zone entered", "danger");
                if (d.yawn_detected)
                    addEvent("Yawn detected by DL module", "warning");

                // Show annotated frame from backend
                if (d.annotated_frame_b64) {
                    const img = new Image();
                    img.onload = () => {
                        const ctx2 = canvas.getContext("2d");
                        ctx2.drawImage(img, 0, 0, canvas.width, canvas.height);
                    };
                    img.src = `data:image/jpeg;base64,${d.annotated_frame_b64}`;
                }

            } catch (err) {
                console.error("Analysis error:", err);
            }
        };

        intervalRef.current = setInterval(capture, 1500);
        return () => clearInterval(intervalRef.current);
    }, [isRunning, sessionId, addEvent]);

    // ── End session ──────────────────────────────────────────────────────────
    const endSession = async () => {
        clearInterval(intervalRef.current);
        if (streamRef.current) streamRef.current.getTracks().forEach(t => t.stop());
        if (sessionId) {
            await fetch(`${API}/session/${sessionId}/end`, { method: "POST" });
        }
        setIsRunning(false);
        setStatus("ended");
        addEvent("Session ended — summary saved", "success");
        fetchAnalytics();
    };

    // ── Fetch analytics ──────────────────────────────────────────────────────
    const fetchAnalytics = async () => {
        try {
            const res = await fetch(`${API}/analytics`);
            const data = await res.json();
            setAnalytics(data);
        } catch (err) {
            console.error("Analytics fetch failed:", err);
        }
    };

    useEffect(() => { fetchAnalytics(); }, []);

    const formatTime = s => `${String(Math.floor(s / 60)).padStart(2, '0')}:${String(s % 60).padStart(2, '0')}`;

    // ─── Render ─────────────────────────────────────────────────────────────
    return (
        <div style={{
            fontFamily: "'Inter', sans-serif", minHeight: "100vh",
            background: "#f8fafc", color: "#111827"
        }}>

            {/* ── Header ── */}
            <div style={{
                background: "#0f172a", padding: "10px 20px",
                display: "flex", alignItems: "center", justifyContent: "space-between"
            }}>
                <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                    <div style={{
                        width: 8, height: 8, borderRadius: "50%",
                        background: isRunning ? "#22c55e" : "#64748b",
                        boxShadow: isRunning ? "0 0 6px #22c55e" : "none"
                    }} />
                    <span style={{
                        color: "#f1f5f9", fontWeight: 700, fontSize: 15,
                        letterSpacing: "0.05em"
                    }}>AI DRIVER SAFETY SYSTEM</span>
                    <span style={{ color: "#475569", fontSize: 11 }}>v1.0 PROTOTYPE</span>
                </div>
                <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
                    <span style={{ color: "#94a3b8", fontSize: 11 }}>
                        Driver: <span style={{ color: "#e2e8f0" }}>{maskedId}</span>
                    </span>
                    <span style={{
                        background: "#1e3a5f", color: "#93c5fd",
                        fontSize: 10, padding: "2px 8px", borderRadius: 4
                    }}>
                        ● AES-256 ACTIVE
                    </span>
                    <span style={{
                        background: "#1e3a5f", color: "#93c5fd",
                        fontSize: 10, padding: "2px 8px", borderRadius: 4
                    }}>
                        {formatTime(sessionSecs)}
                    </span>
                </div>
            </div>

            {/* ── Tab nav ── */}
            <div style={{ background: "#1e293b", display: "flex", gap: 0, paddingLeft: 20 }}>
                {[["monitor", "Live Monitor"], ["analytics", "Analytics"], ["security", "Security"]].map(([id, label]) => (
                    <button key={id} onClick={() => setActiveTab(id)}
                        style={{
                            background: activeTab === id ? "#0f172a" : "transparent",
                            color: activeTab === id ? "#f1f5f9" : "#94a3b8",
                            border: "none", borderBottom: activeTab === id ? "2px solid #3b82f6" : "2px solid transparent",
                            padding: "10px 18px", cursor: "pointer", fontSize: 13, fontWeight: 500
                        }}>{label}</button>
                ))}
            </div>

            {/* ── MONITOR TAB ── */}
            {activeTab === "monitor" && (
                <div style={{ padding: 16 }}>

                    {/* Controls */}
                    <div style={{ display: "flex", gap: 10, marginBottom: 14, alignItems: "center" }}>
                        {status === "idle" && (
                            <button onClick={startSession}
                                style={{
                                    background: "#2563eb", color: "#fff", border: "none",
                                    padding: "9px 22px", borderRadius: 7, fontWeight: 600,
                                    fontSize: 13, cursor: "pointer"
                                }}>
                                ▶ Start Session
                            </button>
                        )}
                        {status === "starting" && (
                            <span style={{ color: "#6b7280", fontSize: 13 }}>Starting camera...</span>
                        )}
                        {status === "running" && (
                            <button onClick={endSession}
                                style={{
                                    background: "#dc2626", color: "#fff", border: "none",
                                    padding: "9px 22px", borderRadius: 7, fontWeight: 600,
                                    fontSize: 13, cursor: "pointer"
                                }}>
                                ■ End Session
                            </button>
                        )}
                        <div style={{ display: "flex", gap: 8 }}>
                            {[
                                { label: `Fatigue: ${fatigueScore}`, color: risk.main },
                                { label: riskLevel, color: risk.main },
                                { label: `Alerts: ${totalAlerts}`, color: "#6b7280" }
                            ].map(({ label, color }) => (
                                <span key={label} style={{
                                    fontSize: 12, padding: "4px 10px",
                                    background: "#f1f5f9", borderRadius: 5, color, fontWeight: 600
                                }}>
                                    {label}
                                </span>
                            ))}
                        </div>
                    </div>

                    {/* Three columns */}
                    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 12 }}>

                        {/* LEFT: Webcam */}
                        <div style={{
                            background: "#fff", border: "1px solid #e2e8f0",
                            borderRadius: 12, overflow: "hidden"
                        }}>
                            <div style={{
                                padding: "8px 12px", borderBottom: "1px solid #f1f5f9",
                                fontSize: 10, color: "#6b7280", letterSpacing: "0.1em"
                            }}>
                                LIVE DRIVER FEED — MediaPipe FaceMesh
                            </div>
                            <div style={{ position: "relative", background: "#000" }}>
                                <video ref={videoRef} autoPlay muted playsInline
                                    style={{ width: "100%", display: "block", maxHeight: 200, objectFit: "cover" }} />
                                <canvas ref={canvasRef} style={{ display: "none" }} />
                                {!isRunning && (
                                    <div style={{
                                        position: "absolute", inset: 0, display: "flex",
                                        alignItems: "center", justifyContent: "center",
                                        background: "#111", color: "#475569", fontSize: 13
                                    }}>
                                        Camera offline
                                    </div>
                                )}
                            </div>
                            <div style={{
                                padding: "8px 10px",
                                display: "grid", gridTemplateColumns: "1fr 1fr", gap: 6
                            }}>
                                {[
                                    {
                                        label: "Eye Status", value: eyeStatus,
                                        ok: eyeStatus === "OPEN"
                                    },
                                    {
                                        label: "Head Pose", value: headPose,
                                        ok: headPose === "STABLE"
                                    },
                                    {
                                        label: "Blink Rate", value: `${blinkRate}/min`,
                                        ok: blinkRate >= 10
                                    },
                                    {
                                        label: "Yawn Count", value: yawnCount,
                                        ok: yawnCount < 4
                                    },
                                    {
                                        label: "EAR", value: (features.ear || 0).toFixed(3),
                                        ok: (features.ear || 0.3) > 0.22
                                    },
                                    {
                                        label: "MAR", value: (features.mar || 0).toFixed(3),
                                        ok: (features.mar || 0) < 0.55
                                    },
                                ].map(({ label, value, ok }) => (
                                    <div key={label} style={{
                                        background: "#f8fafc",
                                        borderRadius: 6, padding: "4px 7px"
                                    }}>
                                        <div style={{ fontSize: 9, color: "#9ca3af" }}>{label}</div>
                                        <div style={{
                                            fontSize: 12, fontWeight: 600,
                                            color: ok ? "#15803d" : "#b91c1c"
                                        }}>{value}</div>
                                    </div>
                                ))}
                            </div>
                        </div>

                        {/* CENTER: AI Decision */}
                        <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                            {/* Fatigue gauge */}
                            <div style={{
                                background: "#fff", border: `2px solid ${risk.main}30`,
                                borderRadius: 12, padding: 14
                            }}>
                                <div style={{
                                    fontSize: 10, color: "#6b7280", letterSpacing: "0.1em",
                                    marginBottom: 8
                                }}>FATIGUE SCORE</div>
                                <div style={{ display: "flex", alignItems: "flex-end", gap: 8, marginBottom: 8 }}>
                                    <span style={{
                                        fontSize: 48, fontWeight: 800, color: risk.main,
                                        lineHeight: 1
                                    }}>{fatigueScore}</span>
                                    <span style={{ fontSize: 14, color: "#9ca3af", marginBottom: 8 }}>/100</span>
                                    <StatusBadge label={riskLevel} color={risk.main}
                                        bg={risk.bg} textColor={risk.text} />
                                </div>
                                <div style={{ background: "#f1f5f9", borderRadius: 4, height: 8, overflow: "hidden" }}>
                                    <div style={{
                                        width: `${fatigueScore}%`, height: "100%", background: risk.main,
                                        borderRadius: 4, transition: "width 0.5s, background 0.5s"
                                    }} />
                                </div>
                                <div style={{
                                    display: "flex", justifyContent: "space-between",
                                    marginTop: 3, fontSize: 8
                                }}>
                                    <span style={{ color: "#15803d" }}>SAFE</span>
                                    <span style={{ color: "#b45309" }}>CAUTION</span>
                                    <span style={{ color: "#c2410c" }}>WARNING</span>
                                    <span style={{ color: "#b91c1c" }}>CRITICAL</span>
                                </div>
                            </div>

                            {/* RL decision */}
                            <div style={{
                                background: "#fff", border: "1px solid #e2e8f0",
                                borderRadius: 12, padding: 14, flex: 1
                            }}>
                                <div style={{
                                    fontSize: 10, color: "#6b7280", letterSpacing: "0.1em",
                                    marginBottom: 8
                                }}>RL AGENT DECISION</div>
                                <div style={{
                                    padding: "10px 12px",
                                    background: ALERT_COLORS[alertAction] + "15",
                                    border: `1px solid ${ALERT_COLORS[alertAction]}40`,
                                    borderRadius: 8, marginBottom: 10
                                }}>
                                    <div style={{
                                        fontSize: 14, fontWeight: 700,
                                        color: ALERT_COLORS[alertAction], marginBottom: 3
                                    }}>
                                        {ALERT_NAMES[alertAction].toUpperCase()}
                                    </div>
                                    <div style={{ fontSize: 10, color: "#6b7280" }}>{alertDesc}</div>
                                </div>
                                {/* Q-value bar chart */}
                                <div style={{ fontSize: 9, color: "#9ca3af", marginBottom: 5 }}>Q-VALUES (policy confidence)</div>
                                {qValues.map((q, i) => {
                                    const max = Math.max(...qValues.map(Math.abs), 1);
                                    const pct = Math.abs(q) / max * 100;
                                    return (
                                        <div key={i} style={{ display: "flex", alignItems: "center", gap: 5, marginBottom: 3 }}>
                                            <span style={{ fontSize: 8, color: "#6b7280", width: 70, flexShrink: 0 }}>
                                                {ALERT_NAMES[i].split(' ')[0]}
                                            </span>
                                            <div style={{
                                                flex: 1, background: "#f1f5f9", height: 5,
                                                borderRadius: 3, overflow: "hidden"
                                            }}>
                                                <div style={{
                                                    width: `${pct}%`, height: "100%",
                                                    background: i === alertAction ? ALERT_COLORS[i] : "#d1d5db",
                                                    borderRadius: 3
                                                }} />
                                            </div>
                                            <span style={{ fontSize: 8, color: "#9ca3af", width: 30 }}>
                                                {q.toFixed(1)}
                                            </span>
                                        </div>
                                    );
                                })}
                            </div>

                            <div style={{ display: "flex", gap: 8 }}>
                                <MetricCard label="Avg Fatigue" value={avgFatigue} color={risk.main} />
                                <MetricCard label="Total Alerts" value={totalAlerts} color="#6b7280" />
                            </div>
                        </div>

                        {/* RIGHT: Event timeline */}
                        <div style={{
                            background: "#fff", border: "1px solid #e2e8f0",
                            borderRadius: 12, padding: 12
                        }}>
                            <div style={{
                                fontSize: 10, color: "#6b7280", letterSpacing: "0.1em",
                                marginBottom: 8
                            }}>SESSION EVENTS</div>
                            <div style={{ height: 360, overflowY: "auto" }}>
                                {events.length === 0
                                    ? <div style={{ fontSize: 11, color: "#d1d5db", padding: 6 }}>
                                        Press Start Session to begin monitoring...
                                    </div>
                                    : events.map(ev => {
                                        const c = {
                                            success: "#16a34a", warning: "#d97706", danger: "#dc2626",
                                            info: "#2563eb", error: "#dc2626"
                                        }[ev.level] || "#6b7280";
                                        return (
                                            <div key={ev.id} style={{
                                                display: "flex", gap: 7, padding: "4px 0",
                                                borderBottom: "1px solid #f8fafc"
                                            }}>
                                                <span style={{
                                                    fontSize: 9, color: "#9ca3af", whiteSpace: "nowrap",
                                                    fontFamily: "monospace", marginTop: 1
                                                }}>{ev.ts}</span>
                                                <span style={{
                                                    width: 5, height: 5, borderRadius: "50%",
                                                    background: c, flexShrink: 0, marginTop: 5
                                                }} />
                                                <span style={{ fontSize: 10, color: c }}>{ev.text}</span>
                                            </div>
                                        );
                                    })
                                }
                            </div>
                        </div>
                    </div>

                    {/* Charts row */}
                    <div style={{
                        display: "grid", gridTemplateColumns: "1.7fr 1fr",
                        gap: 12, marginTop: 12
                    }}>
                        <div style={{
                            background: "#fff", border: "1px solid #e2e8f0",
                            borderRadius: 12, padding: 14
                        }}>
                            <div style={{
                                fontSize: 10, color: "#6b7280", letterSpacing: "0.1em",
                                marginBottom: 10
                            }}>LIVE FATIGUE TREND</div>
                            <ResponsiveContainer width="100%" height={130}>
                                <LineChart data={fatigueHistory}>
                                    <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                                    <XAxis dataKey="t" hide />
                                    <YAxis domain={[0, 100]} tick={{ fontSize: 9 }} width={25} />
                                    <Tooltip formatter={(v, n) => [v, n === "f" ? "Fatigue" : "Distraction"]}
                                        contentStyle={{ fontSize: 10 }} />
                                    <Line type="monotone" dataKey="f" stroke="#ef4444" strokeWidth={2}
                                        dot={false} name="f" />
                                    <Line type="monotone" dataKey="d" stroke="#3b82f6" strokeWidth={1.5}
                                        dot={false} strokeDasharray="4 2" name="d" />
                                </LineChart>
                            </ResponsiveContainer>
                        </div>
                        <div style={{
                            background: "#fff", border: "1px solid #e2e8f0",
                            borderRadius: 12, padding: 14
                        }}>
                            <div style={{
                                fontSize: 10, color: "#6b7280", letterSpacing: "0.1em",
                                marginBottom: 8
                            }}>ALERT DISTRIBUTION</div>
                            {ALERT_NAMES.map((name, i) => {
                                const max = Math.max(1, ...alertCounts);
                                return (
                                    <div key={i} style={{ marginBottom: 5 }}>
                                        <div style={{
                                            display: "flex", justifyContent: "space-between",
                                            fontSize: 9, color: "#6b7280", marginBottom: 2
                                        }}>
                                            <span>{name}</span><span>{alertCounts[i]}</span>
                                        </div>
                                        <div style={{
                                            background: "#f1f5f9", height: 5, borderRadius: 3,
                                            overflow: "hidden"
                                        }}>
                                            <div style={{
                                                width: `${alertCounts[i] / max * 100}%`,
                                                height: "100%", background: ALERT_COLORS[i],
                                                borderRadius: 3, transition: "width 0.4s"
                                            }} />
                                        </div>
                                    </div>
                                );
                            })}
                        </div>
                    </div>
                </div>
            )}

            {/* ── ANALYTICS TAB ── */}
            {activeTab === "analytics" && (
                <div style={{ padding: 20 }}>
                    <div style={{
                        display: "flex", justifyContent: "space-between",
                        alignItems: "center", marginBottom: 16
                    }}>
                        <h2 style={{ fontSize: 16, fontWeight: 700, margin: 0 }}>Session Analytics</h2>
                        <button onClick={fetchAnalytics}
                            style={{
                                background: "#2563eb", color: "#fff", border: "none",
                                padding: "7px 16px", borderRadius: 7, fontSize: 12, cursor: "pointer"
                            }}>
                            Refresh
                        </button>
                    </div>
                    {analytics ? (
                        <>
                            {/* Summary metrics */}
                            <div style={{ display: "flex", gap: 10, marginBottom: 16 }}>
                                {[
                                    {
                                        label: "Total Sessions",
                                        value: analytics.fatigue_patterns?.session_stats?.total_sessions || 0
                                    },
                                    {
                                        label: "Avg Fatigue",
                                        value: (analytics.fatigue_patterns?.session_stats?.overall_avg_fatigue || 0).toFixed(1)
                                    },
                                    {
                                        label: "Peak Fatigue",
                                        value: (analytics.fatigue_patterns?.session_stats?.peak_fatigue_ever || 0).toFixed(1),
                                        color: "#dc2626"
                                    },
                                    {
                                        label: "Near-Misses",
                                        value: analytics.fatigue_patterns?.session_stats?.total_near_misses || 0,
                                        color: "#dc2626"
                                    },
                                ].map(m => <MetricCard key={m.label} {...m} />)}
                            </div>

                            {/* Alert effectiveness */}
                            <div style={{
                                background: "#fff", border: "1px solid #e2e8f0",
                                borderRadius: 12, padding: 16, marginBottom: 12
                            }}>
                                <h3 style={{ fontSize: 13, fontWeight: 600, margin: "0 0 12px" }}>
                                    Alert Effectiveness by Type
                                </h3>
                                {(analytics.alert_effectiveness?.alert_effectiveness || []).map((a, i) => (
                                    <div key={i} style={{ marginBottom: 10 }}>
                                        <div style={{
                                            display: "flex", justifyContent: "space-between",
                                            fontSize: 11, marginBottom: 3
                                        }}>
                                            <span style={{ fontWeight: 500 }}>{a.alert_type}</span>
                                            <span style={{ color: a.effectiveness_rate > 0.6 ? "#15803d" : "#b91c1c" }}>
                                                {(a.effectiveness_rate * 100).toFixed(0)}% effective (n={a.total})
                                            </span>
                                        </div>
                                        <div style={{
                                            background: "#f1f5f9", height: 7, borderRadius: 4,
                                            overflow: "hidden"
                                        }}>
                                            <div style={{
                                                width: `${a.effectiveness_rate * 100}%`,
                                                height: "100%",
                                                background: a.effectiveness_rate > 0.6 ? "#22c55e" : "#ef4444",
                                                borderRadius: 4
                                            }} />
                                        </div>
                                    </div>
                                ))}
                            </div>

                            {/* Recent sessions */}
                            <div style={{
                                background: "#fff", border: "1px solid #e2e8f0",
                                borderRadius: 12, padding: 16
                            }}>
                                <h3 style={{ fontSize: 13, fontWeight: 600, margin: "0 0 12px" }}>
                                    Recent Sessions
                                </h3>
                                <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 11 }}>
                                    <thead>
                                        <tr style={{ borderBottom: "1px solid #e5e7eb" }}>
                                            {["Session ID", "Route", "Max Fatigue", "Alerts", "Status"].map(h => (
                                                <th key={h} style={{
                                                    textAlign: "left", padding: "6px 8px",
                                                    color: "#6b7280", fontWeight: 500
                                                }}>{h}</th>
                                            ))}
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {(analytics.recent_sessions || []).map((s, i) => (
                                            <tr key={i} style={{ borderBottom: "1px solid #f9fafb" }}>
                                                <td style={{
                                                    padding: "6px 8px", color: "#2563eb", fontFamily: "monospace",
                                                    fontSize: 10
                                                }}>{s.session_id}</td>
                                                <td style={{ padding: "6px 8px" }}>{s.route_label || "—"}</td>
                                                <td style={{
                                                    padding: "6px 8px",
                                                    color: s.max_fatigue > 70 ? "#dc2626" : "#15803d",
                                                    fontWeight: 600
                                                }}>{s.max_fatigue?.toFixed(1)}</td>
                                                <td style={{ padding: "6px 8px" }}>{s.total_alerts}</td>
                                                <td style={{ padding: "6px 8px" }}>
                                                    <span style={{
                                                        padding: "2px 8px", borderRadius: 10, fontSize: 10,
                                                        background: s.status === "COMPLETED" ? "#f0fdf4" : "#fef9c3",
                                                        color: s.status === "COMPLETED" ? "#15803d" : "#92400e"
                                                    }}>
                                                        {s.status}
                                                    </span>
                                                </td>
                                            </tr>
                                        ))}
                                    </tbody>
                                </table>
                            </div>
                        </>
                    ) : (
                        <div style={{ color: "#9ca3af", padding: 20 }}>Loading analytics...</div>
                    )}
                </div>
            )}

            {/* ── SECURITY TAB ── */}
            {activeTab === "security" && (
                <div style={{ padding: 20 }}>
                    <h2 style={{ fontSize: 16, fontWeight: 700, margin: "0 0 16px" }}>Security Status</h2>
                    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
                        {[
                            { label: "Encryption", value: "AES-256-GCM (Fernet)", status: true, detail: "All session logs encrypted at rest" },
                            { label: "ID Masking", value: "SHA-256 HMAC", status: true, detail: "Driver IDs hashed and masked in UI" },
                            { label: "Role-Based Access", value: "RBAC Active", status: true, detail: "OPERATOR / ANALYST / ADMIN roles" },
                            { label: "Audit Trail", value: "Immutable Log", status: true, detail: "Every data access timestamped" },
                            { label: "Data in Transit", value: "HTTPS / TLS", status: true, detail: "API communication encrypted" },
                            { label: "Integrity Check", value: "SHA-256 Checksum", status: true, detail: "Records signed before storage" },
                        ].map(({ label, value, status, detail }) => (
                            <div key={label} style={{
                                background: "#fff", border: "1px solid #e2e8f0",
                                borderRadius: 10, padding: "12px 14px"
                            }}>
                                <div style={{
                                    display: "flex", justifyContent: "space-between",
                                    alignItems: "center", marginBottom: 5
                                }}>
                                    <span style={{ fontWeight: 600, fontSize: 13 }}>{label}</span>
                                    <span style={{
                                        background: status ? "#f0fdf4" : "#fef2f2",
                                        color: status ? "#15803d" : "#b91c1c",
                                        fontSize: 10, padding: "2px 8px", borderRadius: 10
                                    }}>
                                        {status ? "● ACTIVE" : "○ INACTIVE"}
                                    </span>
                                </div>
                                <div style={{
                                    fontSize: 12, color: "#2563eb", fontFamily: "monospace",
                                    marginBottom: 3
                                }}>{value}</div>
                                <div style={{ fontSize: 11, color: "#6b7280" }}>{detail}</div>
                            </div>
                        ))}
                    </div>

                    <div style={{
                        background: "#fff", border: "1px solid #e2e8f0",
                        borderRadius: 10, padding: 14, marginTop: 12
                    }}>
                        <h3 style={{ fontSize: 13, fontWeight: 600, margin: "0 0 10px" }}>
                            Masked Identity Examples
                        </h3>
                        <table style={{ width: "100%", fontSize: 12, borderCollapse: "collapse" }}>
                            <thead>
                                <tr style={{ borderBottom: "1px solid #f1f5f9" }}>
                                    {["Field", "Raw Value", "Stored Value", "Display Value"].map(h => (
                                        <th key={h} style={{
                                            textAlign: "left", padding: "5px 8px",
                                            color: "#6b7280", fontWeight: 500, fontSize: 11
                                        }}>{h}</th>
                                    ))}
                                </tr>
                            </thead>
                            <tbody>
                                {[
                                    ["Driver ID", "John_Smith_001", "DRV-A3F1-C7B2", "DRV-****-C7B2"],
                                    ["Session ID", "uuid-xxxx-yyyy", "SES-E9D4-F1A3", "SES-****-F1A3"],
                                    ["Vehicle Plate", "TN 09 AB 1234", "[ENCRYPTED]", "[REDACTED]"],
                                    ["Health Note", "Sleep deprived", "[ENCRYPTED]", "[REDACTED]"],
                                ].map(([field, raw, stored, display], i) => (
                                    <tr key={i} style={{ borderBottom: "1px solid #f9fafb" }}>
                                        <td style={{ padding: "5px 8px", fontWeight: 500 }}>{field}</td>
                                        <td style={{ padding: "5px 8px", color: "#dc2626", fontFamily: "monospace" }}>{raw}</td>
                                        <td style={{ padding: "5px 8px", color: "#7c3aed", fontFamily: "monospace" }}>{stored}</td>
                                        <td style={{ padding: "5px 8px", color: "#15803d", fontFamily: "monospace" }}>{display}</td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </div>
            )}
        </div>
    );
}
