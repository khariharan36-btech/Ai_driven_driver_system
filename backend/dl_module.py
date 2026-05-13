"""
DL MODULE — AI Driver Safety System
====================================
Pipeline:
  Frame → MediaPipe FaceMesh → Feature Extraction → Fatigue Scoring → Risk Output

Features computed:
  - EAR  : Eye Aspect Ratio  (low = drowsy)
  - MAR  : Mouth Aspect Ratio (high = yawning)
  - PER  : Perclos score      (% of time eyes closed over 30-frame window)
  - Head pitch/yaw            (nodding/looking away)
  - Blink rate                (slow = fatigued)
  - Gaze deviation            (distraction)

Outputs per frame:
  {
    fatigue_score   : float  0-100
    distraction_score: float 0-100
    risk_level      : str   "SAFE"|"CAUTION"|"WARNING"|"CRITICAL"
    yawn_detected   : bool
    eye_status      : str   "OPEN"|"CLOSED"|"BLINKING"
    blink_rate      : int   blinks/minute
    yawn_count      : int
    head_pose       : str   "STABLE"|"NODDING"|"TURNED"
    features        : dict  raw computed features
    timestamp       : str
  }
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import math
from collections import deque
from datetime import datetime


# ─── MediaPipe Setup ──────────────────────────────────────────────────────────
mp_face_mesh = mp.solutions.face_mesh
mp_drawing   = mp.solutions.drawing_utils

# MediaPipe FaceMesh landmark indices
# Left eye outer/inner corners and top/bottom lids
LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158, 133, 153, 144]

# Mouth corners and top/bottom lip
MOUTH_OUTER = [61, 291, 39,  181, 0,   17,  269, 405]
MOUTH_INNER = [78, 308, 82,  87,  13,  14,  312, 317]

# Head pose reference points
NOSE_TIP    = 1
CHIN        = 152
LEFT_EYE_L  = 263
RIGHT_EYE_R = 33
LEFT_MOUTH  = 287
RIGHT_MOUTH = 57


# ─── Geometry Helpers ─────────────────────────────────────────────────────────
def _euclidean(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)


def compute_ear(landmarks, eye_indices, w, h):
    """Eye Aspect Ratio: vertical distance / horizontal distance.
    Normal blink ~0.2, sustained closure <0.18 means drowsy."""
    pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in eye_indices]
    # vertical
    v1 = _euclidean(pts[1], pts[5])
    v2 = _euclidean(pts[2], pts[4])
    # horizontal
    hz = _euclidean(pts[0], pts[3])
    return (v1 + v2) / (2.0 * hz + 1e-6)


def compute_mar(landmarks, w, h):
    """Mouth Aspect Ratio: mouth open height / width.
    >0.6 usually indicates a yawn."""
    pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in MOUTH_INNER]
    v1 = _euclidean(pts[1], pts[7])
    v2 = _euclidean(pts[2], pts[6])
    v3 = _euclidean(pts[3], pts[5])
    hz = _euclidean(pts[0], pts[4])
    return (v1 + v2 + v3) / (3.0 * hz + 1e-6)


def compute_head_pose(landmarks, w, h):
    """Estimate head pitch (nod) and yaw (turn) from 2D landmark geometry.
    Returns (pitch_deg, yaw_deg) — rough but fast for real-time use."""
    def pt(idx):
        return (landmarks[idx].x * w, landmarks[idx].y * h, landmarks[idx].z * w)

    nose  = pt(NOSE_TIP)
    chin  = pt(CHIN)
    lmth  = pt(LEFT_MOUTH)
    rmth  = pt(RIGHT_MOUTH)
    leye  = pt(LEFT_EYE_L)
    reye  = pt(RIGHT_EYE_R)

    # Yaw from horizontal eye line asymmetry
    eye_width = abs(leye[0] - reye[0]) + 1e-6
    nose_offset = (nose[0] - reye[0]) / eye_width
    yaw = (nose_offset - 0.5) * 90   # rough degrees

    # Pitch from nose-to-chin vs eye line vertical
    face_height = abs(chin[1] - leye[1]) + 1e-6
    nose_height = abs(nose[1] - leye[1])
    pitch = (nose_height / face_height - 0.4) * 90

    return round(pitch, 1), round(yaw, 1)


# ─── FatigueDetector Class ─────────────────────────────────────────────────────
class FatigueDetector:
    """
    Full real-time fatigue detection pipeline.
    Usage:
        detector = FatigueDetector()
        result   = detector.process_frame(bgr_frame)
    """

    # Thresholds (tuned for typical webcam, 30fps)
    EAR_THRESHOLD      = 0.22    # below this = eye closing
    EAR_CLOSED_THRESH  = 0.18    # below this = eye closed
    MAR_YAWN_THRESH    = 0.55    # above this = yawn
    PERCLOS_WINDOW     = 90      # frames (~3s at 30fps)
    BLINK_WINDOW_SECS  = 60      # for blink-rate calculation
    PITCH_NOD_THRESH   = 15      # degrees, head nod warning
    YAW_TURN_THRESH    = 20      # degrees, head turn warning

    def __init__(self, window_size: int = 90):
        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Rolling windows
        self.ear_window      = deque(maxlen=window_size)
        self.mar_window      = deque(maxlen=window_size)
        self.pitch_window    = deque(maxlen=window_size)
        self.yaw_window      = deque(maxlen=window_size)

        # Blink tracking
        self.blink_timestamps = deque()
        self.eye_was_closed   = False
        self.blink_count_session = 0

        # Yawn tracking
        self.yawn_active  = False
        self.yawn_count   = 0

        # Fatigue smoothing
        self._fatigue_smooth = deque(maxlen=10)
        self._dist_smooth    = deque(maxlen=10)

        # Session start
        self.session_start = time.time()
        self.frame_count   = 0

    # ── Main entry point ────────────────────────────────────────────────────
    def process_frame(self, frame: np.ndarray) -> dict:
        """
        Process one BGR frame.
        Returns a structured result dict ready for the RL module and API.
        """
        self.frame_count += 1
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.face_mesh.process(rgb)

        # Default (no face)
        if not result.multi_face_landmarks:
            return self._no_face_result()

        lm = result.multi_face_landmarks[0].landmark

        # ── Feature extraction ──────────────────────────────────────────────
        left_ear  = compute_ear(lm, LEFT_EYE,  w, h)
        right_ear = compute_ear(lm, RIGHT_EYE, w, h)
        ear       = (left_ear + right_ear) / 2.0

        mar       = compute_mar(lm, w, h)
        pitch, yaw = compute_head_pose(lm, w, h)

        self.ear_window.append(ear)
        self.mar_window.append(mar)
        self.pitch_window.append(pitch)
        self.yaw_window.append(yaw)

        # ── Blink detection ──────────────────────────────────────────────────
        eye_closed = ear < self.EAR_CLOSED_THRESH
        if self.eye_was_closed and not eye_closed:
            # Transition closed→open = one blink
            self.blink_count_session += 1
            self.blink_timestamps.append(time.time())
        self.eye_was_closed = eye_closed

        # Purge blink timestamps older than window
        now = time.time()
        while self.blink_timestamps and now - self.blink_timestamps[0] > self.BLINK_WINDOW_SECS:
            self.blink_timestamps.popleft()
        blink_rate = len(self.blink_timestamps)   # blinks in last 60s

        # ── Yawn detection ───────────────────────────────────────────────────
        yawn_now = mar > self.MAR_YAWN_THRESH
        if yawn_now and not self.yawn_active:
            self.yawn_count += 1
        self.yawn_active = yawn_now

        # ── PERCLOS (% eye closure over window) ─────────────────────────────
        if len(self.ear_window) >= 10:
            closed_frames = sum(1 for e in self.ear_window if e < self.EAR_THRESHOLD)
            perclos = closed_frames / len(self.ear_window)
        else:
            perclos = 0.0

        # ── Eye status ───────────────────────────────────────────────────────
        if eye_closed:
            eye_status = "CLOSED"
        elif ear < self.EAR_THRESHOLD:
            eye_status = "BLINKING"
        else:
            eye_status = "OPEN"

        # ── Head pose status ─────────────────────────────────────────────────
        avg_pitch = np.mean(self.pitch_window) if self.pitch_window else 0
        avg_yaw   = np.mean(self.yaw_window)   if self.yaw_window   else 0

        if abs(avg_pitch) > self.PITCH_NOD_THRESH:
            head_pose = "NODDING"
        elif abs(avg_yaw) > self.YAW_TURN_THRESH:
            head_pose = "TURNED"
        else:
            head_pose = "STABLE"

        # ── Fatigue Score (0–100) ────────────────────────────────────────────
        fatigue_score = self._compute_fatigue_score(
            perclos, ear, blink_rate, yawn_now, avg_pitch, avg_yaw
        )

        # ── Distraction Score (0–100) ─────────────────────────────────────────
        distraction_score = self._compute_distraction_score(avg_yaw, avg_pitch)

        # ── Risk Level ────────────────────────────────────────────────────────
        risk_level = self._get_risk_level(fatigue_score)

        return {
            "face_detected":     True,
            "fatigue_score":     round(fatigue_score, 1),
            "distraction_score": round(distraction_score, 1),
            "risk_level":        risk_level,
            "eye_status":        eye_status,
            "yawn_detected":     yawn_now,
            "yawn_count":        self.yawn_count,
            "blink_rate":        blink_rate,
            "head_pose":         head_pose,
            "features": {
                "ear":           round(ear, 4),
                "mar":           round(mar, 4),
                "perclos":       round(perclos, 4),
                "pitch_deg":     round(avg_pitch, 1),
                "yaw_deg":       round(avg_yaw, 1),
                "left_ear":      round(left_ear, 4),
                "right_ear":     round(right_ear, 4),
            },
            "session_seconds":   round(now - self.session_start, 1),
            "frame_count":       self.frame_count,
            "timestamp":         datetime.utcnow().isoformat()
        }

    # ── Fatigue scoring formula ──────────────────────────────────────────────
    def _compute_fatigue_score(self, perclos, ear, blink_rate, yawn, pitch, yaw):
        """
        Weighted combination of visual fatigue signals.
        Weights are empirically chosen for typical drowsy-driving signatures.
        """
        score = 0.0

        # PERCLOS is the strongest scientific predictor of drowsiness
        score += perclos * 45           # max 45 pts from eye closure pattern

        # EAR deviation from normal (~0.28 open)
        normal_ear = 0.28
        ear_drop   = max(0, normal_ear - ear) / normal_ear
        score += ear_drop * 20          # max 20 pts

        # Slow blink rate (normal ~15-20/min; fatigued <10)
        if blink_rate < 10:
            score += (10 - blink_rate) * 1.5    # up to 15 pts
        elif blink_rate > 25:
            score += (blink_rate - 25) * 0.5    # rapid blinking also adds

        # Yawning
        if yawn:
            score += 12

        # Head nodding (pitch down)
        if abs(pitch) > self.PITCH_NOD_THRESH:
            score += min(15, abs(pitch) - self.PITCH_NOD_THRESH)

        return min(100.0, max(0.0, score))

    def _compute_distraction_score(self, yaw, pitch):
        """
        Distraction = sustained gaze deviation or head turning away.
        """
        score = 0.0
        if abs(yaw) > self.YAW_TURN_THRESH:
            score += min(60, (abs(yaw) - self.YAW_TURN_THRESH) * 2.5)
        if pitch > self.PITCH_NOD_THRESH * 0.8:  # looking down at phone
            score += min(40, (pitch - 10) * 2)
        return min(100.0, max(0.0, score))

    def _get_risk_level(self, score):
        if score < 25:  return "SAFE"
        if score < 50:  return "CAUTION"
        if score < 75:  return "WARNING"
        return "CRITICAL"

    def _no_face_result(self):
        return {
            "face_detected":     False,
            "fatigue_score":     0.0,
            "distraction_score": 100.0,   # no face = driver looking away
            "risk_level":        "WARNING",
            "eye_status":        "UNKNOWN",
            "yawn_detected":     False,
            "yawn_count":        self.yawn_count,
            "blink_rate":        0,
            "head_pose":         "TURNED",
            "features":          {},
            "session_seconds":   round(time.time() - self.session_start, 1),
            "frame_count":       self.frame_count,
            "timestamp":         datetime.utcnow().isoformat()
        }

    # ── Frame annotation helper ──────────────────────────────────────────────
    def annotate_frame(self, frame: np.ndarray, result: dict) -> np.ndarray:
        """Draw detection overlays on the frame for the live video feed."""
        frame = frame.copy()
        risk  = result["risk_level"]

        color_map = {
            "SAFE":     (34,  197, 94),
            "CAUTION":  (251, 191, 36),
            "WARNING":  (249, 115, 22),
            "CRITICAL": (239, 68,  68)
        }
        color = color_map.get(risk, (128, 128, 128))

        h, w = frame.shape[:2]

        # Fatigue score overlay
        cv2.putText(frame, f"FATIGUE: {result['fatigue_score']:.0f}/100",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(frame, f"RISK: {risk}",
                    (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(frame, f"EAR: {result['features'].get('ear', 0):.3f}",
                    (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        cv2.putText(frame, f"BLINK: {result['blink_rate']}/min",
                    (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        if result["yawn_detected"]:
            cv2.putText(frame, "YAWN DETECTED",
                        (w//2 - 90, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (251, 191, 36), 2)

        # Risk border
        border = 3
        cv2.rectangle(frame, (border, border), (w-border, h-border), color, border)
        return frame

    def release(self):
        self.face_mesh.close()


# ─── Quick Test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    detector = FatigueDetector()
    cap      = cv2.VideoCapture(0)       # 0 = default webcam

    print("Starting fatigue detection — press Q to quit\n")

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break

        result = detector.process_frame(frame)
        annotated = detector.annotate_frame(frame, result)
        cv2.imshow("Driver Safety — DL Module", annotated)

        print(
            f"\rFatigue: {result['fatigue_score']:5.1f} | "
            f"Risk: {result['risk_level']:8s} | "
            f"EAR: {result['features'].get('ear',0):.3f} | "
            f"Blink: {result['blink_rate']:2d}/min | "
            f"Yawns: {result['yawn_count']}",
            end=""
        )

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    detector.release()
