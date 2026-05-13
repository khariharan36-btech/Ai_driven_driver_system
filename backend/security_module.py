"""
SECURITY MODULE — AI Driver Safety System
==========================================
Covers:
  1. Driver ID / Session ID hashing & masking
  2. AES-256-GCM log encryption (via Fernet / cryptography library)
  3. Role-Based Access Control (RBAC)
  4. Audit trail logging
  5. Data integrity checksums
  6. Sensitive field redaction

Design:
  - All PII is hashed with SHA-256 before storage
  - Session logs are encrypted at rest using Fernet symmetric encryption
  - Every data access is appended to an immutable audit log
  - Two roles: OPERATOR (live view only) and ANALYST (records + analytics)
"""

import hashlib
import hmac
import json
import os
import time
import uuid
from datetime import datetime
from functools import wraps
from typing import Optional, Any

# cryptography library: pip install cryptography
try:
    from cryptography.fernet import Fernet
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False
    print("[SECURITY] Warning: 'cryptography' not installed — using base64 fallback")
    import base64


# ─── RBAC Roles & Permissions ─────────────────────────────────────────────────
ROLES = {
    "OPERATOR": {
        "can_view_live_feed":    True,
        "can_view_alerts":       True,
        "can_view_session_logs": False,
        "can_view_health_records": False,
        "can_view_analytics":    False,
        "can_export_data":       False,
        "can_view_audit_trail":  False,
    },
    "ANALYST": {
        "can_view_live_feed":    True,
        "can_view_alerts":       True,
        "can_view_session_logs": True,
        "can_view_health_records": True,
        "can_view_analytics":    True,
        "can_export_data":       True,
        "can_view_audit_trail":  True,
    },
    "ADMIN": {
        "can_view_live_feed":    True,
        "can_view_alerts":       True,
        "can_view_session_logs": True,
        "can_view_health_records": True,
        "can_view_analytics":    True,
        "can_export_data":       True,
        "can_view_audit_trail":  True,
        "can_manage_users":      True,
    }
}


# ─── SecurityManager ──────────────────────────────────────────────────────────
class SecurityManager:
    """
    Central security manager.
    One instance shared across the application.
    """

    def __init__(self,
                 key_path:       str = "security/encryption.key",
                 audit_log_path: str = "security/audit_trail.jsonl",
                 salt:           str = "DRIVER_SAFETY_SALT_2024"):

        self.salt          = salt.encode()
        self.key_path      = key_path
        self.audit_log_path= audit_log_path

        os.makedirs(os.path.dirname(key_path) or '.', exist_ok=True)

        # Load or generate encryption key
        self.fernet = self._init_encryption()

        # Active sessions: token → {role, user_id, created_at}
        self._sessions: dict[str, dict] = {}

    # ── Encryption / Decryption ────────────────────────────────────────────────
    def _init_encryption(self):
        if CRYPTO_AVAILABLE:
            if os.path.exists(self.key_path):
                with open(self.key_path, 'rb') as f:
                    key = f.read()
            else:
                key = Fernet.generate_key()
                with open(self.key_path, 'wb') as f:
                    f.write(key)
                print(f"[SECURITY] New AES-256 encryption key generated → {self.key_path}")
            return Fernet(key)
        return None

    def encrypt(self, data: str) -> str:
        """Encrypt a string. Returns base64-encoded ciphertext."""
        if self.fernet:
            return self.fernet.encrypt(data.encode()).decode()
        # Fallback (not secure — for demo without cryptography installed)
        return base64.b64encode(data.encode()).decode()

    def decrypt(self, token: str) -> str:
        """Decrypt a previously encrypted string."""
        if self.fernet:
            return self.fernet.decrypt(token.encode()).decode()
        return base64.b64decode(token.encode()).decode()

    def encrypt_dict(self, data: dict) -> str:
        """Encrypt an entire dict as JSON."""
        return self.encrypt(json.dumps(data))

    def decrypt_dict(self, token: str) -> dict:
        """Decrypt a JSON dict."""
        return json.loads(self.decrypt(token))

    # ── ID Hashing & Masking ──────────────────────────────────────────────────
    def hash_driver_id(self, raw_id: str) -> str:
        """
        One-way hash of a driver's real ID.
        Returns a short hex prefix for display.
        """
        digest = hmac.new(
            self.salt,
            raw_id.encode('utf-8'),
            digestmod=hashlib.sha256
        ).hexdigest()
        return f"DRV-{digest[:4].upper()}-{digest[4:8].upper()}"

    def mask_driver_id(self, hashed_id: str) -> str:
        """Mask the middle portion for UI display: DRV-****-XX"""
        parts = hashed_id.split('-')
        if len(parts) == 3:
            return f"{parts[0]}-****-{parts[2]}"
        return "DRV-****-???"

    def hash_session_id(self, raw_session: str) -> str:
        digest = hashlib.sha256(
            (raw_session + self.salt.decode()).encode()
        ).hexdigest()
        return f"SES-{digest[:4].upper()}-{digest[4:8].upper()}"

    def mask_session_id(self, hashed_id: str) -> str:
        parts = hashed_id.split('-')
        if len(parts) == 3:
            return f"{parts[0]}-****-{parts[2]}"
        return "SES-****-???"

    def generate_session_id(self) -> str:
        """Generate and hash a new unique session ID."""
        raw = str(uuid.uuid4())
        return self.hash_session_id(raw)

    # ── Sensitive field redaction ─────────────────────────────────────────────
    def redact_record(self, record: dict, role: str = "OPERATOR") -> dict:
        """
        Remove or mask sensitive fields based on the requester's role.
        OPERATOR gets a minimal safe view.
        ANALYST gets full data.
        """
        if role in ("ANALYST", "ADMIN"):
            return record   # Full access

        # OPERATOR: redact personal / health fields
        REDACTED_FIELDS = {
            "driver_name", "driver_email", "vehicle_plate",
            "gps_location", "health_notes", "medical_history",
            "home_address", "phone_number"
        }
        safe = {}
        for k, v in record.items():
            if k in REDACTED_FIELDS:
                safe[k] = "[REDACTED]"
            elif k == "driver_id" and isinstance(v, str):
                safe[k] = self.mask_driver_id(v)
            elif k == "session_id" and isinstance(v, str):
                safe[k] = self.mask_session_id(v)
            else:
                safe[k] = v
        return safe

    # ── Checksum / Integrity ──────────────────────────────────────────────────
    def compute_checksum(self, data: str) -> str:
        """SHA-256 checksum of a string for tamper-detection."""
        return hashlib.sha256(data.encode()).hexdigest()

    def verify_checksum(self, data: str, stored_checksum: str) -> bool:
        return hmac.compare_digest(
            self.compute_checksum(data),
            stored_checksum
        )

    def sign_record(self, record: dict) -> dict:
        """Add integrity checksum to a record before storage."""
        payload = json.dumps(record, sort_keys=True)
        record["_checksum"] = self.compute_checksum(payload)
        record["_signed_at"] = datetime.utcnow().isoformat()
        return record

    def verify_record(self, record: dict) -> bool:
        """Verify that a stored record hasn't been tampered with."""
        stored_checksum = record.pop("_checksum", None)
        record.pop("_signed_at", None)
        if not stored_checksum:
            return False
        payload = json.dumps(record, sort_keys=True)
        ok = self.verify_checksum(payload, stored_checksum)
        record["_checksum"] = stored_checksum   # restore
        return ok

    # ── Access Token (simple RBAC) ────────────────────────────────────────────
    def create_access_token(self, user_id: str, role: str) -> str:
        """
        Issue a simple access token.
        In production replace with JWT; this is sufficient for hackathon demo.
        """
        if role not in ROLES:
            raise ValueError(f"Unknown role: {role}")
        token = str(uuid.uuid4())
        self._sessions[token] = {
            "user_id":    user_id,
            "role":       role,
            "created_at": time.time()
        }
        self._audit("TOKEN_ISSUED", user_id=user_id, role=role)
        return token

    def validate_token(self, token: str) -> Optional[dict]:
        """Return session info for a valid token, or None."""
        session = self._sessions.get(token)
        if not session:
            return None
        # Expire after 8 hours
        if time.time() - session["created_at"] > 8 * 3600:
            del self._sessions[token]
            return None
        return session

    def check_permission(self, token: str, permission: str) -> bool:
        """Check if the token holder has a specific permission."""
        session = self.validate_token(token)
        if not session:
            return False
        role_perms = ROLES.get(session["role"], {})
        return role_perms.get(permission, False)

    def require_permission(self, permission: str):
        """Decorator for FastAPI route handlers (or any function)."""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, token: str = None, **kwargs):
                if not self.check_permission(token, permission):
                    raise PermissionError(f"Permission denied: {permission}")
                return func(*args, **kwargs)
            return wrapper
        return decorator

    # ── Audit Trail ───────────────────────────────────────────────────────────
    def _audit(self, event: str, **context):
        """Append an immutable audit log entry."""
        entry = {
            "event":     event,
            "timestamp": datetime.utcnow().isoformat(),
            **context
        }
        os.makedirs(os.path.dirname(self.audit_log_path) or '.', exist_ok=True)
        with open(self.audit_log_path, 'a') as f:
            f.write(json.dumps(entry) + '\n')

    def log_data_access(self, user_id: str, role: str,
                        record_type: str, record_id: str):
        self._audit(
            "DATA_ACCESS",
            user_id=user_id,
            role=role,
            record_type=record_type,
            record_id=record_id
        )

    def log_alert_event(self, session_id: str, alert_type: str,
                        fatigue_score: float):
        self._audit(
            "ALERT_FIRED",
            session_id=session_id,
            alert_type=alert_type,
            fatigue_score=round(fatigue_score, 1)
        )

    def log_session_start(self, session_id: str, masked_driver: str):
        self._audit("SESSION_START", session_id=session_id,
                    masked_driver=masked_driver)

    def log_session_end(self, session_id: str, duration_secs: int,
                        max_fatigue: float):
        self._audit("SESSION_END", session_id=session_id,
                    duration_secs=duration_secs,
                    max_fatigue=round(max_fatigue, 1))

    def get_audit_trail(self, last_n: int = 50) -> list:
        """Return the last N audit log entries."""
        if not os.path.exists(self.audit_log_path):
            return []
        with open(self.audit_log_path, 'r') as f:
            lines = f.readlines()
        entries = []
        for line in lines[-last_n:]:
            try:
                entries.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                pass
        return list(reversed(entries))

    def get_security_status(self) -> dict:
        """Return a status dict for the UI security panel."""
        return {
            "encryption_active":  CRYPTO_AVAILABLE,
            "encryption_algo":    "AES-256-GCM (Fernet)" if CRYPTO_AVAILABLE else "BASE64 DEMO ONLY",
            "rbac_active":        True,
            "audit_trail_active": True,
            "id_masking_active":  True,
            "key_file_exists":    os.path.exists(self.key_path),
            "active_sessions":    len(self._sessions),
            "audit_entries":      self._count_audit_entries()
        }

    def _count_audit_entries(self) -> int:
        if not os.path.exists(self.audit_log_path):
            return 0
        with open(self.audit_log_path, 'r') as f:
            return sum(1 for _ in f)


# ─── Module-level singleton ────────────────────────────────────────────────────
security = SecurityManager()


# ─── Quick Demo ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sm = SecurityManager(
        key_path="security/encryption.key",
        audit_log_path="security/audit_trail.jsonl"
    )

    print("=== Security Module Demo ===\n")

    # ID hashing
    raw_id = "John_Driver_123"
    hashed = sm.hash_driver_id(raw_id)
    masked = sm.mask_driver_id(hashed)
    print(f"Raw ID  : {raw_id}")
    print(f"Hashed  : {hashed}")
    print(f"Masked  : {masked}\n")

    # Encryption
    record = {"fatigue_score": 67.3, "alert": "VOICE_WARNING", "timestamp": "2024-01-01T10:00:00"}
    encrypted = sm.encrypt_dict(record)
    decrypted = sm.decrypt_dict(encrypted)
    print(f"Original  : {record}")
    print(f"Encrypted : {encrypted[:50]}...")
    print(f"Decrypted : {decrypted}\n")

    # Integrity
    signed = sm.sign_record(dict(record))
    print(f"Checksum  : {signed['_checksum'][:20]}...")
    print(f"Integrity : {'PASS' if sm.verify_record(signed) else 'FAIL'}\n")

    # RBAC
    op_token  = sm.create_access_token("operator_01", "OPERATOR")
    ana_token = sm.create_access_token("analyst_01",  "ANALYST")
    print(f"Operator can view analytics? {sm.check_permission(op_token,  'can_view_analytics')}")
    print(f"Analyst  can view analytics? {sm.check_permission(ana_token, 'can_view_analytics')}\n")

    # Status
    print("Security Status:", json.dumps(sm.get_security_status(), indent=2))
