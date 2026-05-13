"""
RL MODULE — AI Driver Safety System
=====================================
Algorithm : Tabular Q-Learning with ε-greedy exploration
State space: (fatigue_bucket × distraction_bucket × alert_ignored × yawn_active)
Action space: 6 alert levels
Reward     : shaped to minimize risk and alert spam

State encoding:
  fatigue_bucket     : 0-4  (SAFE / CAUTION / WARNING / HIGH / CRITICAL)
  distraction_bucket : 0-2  (LOW / MEDIUM / HIGH)
  alert_ignored      : 0-1  (False / True)
  yawn_active        : 0-1  (False / True)
  → 5 × 3 × 2 × 2 = 60 states

Actions:
  0 = NO_ALERT
  1 = SOFT_VISUAL
  2 = VOICE_WARNING
  3 = AUDIO_ALERT
  4 = STRONG_ALARM
  5 = BREAK_REQUIRED
"""

import numpy as np
import json
import os
import pickle
import time
from datetime import datetime


# ─── Constants ────────────────────────────────────────────────────────────────
ACTIONS = [
    "NO_ALERT",
    "SOFT_VISUAL",
    "VOICE_WARNING",
    "AUDIO_ALERT",
    "STRONG_ALARM",
    "BREAK_REQUIRED"
]

ACTION_DESCRIPTIONS = {
    "NO_ALERT":       "Driver fully alert — no intervention needed",
    "SOFT_VISUAL":    "Mild fatigue onset — subtle visual nudge shown",
    "VOICE_WARNING":  "Fatigue rising — verbal prompt: 'Please stay alert'",
    "AUDIO_ALERT":    "Elevated risk — audio chime + dashboard alert",
    "STRONG_ALARM":   "Critical fatigue — loud alarm, urgent break prompt",
    "BREAK_REQUIRED": "EMERGENCY — mandatory stop, hazard lights suggested"
}

# State dimensions
N_FATIGUE  = 5    # 0-4
N_DISTRACT = 3    # 0-2
N_IGNORED  = 2    # 0-1
N_YAWN     = 2    # 0-1
N_STATES   = N_FATIGUE * N_DISTRACT * N_IGNORED * N_YAWN
N_ACTIONS  = len(ACTIONS)


# ─── State Encoding ───────────────────────────────────────────────────────────
def encode_state(fatigue_score: float, distraction_score: float,
                 alert_ignored: bool, yawn_active: bool) -> tuple:
    """Convert continuous sensor values into a discrete (hashable) state tuple."""

    # Fatigue buckets: 0-19 | 20-39 | 40-59 | 60-79 | 80-100
    if fatigue_score < 20:   fb = 0
    elif fatigue_score < 40: fb = 1
    elif fatigue_score < 60: fb = 2
    elif fatigue_score < 80: fb = 3
    else:                    fb = 4

    # Distraction buckets: LOW | MEDIUM | HIGH
    if distraction_score < 30:   db = 0
    elif distraction_score < 65: db = 1
    else:                        db = 2

    ib = int(alert_ignored)
    yb = int(yawn_active)

    return (fb, db, ib, yb)


def state_to_index(state: tuple) -> int:
    fb, db, ib, yb = state
    return fb * (N_DISTRACT * N_IGNORED * N_YAWN) + \
           db * (N_IGNORED * N_YAWN) + \
           ib * N_YAWN + yb


# ─── Q-Learning Agent ─────────────────────────────────────────────────────────
class RLAlertAgent:
    """
    Tabular Q-Learning agent for alert selection.

    The agent learns which alert to fire given:
      - Current fatigue level
      - Distraction level
      - Whether the last alert was ignored
      - Whether a yawn is occurring

    Training: run simulate_training() once, then save the Q-table.
    Inference: call select_action(state) → action index.
    """

    def __init__(self,
                 alpha:   float = 0.1,     # learning rate
                 gamma:   float = 0.95,    # discount factor
                 epsilon: float = 0.1,     # exploration rate (inference)
                 q_table_path: str = "models/q_table.pkl"):

        self.alpha   = alpha
        self.gamma   = gamma
        self.epsilon = epsilon
        self.q_table_path = q_table_path

        # Q-table: shape (60, 6) — one row per state, one column per action
        self.Q = np.zeros((N_STATES, N_ACTIONS))

        # Decision history
        self.history = []
        self.last_action   = 0
        self.last_state    = None
        self.alert_ignored = False
        self.consecutive_ignored = 0

        # Try loading a pre-trained table
        self._load_q_table()

    # ── Core Q-Learning update ────────────────────────────────────────────────
    def update(self, state: tuple, action: int, reward: float, next_state: tuple):
        """Bellman equation Q-table update."""
        s  = state_to_index(state)
        s2 = state_to_index(next_state)
        td_target = reward + self.gamma * np.max(self.Q[s2])
        td_error  = td_target - self.Q[s, action]
        self.Q[s, action] += self.alpha * td_error

    # ── Action selection (ε-greedy) ───────────────────────────────────────────
    def select_action(self, fatigue_score: float, distraction_score: float,
                      yawn_active: bool = False) -> dict:
        """
        Choose the best alert action for the current driver state.
        Returns a rich dict for the API and UI.
        """
        state = encode_state(
            fatigue_score, distraction_score,
            self.alert_ignored, yawn_active
        )

        # ε-greedy: mostly exploit, sometimes explore
        if np.random.random() < self.epsilon:
            action = np.random.randint(N_ACTIONS)
        else:
            s = state_to_index(state)
            action = int(np.argmax(self.Q[s]))

        # --- Safety override rules (hard constraints) ---
        # These ensure dangerous situations always get a response
        if fatigue_score >= 80 and action < 4:
            action = 5   # always BREAK_REQUIRED at critical
        elif fatigue_score >= 60 and action < 3:
            action = 4   # at least STRONG_ALARM when high-risk
        elif self.consecutive_ignored >= 3 and action < 2:
            action = min(action + 2, 5)  # escalate if ignored many times

        # Record decision
        decision = {
            "state":              state,
            "state_index":        state_to_index(state),
            "action_index":       action,
            "action_name":        ACTIONS[action],
            "description":        ACTION_DESCRIPTIONS[ACTIONS[action]],
            "severity":           action,            # 0-5 severity level
            "q_values":           self.Q[state_to_index(state)].tolist(),
            "policy_confidence":  self._confidence(state_to_index(state)),
            "fatigue_bucket":     state[0],
            "distraction_bucket": state[1],
            "alert_ignored":      self.alert_ignored,
            "consecutive_ignored":self.consecutive_ignored,
            "timestamp":          datetime.utcnow().isoformat()
        }

        self.last_action = action
        self.last_state  = state
        self.history.append(decision)
        if len(self.history) > 500:
            self.history = self.history[-500:]

        return decision

    def _confidence(self, state_idx: int) -> float:
        """Softmax-based confidence of the chosen action."""
        q = self.Q[state_idx]
        if q.max() == q.min():
            return 1.0 / N_ACTIONS
        exp_q = np.exp(q - q.max())
        return float(exp_q.max() / exp_q.sum())

    # ── Feedback loop ─────────────────────────────────────────────────────────
    def record_driver_response(self, fatigue_after: float, fatigue_before: float):
        """
        Call this after observing the driver's reaction to the last alert.
        Updates Q-table with the real-world reward signal.
        """
        if self.last_state is None:
            return

        reward = self._compute_reward(
            self.last_action, fatigue_before, fatigue_after, self.alert_ignored
        )

        # Next state uses current fatigue
        next_state = encode_state(
            fatigue_after,
            0,          # distraction unknown at this point
            self.alert_ignored,
            False
        )
        self.update(self.last_state, self.last_action, reward, next_state)

        # Update ignored flag based on fatigue change
        if fatigue_after < fatigue_before - 3:
            self.alert_ignored = False
            self.consecutive_ignored = 0
        else:
            if self.last_action > 0:
                self.consecutive_ignored += 1
                self.alert_ignored = True

    def _compute_reward(self, action, f_before, f_after, ignored) -> float:
        """
        Reward function:
          + Recovery reward if fatigue decreased
          + Alert efficiency: prefer minimal effective alert
          - Penalty for false alarm (alert when safe)
          - Penalty for ignored alert
          - Large penalty if high fatigue persists
        """
        reward = 0.0

        delta = f_before - f_after   # positive = improvement

        # Recovery reward
        if delta > 5:
            reward += 3.0 + delta * 0.1

        # False alarm penalty (alert when not needed)
        if f_before < 20 and action > 0:
            reward -= 2.0 * action

        # Severity-appropriate alert reward
        appropriate = self._appropriate_action(f_before)
        if action == appropriate:
            reward += 1.5
        elif abs(action - appropriate) == 1:
            reward += 0.5
        else:
            reward -= abs(action - appropriate) * 0.5

        # Ignored alert penalty
        if ignored and action > 0 and f_after >= f_before:
            reward -= 1.5

        # Critical situation penalty — high risk persisting
        if f_before >= 75 and f_after >= 70:
            reward -= 3.0

        # Annoying unnecessary escalation
        if action > 2 and f_before < 40:
            reward -= 1.0

        return reward

    def _appropriate_action(self, fatigue_score) -> int:
        """Heuristic for what action 'should' be taken at a given fatigue score."""
        if fatigue_score < 20:  return 0
        if fatigue_score < 35:  return 1
        if fatigue_score < 55:  return 2
        if fatigue_score < 70:  return 3
        if fatigue_score < 85:  return 4
        return 5

    # ── Training simulation ───────────────────────────────────────────────────
    def simulate_training(self, episodes: int = 5000):
        """
        Train the agent offline with simulated driver sessions.
        Simulates 4 driver profiles:
          1. Responsive driver (responds to soft alerts)
          2. Resistant driver (ignores first 1-2 alerts)
          3. Very fatigued (fatigue rises fast)
          4. Normal commuter (light fatigue, recovers quickly)
        """
        print(f"Training Q-Learning agent for {episodes} episodes...")
        epsilon_train = 0.3     # higher exploration during training
        original_eps  = self.epsilon
        self.epsilon  = epsilon_train

        for ep in range(episodes):
            # Randomly choose a driver profile
            profile = np.random.choice(['responsive', 'resistant', 'fatigued', 'normal'])

            fatigue     = np.random.uniform(0, 30)
            distraction = np.random.uniform(0, 20)
            ignored     = False
            yawn        = False
            ignored_streak = 0

            for step in range(80):   # ~80 steps per session
                state = encode_state(fatigue, distraction, ignored, yawn)

                # Action selection
                s = state_to_index(state)
                if np.random.random() < self.epsilon:
                    action = np.random.randint(N_ACTIONS)
                else:
                    action = int(np.argmax(self.Q[s]))

                # Safety override
                if fatigue >= 80 and action < 4: action = 5

                # Simulate driver response based on profile
                f_before = fatigue
                if action == 0:
                    fatigue = min(100, fatigue + np.random.uniform(0.3, 1.2))
                elif profile == 'responsive':
                    fatigue = max(0, fatigue - np.random.uniform(3, 10))
                    ignored = False; ignored_streak = 0
                elif profile == 'resistant':
                    if ignored_streak < 2:
                        fatigue = min(100, fatigue + np.random.uniform(0, 0.5))
                        ignored = True; ignored_streak += 1
                    else:
                        fatigue = max(0, fatigue - np.random.uniform(5, 15))
                        ignored = False; ignored_streak = 0
                elif profile == 'fatigued':
                    fatigue = max(0, fatigue - np.random.uniform(1, 5))
                    if np.random.random() < 0.4:
                        ignored = True; ignored_streak += 1
                else:  # normal
                    fatigue = max(0, fatigue - np.random.uniform(4, 12))
                    ignored = False; ignored_streak = 0

                # Fatigue naturally rises over time
                fatigue = min(100, fatigue + np.random.uniform(0.1, 0.8))
                distraction = min(100, max(0, distraction + np.random.uniform(-5, 5)))
                yawn = fatigue > 55 and np.random.random() < 0.15

                reward = self._compute_reward(action, f_before, fatigue, ignored)

                next_state = encode_state(fatigue, distraction, ignored, yawn)
                self.update(state, action, reward, next_state)

            # Decay exploration
            self.epsilon = max(0.05, self.epsilon * 0.9995)

            if (ep + 1) % 500 == 0:
                print(f"  Episode {ep+1}/{episodes} — ε={self.epsilon:.4f}")

        self.epsilon = original_eps
        self.save_q_table()
        print(f"Training complete. Q-table saved to {self.q_table_path}")

    # ── Persistence ───────────────────────────────────────────────────────────
    def save_q_table(self):
        os.makedirs(os.path.dirname(self.q_table_path) or '.', exist_ok=True)
        with open(self.q_table_path, 'wb') as f:
            pickle.dump({"Q": self.Q, "trained_at": datetime.utcnow().isoformat()}, f)

    def _load_q_table(self):
        if os.path.exists(self.q_table_path):
            with open(self.q_table_path, 'rb') as f:
                data = pickle.load(f)
            self.Q = data["Q"]
            print(f"Q-table loaded from {self.q_table_path}")
        else:
            print("No Q-table found — running simulate_training() recommended")

    def get_policy_summary(self) -> dict:
        """Return a human-readable policy summary for each fatigue level."""
        summary = {}
        for fb in range(N_FATIGUE):
            scores = [fb*20+10, fb*20+10, fb*20+10, fb*20+10, fb*20+10, fb*20+10]
            state = (fb, 0, 0, 0)
            s = state_to_index(state)
            best = int(np.argmax(self.Q[s]))
            label = ["SAFE","CAUTION","WARNING","HIGH","CRITICAL"][fb]
            summary[label] = {
                "best_action": ACTIONS[best],
                "q_values":    dict(zip(ACTIONS, self.Q[s].round(3).tolist()))
            }
        return summary


# ─── Quick Test / Training ─────────────────────────────────────────────────────
if __name__ == "__main__":
    agent = RLAlertAgent(q_table_path="models/q_table.pkl")

    print("Training agent...")
    agent.simulate_training(episodes=3000)

    print("\nPolicy Summary:")
    for level, info in agent.get_policy_summary().items():
        print(f"  {level:10s} → {info['best_action']}")

    print("\nSample decisions:")
    for f, d, y in [(10, 5, False), (35, 20, False), (55, 40, True), (78, 30, False), (90, 60, True)]:
        dec = agent.select_action(f, d, y)
        print(f"  Fatigue={f:3d}, Dist={d:3d}, Yawn={str(y):5s} → {dec['action_name']:15s} (conf={dec['policy_confidence']:.2f})")
