# ─────────────────────────────────────────────────
# Smart Glasses for the Blind
# Module: Weighted Decision Model
# ─────────────────────────────────────────────────
# Combines probability scores from multiple sensors
# (vision, proximity, audio) and outputs a final
# "safe to move" decision with confidence score.
#
# Usage:
#   from modules.decision_model import calculate_move_safe
#   result = calculate_move_safe(obstacle_prob=0.8, proximity_prob=0.3)

from dataclasses import dataclass, field
from typing import Optional


# ── Sensor weight configuration ──────────────────
# Weights must sum to 1.0
SENSOR_WEIGHTS = {
    'vision':    0.50,   # Object detection (primary)
    'proximity': 0.30,   # Ultrasonic/IR distance sensor
    'audio':     0.20,   # Background audio cues
}

# Thresholds
DANGER_THRESHOLD   = 0.40   # Composite score ≥ this → NOT safe to move
WARNING_THRESHOLD  = 0.25   # Composite score ≥ this → caution advised
MIN_CONFIDENCE     = 0.10   # Below this score sensors are treated as 0


@dataclass
class SensorInput:
    """Probability inputs from each modality (0.0 = clear, 1.0 = blocked)."""
    vision_prob:    float = 0.0   # From detection_test.py  obstacle_probability
    proximity_prob: float = 0.0   # From ultrasonic sensor (0-1 normalised)
    audio_prob:     float = 0.0   # From audio classifier (0-1)


@dataclass
class DecisionResult:
    """Output of the decision model."""
    is_safe:           bool
    composite_score:   float          # Weighted danger score (0-1)
    risk_level:        str            # 'SAFE', 'CAUTION', 'DANGER'
    recommended_action: str
    sensor_scores:     dict = field(default_factory=dict)


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp value to [lo, hi]."""
    return max(lo, min(hi, value))


def calculate_move_safe(
    obstacle_prob:  float = 0.0,
    proximity_prob: float = 0.0,
    audio_prob:     float = 0.0,
) -> DecisionResult:
    """
    Core decision function.

    Parameters
    ----------
    obstacle_prob  : 0–1 probability of obstacle from vision/detection model.
    proximity_prob : 0–1 probability of close object from proximity sensor.
    audio_prob     : 0–1 probability of danger cue from audio model.

    Returns
    -------
    DecisionResult with is_safe flag and composite danger score.
    """
    # Sanitise and clamp inputs
    v = _clamp(float(obstacle_prob))
    p = _clamp(float(proximity_prob))
    a = _clamp(float(audio_prob))

    # Apply minimum confidence filter
    v = 0.0 if v < MIN_CONFIDENCE else v
    p = 0.0 if p < MIN_CONFIDENCE else p
    a = 0.0 if a < MIN_CONFIDENCE else a

    # Weighted composite danger score
    composite = (
        SENSOR_WEIGHTS['vision']    * v +
        SENSOR_WEIGHTS['proximity'] * p +
        SENSOR_WEIGHTS['audio']     * a
    )
    composite = _clamp(composite)

    # Determine risk level
    if composite >= DANGER_THRESHOLD:
        risk_level = 'DANGER'
        is_safe    = False
        action     = 'STOP – obstacle detected, do not proceed'
    elif composite >= WARNING_THRESHOLD:
        risk_level = 'CAUTION'
        is_safe    = False
        action     = 'SLOW DOWN – potential hazard ahead, proceed carefully'
    else:
        risk_level = 'SAFE'
        is_safe    = True
        action     = 'CLEAR – path appears safe to proceed'

    return DecisionResult(
        is_safe=is_safe,
        composite_score=round(composite, 4),
        risk_level=risk_level,
        recommended_action=action,
        sensor_scores={
            'vision':    round(v, 4),
            'proximity': round(p, 4),
            'audio':     round(a, 4),
        },
    )


# ── Unit tests ────────────────────────────────────

def _run_tests():
    """7 test cases covering normal and edge-case scenarios."""
    tests = [
        # (description, kwargs, expected_is_safe, expected_risk)
        ("All clear",
         dict(obstacle_prob=0.0, proximity_prob=0.0, audio_prob=0.0),
         True,  'SAFE'),

        ("Tiny obstacle signal (below MIN_CONFIDENCE)",
         dict(obstacle_prob=0.05, proximity_prob=0.0, audio_prob=0.0),
         True,  'SAFE'),

        ("Low obstacle, clear proximity",
         dict(obstacle_prob=0.25, proximity_prob=0.0, audio_prob=0.0),
         True,  'SAFE'),

        ("Moderate obstacle – caution zone",
         dict(obstacle_prob=0.65, proximity_prob=0.0, audio_prob=0.0),
         False, 'CAUTION'),

        ("High obstacle probability",
         dict(obstacle_prob=0.95, proximity_prob=0.0, audio_prob=0.0),
         False, 'DANGER'),

        ("Proximity sensor triggers alone",
         dict(obstacle_prob=0.0, proximity_prob=0.90, audio_prob=0.0),
         False, 'CAUTION'),

        ("All sensors high – maximum danger",
         dict(obstacle_prob=1.0, proximity_prob=1.0, audio_prob=1.0),
         False, 'DANGER'),
    ]

    print("\n─── Decision Model Test Suite ───────────────────")
    passed = 0
    for i, (desc, kwargs, exp_safe, exp_risk) in enumerate(tests, 1):
        result = calculate_move_safe(**kwargs)
        ok = (result.is_safe == exp_safe) and (result.risk_level == exp_risk)
        status = "PASS ✓" if ok else "FAIL ✗"
        if ok:
            passed += 1
        print(f"  [{i}] {status}  {desc}")
        if not ok:
            print(f"       Expected: is_safe={exp_safe}, risk={exp_risk}")
            print(f"       Got:      is_safe={result.is_safe}, risk={result.risk_level}")

    print(f"\n  {passed}/{len(tests)} tests passed")
    print("─────────────────────────────────────────────────\n")
    return passed == len(tests)


if __name__ == "__main__":
    success = _run_tests()
    if not success:
        raise SystemExit(1)
