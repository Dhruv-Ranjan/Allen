from pathlib import Path

code = r'''
"""
===============================================================================
STUDENT WELLBEING EARLY-WARNING AI
VERSION 4.0 — TRAJECTORY + BEHAVIOURAL CASCADE ENGINE
===============================================================================

WHAT CHANGED FROM V3
--------------------
The original early-warning pipeline is preserved:
    1. Individual historical baselines
    2. Feature-level deviations
    3. Personalized risk
    4. Population anomaly
    5. Multi-signal corroboration
    6. Weekly 0-100 behavioral risk
    7. Rolling risk
    8. Risk velocity
    9. Risk acceleration
   10. First meaningful deviation
   11. Early-warning lead time
   12. Replay data

NEW X-FACTOR LAYER
------------------
   13. Signal onset ordering
   14. Behavioral cascade detection
   15. Cascade depth / duration / velocity
   16. Leading-signal analysis
   17. Temporal signal transition matrix
   18. Trajectory phases
   19. Convergence strength
   20. "Why now?" explanation engine
   21. Rewind-to-first-detection replay metadata
   22. Website-ready JSON bundles

IMPORTANT
---------
This is a behavioral anomaly / early-warning prototype.
It does NOT diagnose stress, depression, self-harm, or any other condition.
It does NOT establish causality. "Leading signal" means temporally earlier
in this model, not a proven cause. Real deployment requires human review,
data minimization, consent/governance, and validation on appropriate data.

WEBSITE INTEGRATION
-------------------
The generated files are deliberately separated into:
    student_risk_cards.csv/json
        -> current student cards / overview page

    early_warning_replay.csv/json
        -> 9-week timeline and rewind animation

    behavioral_cascade_analysis.csv/json
        -> cascade panel / "Why now?" panel

    signal_transition_matrix.csv/json
        -> aggregate signal-flow visualization

    trajectory_events.csv/json
        -> timeline event markers

    signal_heatmap_data.csv
        -> existing heatmap

    website_data.json
        -> ONE bundled JSON object containing everything the frontend needs.

FRONTEND MAPPING
----------------
Do not redesign the existing website. Add fields/components to the
student-detail page:

    current_risk_score
        -> existing risk number

    trajectory_phase
        -> phase badge

    cascade_depth
        -> "6 signals involved"

    cascade_start_week
        -> first event marker

    leading_signal
        -> "Earliest observed signal"

    convergence_week
        -> convergence marker on timeline

    early_warning_window
        -> lead-time card

    why_now
        -> explanation panel

    signal_onset_order
        -> ordered cascade list

    trajectory_events
        -> timeline dots

For replay:
    use early_warning_replay.json and move the UI week-by-week.
    At first_deviation_week, place the "FIRST DETECTION" marker.
    At convergence_week, place the "CONVERGENCE" marker.
    At current_week, place the "CURRENT STATE" marker.

The backend remains Python/CSV/JSON. The website does not need to run
the ML model itself; it can simply consume website_data.json or the
individual JSON files.

INSTALL
-------
pip install numpy pandas scikit-learn

RUN
---
python student_wellbeing_ai_v4.py
"""

from __future__ import annotations

import json
import os
from itertools import combinations
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


# =============================================================================
# CONFIGURATION
# =============================================================================

N_STUDENTS = 50
N_WEEKS = 9
BASELINE_WEEKS = 4
ROLLING_WINDOW = 3

RANDOM_SEED = 42

# First-stage standardized deviation thresholds.
Z_THRESHOLD = 2.0
STRONG_Z_THRESHOLD = 2.5
CONVERGENCE_SIGNAL_COUNT = 2

WATCH_THRESHOLD = 20.0
MODERATE_THRESHOLD = 40.0
HIGH_THRESHOLD = 60.0
VERY_HIGH_THRESHOLD = 80.0

OUTPUT_DIR = "wellbeing_outputs"


# =============================================================================
# FEATURE DEFINITIONS
# =============================================================================

# direction:
#   -1 = lower-than-baseline is concerning
#   +1 = higher-than-baseline is concerning
#
# WEBSITE:
# These labels are the display names the frontend should use.
FEATURE_CONFIG = {
    "class_participation": {
        "label": "Class participation",
        "weight": 0.14,
        "direction": -1,
    },
    "arrival_irregularity": {
        "label": "Arrival irregularity",
        "weight": 0.08,
        "direction": +1,
    },
    "early_departures": {
        "label": "Early departures",
        "weight": 0.08,
        "direction": +1,
    },
    "extracurricular_activity": {
        "label": "Extracurricular activity",
        "weight": 0.10,
        "direction": -1,
    },
    "library_resource_usage": {
        "label": "Library/resource usage",
        "weight": 0.08,
        "direction": -1,
    },
    "lms_session_duration": {
        "label": "LMS session duration",
        "weight": 0.10,
        "direction": -1,
    },
    "academic_help_requests": {
        "label": "Academic-help requests",
        "weight": 0.08,
        "direction": +1,
    },
    "meal_usage": {
        "label": "Meal usage",
        "weight": 0.08,
        "direction": -1,
    },
    "transport_irregularity": {
        "label": "Transport irregularity",
        "weight": 0.04,
        "direction": +1,
    },
    "schedule_changes": {
        "label": "Schedule changes",
        "weight": 0.12,
        "direction": +1,
    },
    "digital_timing_shift": {
        "label": "Digital timing shift",
        "weight": 0.10,
        "direction": +1,
    },
}

FEATURES = list(FEATURE_CONFIG.keys())


# =============================================================================
# BASIC RISK FUNCTIONS
# =============================================================================

def severity_from_score(score: float) -> str:
    if score >= VERY_HIGH_THRESHOLD:
        return "VERY HIGH"
    if score >= HIGH_THRESHOLD:
        return "HIGH"
    if score >= MODERATE_THRESHOLD:
        return "MODERATE"
    if score >= WATCH_THRESHOLD:
        return "WATCH"
    return "LOW"


def z_to_risk(z: float) -> float:
    """
    Convert a one-sided standardized deviation to 0-100.

    This is intentionally transparent rather than a black-box classifier.
    z <= 0 is treated as no concerning deviation for that feature.
    """
    z = max(0.0, float(z))

    # Smooth saturation:
    # z=0 -> 0
    # z=2 -> ~50
    # z=3 -> ~73
    # z=4 -> ~88
    risk = 100.0 * (1.0 - np.exp(-0.35 * z * z))
    return float(np.clip(risk, 0.0, 100.0))


# =============================================================================
# SYNTHETIC DATA GENERATION
# =============================================================================

def generate_synthetic_dataset(
    n_students: int = N_STUDENTS,
    n_weeks: int = N_WEEKS,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:

    rng = np.random.default_rng(seed)
    rows = []

    for student_index in range(1, n_students + 1):

        student_id = f"STU-{student_index:03d}"

        # Student-specific baseline behavior.
        participation_base = np.clip(rng.normal(72, 7), 45, 95)
        arrival_base = np.clip(rng.normal(8.0, 0.15), 7.4, 8.8)
        departure_base = np.clip(rng.normal(0.4, 0.15), 0, 1.5)
        extracurricular_base = np.clip(rng.normal(4.0, 0.8), 1, 6)
        library_base = np.clip(rng.normal(3.0, 0.7), 1, 6)
        lms_base = np.clip(rng.normal(48, 8), 20, 80)
        help_base = np.clip(rng.normal(1.5, 0.5), 0, 4)
        meal_base = np.clip(rng.normal(4.0, 0.5), 2, 5)
        transport_base = np.clip(rng.normal(0.4, 0.15), 0, 1.5)
        schedule_base = np.clip(rng.normal(0.3, 0.1), 0, 1)
        digital_base = np.clip(rng.normal(0.5, 0.15), 0, 1)

        # Synthetic hidden patterns create different temporal trajectories.
        if student_index in [7, 19, 31]:
            pattern = "gradual_escalation"
        elif student_index in [12, 28]:
            pattern = "sudden_change"
        elif student_index in [4, 42]:
            pattern = "academic_engagement_decline"
        elif student_index in [15, 36]:
            pattern = "social_withdrawal"
        elif student_index in [23]:
            pattern = "transport_schedule_shift"
        elif student_index in [9, 47]:
            pattern = "temporary_spike"
        else:
            pattern = "stable"

        for week in range(1, n_weeks + 1):

            participation = participation_base + rng.normal(0, 4)
            arrival = arrival_base + rng.normal(0, 0.08)
            departure = departure_base + rng.normal(0, 0.15)
            extracurricular = extracurricular_base + rng.normal(0, 0.4)
            library = library_base + rng.normal(0, 0.35)
            lms = lms_base + rng.normal(0, 4)
            help_requests = help_base + rng.normal(0, 0.3)
            meal = meal_base + rng.normal(0, 0.25)
            transport = transport_base + rng.normal(0, 0.12)
            schedule = schedule_base + rng.normal(0, 0.08)
            digital = digital_base + rng.normal(0, 0.08)

            # -------------------------------------------------------------
            # Synthetic trajectory injections.
            # -------------------------------------------------------------

            if pattern == "gradual_escalation" and week >= 5:
                intensity = week - 4

                participation -= 4.0 * intensity
                arrival += 0.06 * intensity
                departure += 0.18 * intensity
                extracurricular -= 0.30 * intensity
                lms -= 2.5 * intensity
                meal -= 0.18 * intensity
                schedule += 0.08 * intensity
                digital += 0.06 * intensity

            elif pattern == "sudden_change" and week >= 7:
                participation -= 15
                arrival += 0.35
                departure += 0.7
                extracurricular -= 1.8
                lms -= 15
                meal -= 0.8
                schedule += 0.45
                digital += 0.35

            elif pattern == "academic_engagement_decline" and week >= 6:
                intensity = week - 5

                participation -= 2.0 * intensity
                library -= 0.5 * intensity
                lms -= 5.0 * intensity
                help_requests += 0.7 * intensity
                digital += 0.05 * intensity

            elif pattern == "social_withdrawal" and week >= 5:
                intensity = week - 4

                extracurricular -= 0.75 * intensity
                library -= 0.15 * intensity
                meal -= 0.20 * intensity
                participation -= 1.5 * intensity

            elif pattern == "transport_schedule_shift" and week >= 6:
                intensity = week - 5

                arrival += 0.15 * intensity
                transport += 0.35 * intensity
                schedule += 0.18 * intensity
                departure += 0.1 * intensity

            elif pattern == "temporary_spike" and week in [6, 7]:
                participation -= 14
                arrival += 0.4
                departure += 0.8
                extracurricular -= 1.6
                lms -= 12
                meal -= 0.7
                schedule += 0.5
                digital += 0.4

            rows.append({
                "student_id": student_id,
                "week": week,
                "class_participation": np.clip(participation, 0, 100),
                "arrival_irregularity": np.clip(arrival, 6.5, 10),
                "early_departures": np.clip(departure, 0, 5),
                "extracurricular_activity": np.clip(extracurricular, 0, 10),
                "library_resource_usage": np.clip(library, 0, 10),
                "lms_session_duration": np.clip(lms, 0, 120),
                "academic_help_requests": np.clip(help_requests, 0, 10),
                "meal_usage": np.clip(meal, 0, 7),
                "transport_irregularity": np.clip(transport, 0, 5),
                "schedule_changes": np.clip(schedule, 0, 5),
                "digital_timing_shift": np.clip(digital, 0, 5),
                "synthetic_pattern": pattern,
            })

    return pd.DataFrame(rows)


# =============================================================================
# PERSONAL BASELINES
# =============================================================================

def build_baselines(
    raw_df: pd.DataFrame,
    baseline_weeks: int = BASELINE_WEEKS,
) -> pd.DataFrame:

    baseline = raw_df[
        raw_df["week"] <= baseline_weeks
    ].copy()

    rows = []

    for student_id, group in baseline.groupby("student_id"):
        row = {"student_id": student_id}

        for feature in FEATURES:
            values = group[feature].astype(float)

            mean = float(values.mean())
            std = float(values.std(ddof=0))

            # Prevent division by an extremely small standard deviation.
            if std < 1e-6:
                std = max(abs(mean) * 0.05, 0.05)

            row[f"{feature}_mean"] = mean
            row[f"{feature}_std"] = std

        rows.append(row)

    return pd.DataFrame(rows)


# =============================================================================
# FEATURE RISK ENGINE
# =============================================================================

def calculate_feature_risks(
    raw_df: pd.DataFrame,
    baselines: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    df = raw_df.copy()
    baseline_lookup = baselines.set_index("student_id")

    for feature in FEATURES:

        direction = FEATURE_CONFIG[feature]["direction"]

        means = df["student_id"].map(
            baseline_lookup[f"{feature}_mean"]
        )

        stds = df["student_id"].map(
            baseline_lookup[f"{feature}_std"]
        )

        z = direction * (
            (df[feature].astype(float) - means) / stds
        )

        df[f"{feature}_z"] = z.astype(float)
        df[f"{feature}_risk"] = z.map(z_to_risk)

    df["meaningful_signals"] = df.apply(
        lambda row: [
            feature
            for feature in FEATURES
            if row[f"{feature}_z"] >= Z_THRESHOLD
        ],
        axis=1,
    )

    df["strong_signals"] = df.apply(
        lambda row: [
            feature
            for feature in FEATURES
            if row[f"{feature}_z"] >= STRONG_Z_THRESHOLD
        ],
        axis=1,
    )

    df["meaningful_signal_count"] = (
        df["meaningful_signals"].str.len()
    )

    df["strong_signal_count"] = (
        df["strong_signals"].str.len()
    )

    # Weighted personalized risk.
    weighted = np.zeros(len(df), dtype=float)

    for feature in FEATURES:
        weighted += (
            FEATURE_CONFIG[feature]["weight"]
            * df[f"{feature}_risk"].to_numpy()
        )

    df["personalized_risk"] = np.clip(
        weighted,
        0,
        100,
    )

    return df, df.copy()


# =============================================================================
# POPULATION ANOMALY
# =============================================================================

def calculate_population_anomaly(
    raw_df: pd.DataFrame,
    risk_df: pd.DataFrame,
) -> pd.DataFrame:

    result = risk_df.copy()

    anomaly_values = []

    for week, week_raw in raw_df.groupby("week"):

        indices = week_raw.index

        X = week_raw[FEATURES].astype(float).to_numpy()

        if len(X) >= 10:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            model = IsolationForest(
                n_estimators=200,
                contamination="auto",
                random_state=RANDOM_SEED + int(week),
            )

            model.fit(X_scaled)

            raw_anomaly = -model.score_samples(X_scaled)

            lo = float(raw_anomaly.min())
            hi = float(raw_anomaly.max())

            if hi - lo < 1e-9:
                normalized = np.zeros_like(raw_anomaly)
            else:
                normalized = (
                    (raw_anomaly - lo)
                    / (hi - lo)
                    * 100.0
                )

            anomaly_map = dict(
                zip(indices, normalized)
            )

        else:
            anomaly_map = {i: 0.0 for i in indices}

        for index in indices:
            anomaly_values.append(
                (
                    index,
                    float(
                        np.clip(
                            anomaly_map.get(index, 0.0),
                            0,
                            100,
                        )
                    ),
                )
            )

    anomaly_lookup = dict(anomaly_values)

    result["population_anomaly"] = [
        anomaly_lookup.get(i, 0.0)
        for i in result.index
    ]

    return result


# =============================================================================
# FINAL WEEKLY RISK
# =============================================================================

def calculate_final_weekly_risk(
    risk_df: pd.DataFrame,
) -> pd.DataFrame:

    df = risk_df.copy()

    # Multi-signal corroboration:
    # two meaningful signals = 100 corroboration.
    df["corroboration"] = np.clip(
        (
            df["meaningful_signal_count"]
            / CONVERGENCE_SIGNAL_COUNT
        )
        * 100.0,
        0,
        100,
    )

    df["weekly_risk"] = (
        0.75 * df["personalized_risk"]
        + 0.15 * df["population_anomaly"]
        + 0.10 * df["corroboration"]
    ).clip(0, 100)

    df["severity"] = df["weekly_risk"].map(
        severity_from_score
    )

    return df


# =============================================================================
# TRAJECTORY ENGINE
# =============================================================================

def calculate_linear_slope(
    x: np.ndarray,
    y: np.ndarray,
) -> float:

    if len(x) < 2:
        return 0.0

    return float(np.polyfit(x, y, 1)[0])


def trajectory_state(
    rolling_risk: float,
    velocity: float,
    acceleration: float,
) -> str:

    if (
        rolling_risk >= HIGH_THRESHOLD
        and velocity >= 8
    ):
        return "RAPIDLY ESCALATING"

    if (
        rolling_risk >= MODERATE_THRESHOLD
        and velocity >= 5
    ):
        return "ESCALATING"

    if (
        velocity >= 3
        or (
            acceleration >= 2
            and rolling_risk >= WATCH_THRESHOLD
        )
    ):
        return "EMERGING"

    return "STABLE"


def calculate_trajectory_for_student(
    group: pd.DataFrame,
) -> pd.DataFrame:

    group = group.sort_values("week").copy()

    risks = group["weekly_risk"].to_numpy(dtype=float)
    weeks = group["week"].to_numpy(dtype=float)

    rolling_risks = []
    velocities = []
    accelerations = []

    previous_velocity = 0.0

    for i in range(len(group)):

        start = max(
            0,
            i - ROLLING_WINDOW + 1,
        )

        local_weeks = weeks[start:i + 1]
        local_risks = risks[start:i + 1]

        rolling = float(local_risks.mean())

        velocity = calculate_linear_slope(
            local_weeks,
            local_risks,
        )

        acceleration = (
            velocity - previous_velocity
        )

        rolling_risks.append(rolling)
        velocities.append(velocity)
        accelerations.append(acceleration)

        previous_velocity = velocity

    group["rolling_risk_3w"] = rolling_risks
    group["risk_velocity"] = velocities
    group["risk_acceleration"] = accelerations

    group["trajectory_state"] = [
        trajectory_state(
            rolling_risk=rolling_risks[i],
            velocity=velocities[i],
            acceleration=accelerations[i],
        )
        for i in range(len(group))
    ]

    return group


def add_trajectory_engine(
    risk_df: pd.DataFrame,
) -> pd.DataFrame:

    pieces = []

    for _, group in risk_df.groupby("student_id"):
        pieces.append(
            calculate_trajectory_for_student(group)
        )

    return (
        pd.concat(pieces, ignore_index=True)
        .sort_values(["student_id", "week"])
        .reset_index(drop=True)
    )


# =============================================================================
# FIRST MEANINGFUL DEVIATION
# =============================================================================

def detect_first_meaningful_deviation(
    group: pd.DataFrame,
    baseline_weeks: int = BASELINE_WEEKS,
) -> Tuple[int, str, str]:

    group = group.sort_values("week").copy()

    candidate = group[
        group["week"] > baseline_weeks
    ].copy()

    if candidate.empty:
        return -1, "NONE", "NONE"

    previous_strong = set()

    for _, row in candidate.iterrows():

        week = int(row["week"])

        signals = [
            feature
            for feature in FEATURES
            if row[f"{feature}_z"] >= Z_THRESHOLD
        ]

        strong = [
            feature
            for feature in FEATURES
            if row[f"{feature}_z"] >= STRONG_Z_THRESHOLD
        ]

        # First meaningful convergence.
        if len(signals) >= CONVERGENCE_SIGNAL_COUNT:

            strongest = max(
                signals,
                key=lambda f: row[f"{f}_z"],
            )

            return (
                week,
                strongest,
                "MULTI_SIGNAL_CONVERGENCE",
            )

        # Persistent single strong signal.
        persistent = set(strong) & previous_strong

        if persistent:

            strongest = max(
                persistent,
                key=lambda f: row[f"{f}_z"],
            )

            return (
                week - 1,
                strongest,
                "PERSISTENT_SINGLE_SIGNAL",
            )

        previous_strong = set(strong)

    return -1, "NONE", "NONE"


# =============================================================================
# NEW: TRAJECTORY PHASE ENGINE
# =============================================================================

def phase_from_history(
    row: pd.Series,
    group: pd.DataFrame,
    first_deviation_week: int,
    convergence_week: int,
) -> str:

    week = int(row["week"])
    risk = float(row["weekly_risk"])
    velocity = float(row["risk_velocity"])

    if week <= BASELINE_WEEKS:
        return "BASELINE"

    if first_deviation_week == -1:
        if velocity >= 3:
            return "DRIFT"
        return "STABLE"

    if week < first_deviation_week:
        return "BASELINE"

    if (
        convergence_week > 0
        and week >= convergence_week
    ):
        if risk >= HIGH_THRESHOLD or velocity >= 5:
            return "ESCALATION"
        return "CONVERGENCE"

    if velocity >= 3 or risk >= WATCH_THRESHOLD:
        return "EMERGENCE"

    return "DRIFT"


# =============================================================================
# NEW: SIGNAL ONSET + CASCADE ENGINE
# =============================================================================

def get_signal_onsets(
    group: pd.DataFrame,
) -> Dict[str, int]:

    group = group.sort_values("week")
    onsets = {}

    for feature in FEATURES:

        meaningful = group[
            group[f"{feature}_z"] >= Z_THRESHOLD
        ]

        if meaningful.empty:
            onsets[feature] = -1
        else:
            onsets[feature] = int(
                meaningful.iloc[0]["week"]
            )

    return onsets


def calculate_cascade_metrics(
    group: pd.DataFrame,
) -> Dict:

    group = group.sort_values("week").copy()

    onsets = get_signal_onsets(group)

    valid = {
        feature: week
        for feature, week in onsets.items()
        if week > BASELINE_WEEKS
    }

    if not valid:
        return {
            "cascade_start_week": -1,
            "cascade_end_week": -1,
            "cascade_duration": 0,
            "cascade_depth": 0,
            "leading_signal": "NONE",
            "second_signal": "NONE",
            "signal_onset_order": [],
            "new_signals_per_week": 0.0,
            "convergence_week": -1,
            "convergence_strength": 0.0,
            "cascade_velocity": 0.0,
        }

    ordered = sorted(
        valid.items(),
        key=lambda x: (x[1], x[0]),
    )

    cascade_start = min(valid.values())
    cascade_end = max(valid.values())

    duration = cascade_end - cascade_start + 1
    depth = len(ordered)

    # Leading signals are the earliest temporally meaningful signals.
    leading_signal = ordered[0][0]

    second_signal = (
        ordered[1][0]
        if len(ordered) >= 2
        else "NONE"
    )

    # Human-readable ordered representation for the website.
    signal_onset_order = [
        {
            "signal": feature,
            "label": FEATURE_CONFIG[feature]["label"],
            "week": int(week),
        }
        for feature, week in ordered
    ]

    # Find first week where at least two independent signals
    # are meaningful simultaneously.
    convergence_week = -1
    convergence_strength = 0.0

    for week, row in group.iterrows():

        current_week = int(row["week"])

        if current_week <= BASELINE_WEEKS:
            continue

        active = [
            feature
            for feature in FEATURES
            if row[f"{feature}_z"] >= Z_THRESHOLD
        ]

        if len(active) >= CONVERGENCE_SIGNAL_COUNT:

            convergence_week = current_week

            # More simultaneous signals = stronger convergence.
            convergence_strength = float(
                np.clip(
                    len(active)
                    / len(FEATURES)
                    * 100,
                    0,
                    100,
                )
            )

            break

    # Number of newly appearing meaningful signals per week.
    if duration > 0:
        cascade_velocity = (
            depth / duration
        )
    else:
        cascade_velocity = 0.0

    return {
        "cascade_start_week": int(cascade_start),
        "cascade_end_week": int(cascade_end),
        "cascade_duration": int(duration),
        "cascade_depth": int(depth),
        "leading_signal": leading_signal,
        "second_signal": second_signal,
        "signal_onset_order": signal_onset_order,
        "new_signals_per_week": round(
            cascade_velocity,
            3,
        ),
        "convergence_week": int(convergence_week),
        "convergence_strength": round(
            convergence_strength,
            2,
        ),
        "cascade_velocity": round(
            cascade_velocity,
            3,
        ),
    }


def add_cascade_engine(
    trajectory_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    result = trajectory_df.copy()
    cascade_rows = []

    for student_id, group in result.groupby("student_id"):

        group = group.sort_values("week").copy()

        first_week, first_signal, detection_type = (
            detect_first_meaningful_deviation(group)
        )

        metrics = calculate_cascade_metrics(group)

        convergence_week = metrics[
            "convergence_week"
        ]

        # Assign a phase to every weekly record.
        indices = group.index

        for index in indices:

            phase = phase_from_history(
                result.loc[index],
                group,
                first_week,
                convergence_week,
            )

            result.loc[
                index,
                "trajectory_phase"
            ] = phase

            result.loc[
                index,
                "cascade_active"
            ] = (
                metrics["cascade_start_week"] > 0
                and int(result.loc[index, "week"])
                >= metrics["cascade_start_week"]
            )

            result.loc[
                index,
                "cascade_depth"
            ] = metrics["cascade_depth"]

            result.loc[
                index,
                "cascade_start_week"
            ] = metrics["cascade_start_week"]

            result.loc[
                index,
                "convergence_week"
            ] = convergence_week

            result.loc[
                index,
                "convergence_strength"
            ] = metrics["convergence_strength"]

        cascade_rows.append({
            "student_id": student_id,
            **metrics,
            "first_deviation_week": first_week,
            "earliest_signal": first_signal,
            "deviation_detection_type": detection_type,
            "early_warning_window": (
                max(0, int(group["week"].max()) - first_week)
                if first_week > 0
                else 0
            ),
        })

    cascade_df = pd.DataFrame(cascade_rows)

    result["cascade_depth"] = (
        pd.to_numeric(
            result["cascade_depth"],
            errors="coerce",
        )
        .fillna(0)
        .astype(int)
    )

    result["cascade_start_week"] = (
        pd.to_numeric(
            result["cascade_start_week"],
            errors="coerce",
        )
        .fillna(-1)
        .astype(int)
    )

    result["convergence_week"] = (
        pd.to_numeric(
            result["convergence_week"],
            errors="coerce",
        )
        .fillna(-1)
        .astype(int)
    )

    result["cascade_active"] = (
        result["cascade_active"]
        .fillna(False)
        .astype(bool)
    )

    return result, cascade_df


# =============================================================================
# NEW: TEMPORAL SIGNAL TRANSITION MATRIX
# =============================================================================

def build_signal_transition_matrix(
    trajectory_df: pd.DataFrame,
) -> pd.DataFrame:

    transition_records = []

    for student_id, group in trajectory_df.groupby(
        "student_id"
    ):

        group = group.sort_values("week")

        onsets = get_signal_onsets(group)

        valid = [
            (feature, week)
            for feature, week in onsets.items()
            if week > BASELINE_WEEKS
        ]

        # Every ordered pair where signal A appears before B.
        for (a, week_a), (b, week_b) in combinations(
            sorted(valid, key=lambda x: x[1]),
            2,
        ):

            if week_b >= week_a:

                transition_records.append({
                    "student_id": student_id,
                    "from_signal": a,
                    "from_label": FEATURE_CONFIG[a]["label"],
                    "to_signal": b,
                    "to_label": FEATURE_CONFIG[b]["label"],
                    "from_week": week_a,
                    "to_week": week_b,
                    "lag_weeks": week_b - week_a,
                })

    if not transition_records:
        return pd.DataFrame(columns=[
            "from_signal",
            "from_label",
            "to_signal",
            "to_label",
            "observed_student_count",
            "mean_lag_weeks",
            "median_lag_weeks",
            "min_lag_weeks",
            "max_lag_weeks",
        ])

    transitions = pd.DataFrame(
        transition_records
    )

    matrix = (
        transitions.groupby(
            [
                "from_signal",
                "from_label",
                "to_signal",
                "to_label",
            ],
            as_index=False,
        )
        .agg(
            observed_student_count=(
                "student_id",
                "nunique",
            ),
            mean_lag_weeks=(
                "lag_weeks",
                "mean",
            ),
            median_lag_weeks=(
                "lag_weeks",
                "median",
            ),
            min_lag_weeks=(
                "lag_weeks",
                "min",
            ),
            max_lag_weeks=(
                "lag_weeks",
                "max",
            ),
        )
    )

    return matrix.sort_values(
        [
            "observed_student_count",
            "mean_lag_weeks",
        ],
        ascending=[False, True],
    ).reset_index(drop=True)


# =============================================================================
# NEW: WHY-NOW ENGINE
# =============================================================================

def build_why_now(
    row: pd.Series,
    cascade: Dict,
) -> str:

    current_signals = [
        feature
        for feature in FEATURES
        if row[f"{feature}_z"] >= Z_THRESHOLD
    ]

    current_signals = sorted(
        current_signals,
        key=lambda f: row[f"{f}_z"],
        reverse=True,
    )

    if not current_signals:
        return (
            "No major independent behavioral deviations "
            "are currently detected."
        )

    labels = [
        FEATURE_CONFIG[f]["label"]
        for f in current_signals[:4]
    ]

    parts = [
        f"Risk is {float(row['weekly_risk']):.1f}/100.",
        f"{len(current_signals)} signal(s) currently exceed "
        f"the standardized deviation threshold.",
    ]

    if cascade["cascade_start_week"] > 0:
        parts.append(
            f"The cascade began in Week "
            f"{cascade['cascade_start_week']}."
        )

    if cascade["convergence_week"] > 0:
        parts.append(
            f"Multi-signal convergence first appeared "
            f"in Week {cascade['convergence_week']}."
        )

    parts.append(
        "Current strongest deviations: "
        + ", ".join(labels)
        + "."
    )

    return " ".join(parts)


# =============================================================================
# STUDENT CARDS
# =============================================================================

def build_student_cards(
    trajectory_df: pd.DataFrame,
    cascade_df: pd.DataFrame,
) -> pd.DataFrame:

    cards = []

    cascade_lookup = cascade_df.set_index(
        "student_id"
    )

    for student_id, group in trajectory_df.groupby(
        "student_id"
    ):

        group = group.sort_values("week")
        current = group.iloc[-1]

        first_week, first_signal, detection_type = (
            detect_first_meaningful_deviation(group)
        )

        current_week = int(current["week"])

        lead_time = (
            max(0, current_week - first_week)
            if first_week > 0
            else 0
        )

        weeks_watch = int(
            (group["weekly_risk"] >= WATCH_THRESHOLD).sum()
        )

        weeks_moderate = int(
            (group["weekly_risk"] >= MODERATE_THRESHOLD).sum()
        )

        weeks_high = int(
            (group["weekly_risk"] >= HIGH_THRESHOLD).sum()
        )

        contributors = sorted(
            [
                (
                    feature,
                    float(current[f"{feature}_risk"]),
                )
                for feature in FEATURES
            ],
            key=lambda x: x[1],
            reverse=True,
        )

        top_contributors = [
            FEATURE_CONFIG[f]["label"]
            for f, _ in contributors[:3]
        ]

        signal_count = int(
            current["meaningful_signal_count"]
        )

        anomaly = float(
            current["population_anomaly"]
        )

        if signal_count >= 3 and anomaly >= 60:
            confidence = "HIGH"
        elif signal_count >= 2 or anomaly >= 50:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"

        cascade = cascade_lookup.loc[student_id].to_dict()

        why_now = build_why_now(
            current,
            cascade,
        )

        cards.append({
            "student_id": student_id,
            "current_week": current_week,

            "current_risk_score": round(
                float(current["weekly_risk"]),
                2,
            ),

            "current_severity": current["severity"],

            "trajectory_state": current[
                "trajectory_state"
            ],

            "trajectory_phase": current[
                "trajectory_phase"
            ],

            "risk_velocity": round(
                float(current["risk_velocity"]),
                3,
            ),

            "risk_acceleration": round(
                float(current["risk_acceleration"]),
                3,
            ),

            "first_deviation_week": first_week,

            "earliest_signal": (
                FEATURE_CONFIG[first_signal]["label"]
                if first_signal in FEATURE_CONFIG
                else "NONE"
            ),

            "deviation_detection_type": detection_type,

            "early_warning_lead_time_weeks": lead_time,

            "weeks_above_watch": weeks_watch,
            "weeks_above_moderate": weeks_moderate,
            "weeks_above_high": weeks_high,

            "meaningful_signal_count": signal_count,

            "population_anomaly": round(
                anomaly,
                2,
            ),

            "confidence": confidence,

            "top_contributors": " | ".join(
                top_contributors
            ),

            # ---------------- NEW CASCADE FIELDS ----------------
            "cascade_start_week": cascade[
                "cascade_start_week"
            ],

            "cascade_end_week": cascade[
                "cascade_end_week"
            ],

            "cascade_duration": cascade[
                "cascade_duration"
            ],

            "cascade_depth": cascade[
                "cascade_depth"
            ],

            "leading_signal": (
                FEATURE_CONFIG[
                    cascade["leading_signal"]
                ]["label"]
                if cascade["leading_signal"]
                in FEATURE_CONFIG
                else "NONE"
            ),

            "second_signal": (
                FEATURE_CONFIG[
                    cascade["second_signal"]
                ]["label"]
                if cascade["second_signal"]
                in FEATURE_CONFIG
                else "NONE"
            ),

            "new_signals_per_week": cascade[
                "new_signals_per_week"
            ],

            "convergence_week": cascade[
                "convergence_week"
            ],

            "convergence_strength": cascade[
                "convergence_strength"
            ],

            "cascade_velocity": cascade[
                "cascade_velocity"
            ],

            "early_warning_window": cascade[
                "early_warning_window"
            ],

            "signal_onset_order": json.dumps(
                cascade["signal_onset_order"]
            ),

            "why_now": why_now,

            "explanation": build_why_now(
                current,
                cascade,
            ),
        })

    cards_df = pd.DataFrame(cards)

    cards_df = cards_df.sort_values(
        "current_risk_score",
        ascending=False,
    ).reset_index(drop=True)

    cards_df.insert(
        0,
        "rank",
        np.arange(
            1,
            len(cards_df) + 1,
        ),
    )

    return cards_df


# =============================================================================
# REPLAY DATASET
# =============================================================================

def build_replay_dataset(
    trajectory_df: pd.DataFrame,
    cards_df: pd.DataFrame,
) -> pd.DataFrame:

    replay = trajectory_df.copy()

    card_lookup = cards_df.set_index(
        "student_id"
    )

    replay["first_deviation_week"] = (
        replay["student_id"].map(
            card_lookup["first_deviation_week"]
        )
    )

    replay["earliest_signal"] = (
        replay["student_id"].map(
            card_lookup["earliest_signal"]
        )
    )

    replay["early_warning_lead_time_weeks"] = (
        replay["student_id"].map(
            card_lookup[
                "early_warning_lead_time_weeks"
            ]
        )
    )

    replay["convergence_strength"] = (
        replay["student_id"].map(
            card_lookup[
                "convergence_strength"
            ]
        )
    )

    replay["early_warning_window"] = (
        replay["student_id"].map(
            card_lookup[
                "early_warning_window"
            ]
        )
    )

    # WEBSITE:
    # This replaces the old ambiguous "lead_time_if_current".
    # The frontend can use this as the fixed detection-to-current window.
    replay["lead_time_weeks"] = replay.apply(
        lambda row: (
            max(
                0,
                int(row["week"])
                - int(row["first_deviation_week"]),
            )
            if int(row["first_deviation_week"]) > 0
            else 0
        ),
        axis=1,
    )

    # Event markers for the timeline.
    replay["timeline_event"] = ""

    replay.loc[
        replay["week"]
        == replay["first_deviation_week"],
        "timeline_event",
    ] = "FIRST_DETECTION"

    replay.loc[
        replay["week"]
        == replay["convergence_week"],
        "timeline_event",
    ] = replay.loc[
        replay["week"]
        == replay["convergence_week"],
        "timeline_event",
    ].apply(
        lambda x: (
            "FIRST_DETECTION + CONVERGENCE"
            if x == "FIRST_DETECTION"
            else "CONVERGENCE"
        )
    )

    replay.loc[
        replay["week"]
        == replay.groupby("student_id")["week"].transform("max"),
        "timeline_event",
    ] = replay.loc[
        replay["week"]
        == replay.groupby("student_id")["week"].transform("max"),
        "timeline_event",
    ].apply(
        lambda x: (
            x + " + CURRENT"
            if x
            else "CURRENT"
        )
    )

    columns = [
        "student_id",
        "week",
        "weekly_risk",
        "rolling_risk_3w",
        "severity",
        "risk_velocity",
        "risk_acceleration",
        "trajectory_state",
        "trajectory_phase",
        "personalized_risk",
        "population_anomaly",
        "corroboration",
        "meaningful_signal_count",
        "strong_signal_count",
        "meaningful_signals",
        "cascade_active",
        "cascade_depth",
        "cascade_start_week",
        "convergence_week",
        "convergence_strength",
        "first_deviation_week",
        "earliest_signal",
        "lead_time_weeks",
        "early_warning_window",
        "timeline_event",
    ]

    return replay[
        columns
    ].sort_values(
        ["student_id", "week"]
    ).reset_index(drop=True)


# =============================================================================
# SIGNAL HEATMAP
# =============================================================================

def build_signal_heatmap(
    trajectory_df: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for _, row in trajectory_df.iterrows():

        for feature in FEATURES:

            rows.append({
                "student_id": row["student_id"],
                "week": int(row["week"]),
                "signal": feature,
                "signal_label": FEATURE_CONFIG[
                    feature
                ]["label"],
                "z_score": round(
                    float(row[f"{feature}_z"]),
                    4,
                ),
                "risk": round(
                    float(row[f"{feature}_risk"]),
                    2,
                ),
                "meaningful_deviation": bool(
                    row[f"{feature}_z"]
                    >= Z_THRESHOLD
                ),
            })

    return pd.DataFrame(rows)


# =============================================================================
# TRAJECTORY EVENTS
# =============================================================================

def build_trajectory_events(
    replay_df: pd.DataFrame,
    cascade_df: pd.DataFrame,
) -> pd.DataFrame:

    events = []

    for student_id, group in replay_df.groupby(
        "student_id"
    ):

        group = group.sort_values("week")
        cascade = cascade_df[
            cascade_df["student_id"] == student_id
        ].iloc[0]

        # First meaningful deviation.
        first_week = int(
            cascade["first_deviation_week"]
        )

        if first_week > 0:
            events.append({
                "student_id": student_id,
                "week": first_week,
                "event_type": "FIRST_DETECTION",
                "event_label": "First meaningful deviation",
                "description": (
                    "The predefined early-warning criteria "
                    "were first met."
                ),
            })

        convergence_week = int(
            cascade["convergence_week"]
        )

        if convergence_week > 0:
            events.append({
                "student_id": student_id,
                "week": convergence_week,
                "event_type": "CONVERGENCE",
                "event_label": "Multi-signal convergence",
                "description": (
                    "At least two independent behavioral "
                    "signals were simultaneously meaningful."
                ),
            })

        # Highest risk week.
        peak = group.loc[
            group["weekly_risk"].idxmax()
        ]

        events.append({
            "student_id": student_id,
            "week": int(peak["week"]),
            "event_type": "PEAK_RISK",
            "event_label": "Peak observed risk",
            "description": (
                f"Observed weekly behavioral risk reached "
                f"{float(peak['weekly_risk']):.1f}/100."
            ),
        })

    return pd.DataFrame(events).sort_values(
        ["student_id", "week"]
    ).reset_index(drop=True)


# =============================================================================
# WEBSITE JSON BUNDLE
# =============================================================================

# FINAL WEBPAGE IMPLEMENTATION NOTE
# -----------------------------------------------------------------------------
# The frontend should not re-derive the risk logic itself. It should load the
# generated website_data.json bundle and render the following sections:
#
# 1) Student cards dashboard:
#       - Use student_cards[] for the list of current risk summaries.
#       - Show current_risk_score, current_severity, trajectory_state,
#         trajectory_phase, cascade_depth, leading_signal, convergence_week,
#         early_warning_window, and why_now.
#
# 2) Student detail view:
#       - Use student_cards for the selected student.
#       - Display risk score, severity, phase badge, earliest signal,
#         signal_onset_order, convergence timing, and explanation text.
#
# 3) Replay timeline:
#       - Use replay[] grouped by student_id and ordered by week.
#       - Plot weekly_risk and rolling_risk_3w over time.
#       - Mark timeline_event values: FIRST_DETECTION, CONVERGENCE, CURRENT.
#
# 4) Cascade analysis panel:
#       - Use cascades[] to show cascade_start_week, cascade_depth,
#         leading_signal, convergence_strength, and signal_onset_order.
#
# 5) Aggregate visuals:
#       - Use signal_transitions[] for the temporal signal-flow matrix.
#       - Use trajectory_events[] for the event markers on the timeline.
#
# This keeps the backend and frontend responsibilities clean: the Python code
# calculates risk, while the webpage simply consumes and visualizes the output.
# -----------------------------------------------------------------------------

def dataframe_records(
    df: pd.DataFrame,
) -> List[Dict]:

    # Convert NumPy scalars to native JSON-compatible values.
    records = df.replace(
        {np.nan: None}
    ).to_dict(orient="records")

    return json.loads(
        json.dumps(
            records,
            default=lambda x: (
                x.item()
                if hasattr(x, "item")
                else str(x)
            ),
        )
    )


def build_website_bundle(
    cards_df: pd.DataFrame,
    replay_df: pd.DataFrame,
    cascade_df: pd.DataFrame,
    transitions_df: pd.DataFrame,
    events_df: pd.DataFrame,
) -> Dict:

    bundle = {
        "schema_version": "4.0",
        "project": {
            "name": "Student Wellbeing Early-Warning AI",
            "engine": "Trajectory + Behavioural Cascade Engine",
            "weeks": N_WEEKS,
            "baseline_weeks": BASELINE_WEEKS,
            "signals": [
                {
                    "id": feature,
                    "label": FEATURE_CONFIG[
                        feature
                    ]["label"],
                }
                for feature in FEATURES
            ],
        },

        # WEBSITE:
        # Existing dashboard can keep using this array.
        "student_cards": dataframe_records(
            cards_df
        ),

        # WEBSITE:
        # Group by student_id and animate these records
        # to implement the rewind/replay interface.
        "replay": dataframe_records(
            replay_df
        ),

        # WEBSITE:
        # Use this for the new "cascade analysis" panel.
        "cascades": dataframe_records(
            cascade_df
        ),

        # WEBSITE:
        # Use this for an aggregate signal-transition visualization.
        "signal_transitions": dataframe_records(
            transitions_df
        ),

        # WEBSITE:
        # Timeline markers for first detection, convergence and peak.
        "trajectory_events": dataframe_records(
            events_df
        ),

        "frontend_contract": {
            "student_detail": {
                "risk": "student_cards.current_risk_score",
                "severity": "student_cards.current_severity",
                "trajectory": "student_cards.trajectory_state",
                "phase": "student_cards.trajectory_phase",
                "cascade_depth": "student_cards.cascade_depth",
                "leading_signal": "student_cards.leading_signal",
                "first_detection": "student_cards.first_deviation_week",
                "convergence": "student_cards.convergence_week",
                "early_warning_window": (
                    "student_cards.early_warning_window"
                ),
                "why_now": "student_cards.why_now",
                "onset_order": (
                    "student_cards.signal_onset_order"
                ),
            },

            "replay": {
                "source": "replay",
                "group_by": "student_id",
                "x_axis": "week",
                "risk_line": "weekly_risk",
                "rolling_line": "rolling_risk_3w",
                "event_marker": "timeline_event",
                "phase": "trajectory_phase",
            },
        },
    }

    return bundle


# =============================================================================
# VALIDATION
# =============================================================================

def validate_outputs(
    original_df: pd.DataFrame,
    trajectory_df: pd.DataFrame,
    cards_df: pd.DataFrame,
    replay_df: pd.DataFrame,
    cascade_df: pd.DataFrame,
    transitions_df: pd.DataFrame,
    events_df: pd.DataFrame,
):

    expected_rows = (
        N_STUDENTS * N_WEEKS
    )

    assert len(original_df) == expected_rows
    assert len(trajectory_df) == expected_rows
    assert len(replay_df) == expected_rows

    assert len(cards_df) == N_STUDENTS
    assert len(cascade_df) == N_STUDENTS

    assert trajectory_df[
        "weekly_risk"
    ].between(0, 100).all()

    assert trajectory_df[
        "rolling_risk_3w"
    ].between(0, 100).all()

    assert trajectory_df[
        "population_anomaly"
    ].between(0, 100).all()

    assert trajectory_df[
        "convergence_strength"
    ].between(0, 100).all()

    # Every student must have all weeks.
    week_counts = (
        trajectory_df.groupby(
            "student_id"
        )["week"].nunique()
    )

    assert (
        week_counts == N_WEEKS
    ).all()

    # Every cascade must have a valid depth.
    assert (
        cascade_df["cascade_depth"]
        .between(0, len(FEATURES))
        .all()
    )

    # Replay must be unique by student/week.
    assert not replay_df.duplicated(
        ["student_id", "week"]
    ).any()

    print("\nVALIDATION PASSED")
    print("-----------------")
    print(f"Students: {N_STUDENTS}")
    print(f"Weeks/student: {N_WEEKS}")
    print(f"Trajectory rows: {len(trajectory_df)}")
    print(f"Cascade records: {len(cascade_df)}")
    print(
        f"Transition records: {len(transitions_df)}"
    )
    print(f"Timeline events: {len(events_df)}")
    print("Risk range: 0-100")
    print("Replay keys: student_id + week")
    print("All outputs are structurally complete.")


# =============================================================================
# SAVE OUTPUTS
# =============================================================================

def save_outputs(
    raw_df: pd.DataFrame,
    trajectory_df: pd.DataFrame,
    cards_df: pd.DataFrame,
    replay_df: pd.DataFrame,
    heatmap_df: pd.DataFrame,
    cascade_df: pd.DataFrame,
    transitions_df: pd.DataFrame,
    events_df: pd.DataFrame,
    website_bundle: Dict,
    output_dir: str = OUTPUT_DIR,
) -> Dict[str, str]:

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    dataframes = {
        "raw": (
            raw_df,
            "synthetic_student_behavior_9_weeks.csv",
        ),
        "trajectory": (
            trajectory_df,
            "student_trajectory_engine.csv",
        ),
        "cards": (
            cards_df,
            "student_risk_cards.csv",
        ),
        "replay": (
            replay_df,
            "early_warning_replay.csv",
        ),
        "heatmap": (
            heatmap_df,
            "signal_heatmap_data.csv",
        ),
        "cascade": (
            cascade_df,
            "behavioral_cascade_analysis.csv",
        ),
        "transitions": (
            transitions_df,
            "signal_transition_matrix.csv",
        ),
        "events": (
            events_df,
            "trajectory_events.csv",
        ),
    }

    paths = {}

    for name, (df, filename) in dataframes.items():

        path = os.path.join(
            output_dir,
            filename,
        )

        df.to_csv(
            path,
            index=False,
        )

        paths[name] = path

    # Existing website-friendly current-card JSON.
    cards_json_path = os.path.join(
        output_dir,
        "student_risk_cards.json",
    )

    with open(
        cards_json_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            dataframe_records(cards_df),
            f,
            indent=2,
        )

    paths["cards_json"] = cards_json_path

    # New replay JSON.
    replay_json_path = os.path.join(
        output_dir,
        "early_warning_replay.json",
    )

    with open(
        replay_json_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            dataframe_records(replay_df),
            f,
            indent=2,
        )

    paths["replay_json"] = replay_json_path

    # New cascade JSON.
    cascade_json_path = os.path.join(
        output_dir,
        "behavioral_cascade_analysis.json",
    )

    with open(
        cascade_json_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            dataframe_records(cascade_df),
            f,
            indent=2,
        )

    paths["cascade_json"] = cascade_json_path

    # Aggregate transition JSON.
    transition_json_path = os.path.join(
        output_dir,
        "signal_transition_matrix.json",
    )

    with open(
        transition_json_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            dataframe_records(
                transitions_df
            ),
            f,
            indent=2,
        )

    paths["transition_json"] = (
        transition_json_path
    )

    # Timeline events JSON.
    events_json_path = os.path.join(
        output_dir,
        "trajectory_events.json",
    )

    with open(
        events_json_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            dataframe_records(events_df),
            f,
            indent=2,
        )

    paths["events_json"] = events_json_path

    # ONE file for the website.
    website_path = os.path.join(
        output_dir,
        "website_data.json",
    )

    with open(
        website_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            website_bundle,
            f,
            indent=2,
        )

    paths["website_bundle"] = website_path

    return paths


# =============================================================================
# TERMINAL REPORT
# =============================================================================

def print_final_report(
    cards_df: pd.DataFrame,
    replay_df: pd.DataFrame,
    cascade_df: pd.DataFrame,
):

    print("\n")
    print("=" * 90)
    print("STUDENT WELLBEING EARLY-WARNING AI")
    print("VERSION 4.0 — TRAJECTORY + BEHAVIOURAL CASCADE ENGINE")
    print("=" * 90)

    print(
        f"\nStudents analysed : "
        f"{cards_df['student_id'].nunique()}"
    )

    print(
        f"Weeks analysed   : "
        f"{replay_df['week'].nunique()}"
    )

    print(
        f"Total trajectories: "
        f"{len(replay_df)}"
    )

    print("\nTOP CURRENT TRAJECTORIES")
    print("-" * 90)

    display_columns = [
        "rank",
        "student_id",
        "current_risk_score",
        "current_severity",
        "trajectory_state",
        "trajectory_phase",
        "cascade_depth",
        "leading_signal",
        "convergence_week",
        "early_warning_window",
    ]

    print(
        cards_df[
            display_columns
        ]
        .head(10)
        .to_string(index=False)
    )

    print("\n")
    print("CASCADE EXAMPLE")
    print("-" * 90)

    # Pick the highest current-risk student.
    top_student = cards_df.iloc[0][
        "student_id"
    ]

    top_cascade = cascade_df[
        cascade_df["student_id"]
        == top_student
    ].iloc[0]

    print(
        f"\nStudent: {top_student}"
    )

    print(
        f"Cascade start       : "
        f"Week {int(top_cascade['cascade_start_week'])}"
    )

    print(
        f"Cascade depth       : "
        f"{int(top_cascade['cascade_depth'])} signals"
    )

    print(
        f"Leading signal      : "
        f"{top_cascade['leading_signal']}"
    )

    print(
        f"Convergence week    : "
        f"Week {int(top_cascade['convergence_week'])}"
    )

    print(
        f"Early-warning window: "
        f"{int(top_cascade['early_warning_window'])} weeks"
    )

    print("\nSignal onset order:")

    for event in top_cascade[
        "signal_onset_order"
    ]:
        print(
            f"  W{event['week']}  "
            f"{event['label']}"
        )

    print("\n")
    print("PRIVACY / SAFETY NOTE")
    print("-" * 90)

    print(
        "This system detects unusual behavioral change patterns. "
        "It does not diagnose stress, depression, self-harm, "
        "or any other condition. Leading signals are temporal "
        "indicators, not causal claims. High-risk records require "
        "human review and contextual verification."
    )


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():

    print("=" * 90)
    print("STUDENT WELLBEING EARLY-WARNING AI")
    print("VERSION 4.0 — TRAJECTORY + BEHAVIOURAL CASCADE ENGINE")
    print("=" * 90)

    # -------------------------------------------------------------------------
    # 1. SYNTHETIC DATA
    # -------------------------------------------------------------------------

    print(
        "\n[1/12] Generating synthetic 9-week dataset..."
    )

    raw_df = generate_synthetic_dataset()

    print(
        f"       Generated {len(raw_df)} records "
        f"for {N_STUDENTS} students."
    )

    # -------------------------------------------------------------------------
    # 2. PERSONAL BASELINES
    # -------------------------------------------------------------------------

    print(
        "\n[2/12] Building individual historical baselines..."
    )

    baselines = build_baselines(
        raw_df,
        BASELINE_WEEKS,
    )

    print(
        f"       Baseline period: "
        f"weeks 1-{BASELINE_WEEKS}"
    )

    # -------------------------------------------------------------------------
    # 3. FEATURE DEVIATIONS
    # -------------------------------------------------------------------------

    print(
        "\n[3/12] Calculating feature-level deviations..."
    )

    _, risk_df = calculate_feature_risks(
        raw_df,
        baselines,
    )

    print(
        f"       Calculated {len(FEATURES)} "
        f"independent behavioral signals."
    )

    # -------------------------------------------------------------------------
    # 4. POPULATION ANOMALY
    # -------------------------------------------------------------------------

    print(
        "\n[4/12] Calculating population-level anomalies..."
    )

    risk_df = calculate_population_anomaly(
        raw_df,
        risk_df,
    )

    print(
        "       Isolation Forest anomaly model complete."
    )

    # -------------------------------------------------------------------------
    # 5. FINAL WEEKLY RISK
    # -------------------------------------------------------------------------

    print(
        "\n[5/12] Calculating final weekly behavioral risk..."
    )

    risk_df = calculate_final_weekly_risk(
        risk_df,
    )

    print(
        "       Weekly 0-100 behavioral risk generated."
    )

    # -------------------------------------------------------------------------
    # 6. TRAJECTORY
    # -------------------------------------------------------------------------

    print(
        "\n[6/12] Running trajectory engine..."
    )

    trajectory_df = add_trajectory_engine(
        risk_df,
    )

    print(
        "       Rolling risk + velocity + acceleration calculated."
    )

    # -------------------------------------------------------------------------
    # 7. CASCADE ENGINE
    # -------------------------------------------------------------------------

    print(
        "\n[7/12] Running behavioral cascade engine..."
    )

    trajectory_df, cascade_df = (
        add_cascade_engine(
            trajectory_df
        )
    )

    print(
        "       Signal onset order + cascade phases calculated."
    )

    # -------------------------------------------------------------------------
    # 8. TRANSITION MATRIX
    # -------------------------------------------------------------------------

    print(
        "\n[8/12] Building temporal signal-transition matrix..."
    )

    transitions_df = (
        build_signal_transition_matrix(
            trajectory_df
        )
    )

    print(
        f"       {len(transitions_df)} "
        f"aggregate signal transitions generated."
    )

    # -------------------------------------------------------------------------
    # 9. STUDENT CARDS
    # -------------------------------------------------------------------------

    print(
        "\n[9/12] Building explainable student cards..."
    )

    cards_df = build_student_cards(
        trajectory_df,
        cascade_df,
    )

    print(
        "       Risk + trajectory + cascade cards generated."
    )

    # -------------------------------------------------------------------------
    # 10. REPLAY / HEATMAP / EVENTS
    # -------------------------------------------------------------------------

    print(
        "\n[10/12] Generating website replay and event datasets..."
    )

    replay_df = build_replay_dataset(
        trajectory_df,
        cards_df,
    )

    heatmap_df = build_signal_heatmap(
        trajectory_df,
    )

    events_df = build_trajectory_events(
        replay_df,
        cascade_df,
    )

    print(
        "       Replay + heatmap + timeline events generated."
    )

    # -------------------------------------------------------------------------
    # 11. WEBSITE BUNDLE
    # -------------------------------------------------------------------------

    print(
        "\n[11/12] Building website-ready JSON bundle..."
    )

    website_bundle = build_website_bundle(
        cards_df,
        replay_df,
        cascade_df,
        transitions_df,
        events_df,
    )

    print(
        "       website_data.json ready for frontend integration."
    )

    # -------------------------------------------------------------------------
    # 12. VALIDATE + SAVE
    # -------------------------------------------------------------------------

    print(
        "\n[12/12] Validating and saving all outputs..."
    )

    validate_outputs(
        raw_df,
        trajectory_df,
        cards_df,
        replay_df,
        cascade_df,
        transitions_df,
        events_df,
    )

    paths = save_outputs(
        raw_df,
        trajectory_df,
        cards_df,
        replay_df,
        heatmap_df,
        cascade_df,
        transitions_df,
        events_df,
        website_bundle,
    )

    print_final_report(
        cards_df,
        replay_df,
        cascade_df,
    )

    print("\n")
    print("=" * 90)
    print("FILES GENERATED")
    print("=" * 90)

    for name, path in paths.items():
        print(
            f"{name:18s} -> {path}"
        )

    print("\n")
    print("=" * 90)
    print("PIPELINE COMPLETE")
    print("=" * 90)

    print(
        "\nYour existing early-warning system now has:"
    )

    print(
        "  • personalized behavioral baselines"
    )
    print(
        "  • 11-signal risk fusion"
    )
    print(
        "  • population anomaly detection"
    )
    print(
        "  • rolling trajectory"
    )
    print(
        "  • velocity + acceleration"
    )
    print(
        "  • first meaningful deviation"
    )
    print(
        "  • early-warning lead time"
    )
    print(
        "  • signal onset ordering"
    )
    print(
        "  • behavioral cascade detection"
    )
    print(
        "  • convergence strength"
    )
    print(
        "  • trajectory phase classification"
    )
    print(
        "  • temporal transition matrix"
    )
    print(
        "  • why-now explanations"
    )
    print(
        "  • rewind/replay event markers"
    )
    print(
        "  • website-ready JSON bundle"
    )


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()
'''

exec(code, globals())
