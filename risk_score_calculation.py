"""Student wellbeing risk-score calculations used by the main pipeline.

For each signal, the model compares a student's current value with their
personal baseline. Those feature risks are weighted, then combined with a
population anomaly score and multi-signal corroboration:

    weekly risk = 75% personalized risk + 15% population anomaly
                  + 10% signal corroboration

The result is a 0-100 behavioral change indicator, not a diagnosis.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


RANDOM_SEED = 42
BASELINE_WEEKS = 4
Z_THRESHOLD = 2.0
STRONG_Z_THRESHOLD = 2.5
CONVERGENCE_SIGNAL_COUNT = 2

WATCH_THRESHOLD = 20
MODERATE_THRESHOLD = 40
HIGH_THRESHOLD = 60
VERY_HIGH_THRESHOLD = 80

FEATURE_CONFIG = {
    "class_participation": {
        "weight": 0.14,
        "direction": -1,
        "label": "Class participation",
        "description": "Change in classroom participation",
    },
    "arrival_irregularity": {
        "weight": 0.08,
        "direction": 1,
        "label": "Arrival irregularity",
        "description": "Change in arrival-time consistency",
    },
    "early_departures": {
        "weight": 0.08,
        "direction": 1,
        "label": "Early departures",
        "description": "Change in early school departures",
    },
    "extracurricular_activity": {
        "weight": 0.10,
        "direction": -1,
        "label": "Extracurricular activity",
        "description": "Change in extracurricular participation",
    },
    "library_resource_usage": {
        "weight": 0.08,
        "direction": -1,
        "label": "Library/resource usage",
        "description": "Change in school resource engagement",
    },
    "lms_session_duration": {
        "weight": 0.10,
        "direction": -1,
        "label": "LMS session duration",
        "description": "Change in learning-platform engagement",
    },
    "academic_help_requests": {
        "weight": 0.08,
        "direction": 1,
        "label": "Academic help requests",
        "description": "Unusual change in academic-help utilization",
    },
    "meal_usage": {
        "weight": 0.08,
        "direction": -1,
        "label": "Meal usage",
        "description": "Change in school meal-system usage",
    },
    "transport_irregularity": {
        "weight": 0.04,
        "direction": 1,
        "label": "Transport irregularity",
        "description": "Change in school-transport regularity",
    },
    "schedule_changes": {
        "weight": 0.12,
        "direction": 1,
        "label": "Schedule changes",
        "description": "Change in schedule/attendance-pattern irregularity",
    },
    "digital_timing_shift": {
        "weight": 0.10,
        "direction": 1,
        "label": "Digital timing shift",
        "description": "Change in learning activity timing",
    },
}

FEATURES = list(FEATURE_CONFIG)


@dataclass
class StudentBaseline:
    means: Dict[str, float]
    stds: Dict[str, float]


def clip(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return float(np.clip(value, low, high))


def safe_std(values: np.ndarray, minimum: float = 1e-6) -> float:
    standard_deviation = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return max(standard_deviation, minimum)


def risk_from_z(z_score: float) -> float:
    """Convert a directional z-score into a nonlinear 0-100 risk value."""
    score = 100.0 / (1.0 + math.exp(-2.0 * (z_score - 1.5)))
    return clip(score)


def severity_from_score(score: float) -> str:
    if score < WATCH_THRESHOLD:
        return "LOW"
    if score < MODERATE_THRESHOLD:
        return "WATCH"
    if score < HIGH_THRESHOLD:
        return "MODERATE"
    if score < VERY_HIGH_THRESHOLD:
        return "HIGH"
    return "VERY HIGH"


def build_baselines(
    df: pd.DataFrame,
    baseline_weeks: int = BASELINE_WEEKS,
) -> Dict[str, StudentBaseline]:
    """Summarize each student's first four weeks for personal comparison."""
    baselines = {}

    for student_id, group in df.groupby("student_id"):
        baseline_rows = group[group["week"] <= baseline_weeks]
        means = {}
        standard_deviations = {}

        for feature in FEATURES:
            values = baseline_rows[feature].to_numpy(dtype=float)
            means[feature] = float(np.mean(values))
            standard_deviations[feature] = safe_std(values)

        baselines[student_id] = StudentBaseline(
            means=means,
            stds=standard_deviations,
        )

    return baselines


def calculate_feature_risks(
    df: pd.DataFrame,
    baselines: Dict[str, StudentBaseline],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate signal z-scores, weighted personal risk, and corroboration."""
    risk_records = []

    for _, row in df.iterrows():
        student_id = row["student_id"]
        baseline = baselines[student_id]
        feature_z_scores = {}
        feature_risks = {}

        for feature in FEATURES:
            config = FEATURE_CONFIG[feature]
            z_score = (
                config["direction"]
                * (float(row[feature]) - baseline.means[feature])
                / baseline.stds[feature]
            )
            feature_z_scores[feature] = z_score
            feature_risks[f"{feature}_z"] = z_score
            feature_risks[f"{feature}_risk"] = risk_from_z(z_score)

        personalized_risk = sum(
            FEATURE_CONFIG[feature]["weight"]
            * feature_risks[f"{feature}_risk"]
            for feature in FEATURES
        )
        meaningful_signals = [
            feature
            for feature in FEATURES
            if feature_z_scores[feature] >= Z_THRESHOLD
        ]
        strong_signals = [
            feature
            for feature in FEATURES
            if feature_z_scores[feature] >= STRONG_Z_THRESHOLD
        ]
        corroboration = clip(
            len(meaningful_signals) / CONVERGENCE_SIGNAL_COUNT * 100
        )

        record = {
            "student_id": student_id,
            "week": int(row["week"]),
            "personalized_risk": personalized_risk,
            "meaningful_signal_count": len(meaningful_signals),
            "strong_signal_count": len(strong_signals),
            "corroboration": corroboration,
            "meaningful_signals": "|".join(meaningful_signals),
            **feature_risks,
        }
        risk_records.append(record)

    return df.copy(), pd.DataFrame(risk_records)


def calculate_population_anomaly(
    original_df: pd.DataFrame,
    risk_df: pd.DataFrame,
) -> pd.DataFrame:
    """Score each student's unusualness relative to peers in the same week."""
    result = risk_df.copy()
    result["population_anomaly"] = 0.0

    for week in sorted(result["week"].unique()):
        week_mask = result["week"] == week
        week_data = result.loc[week_mask]
        feature_values = [
            original_df.loc[original_df["week"] == week, feature].to_numpy(dtype=float)
            for feature in FEATURES
        ]
        feature_values = [
            values for values in feature_values if len(values) == len(week_data)
        ]
        data = np.array(feature_values).T

        if len(data) < 10:
            continue

        scaled_data = StandardScaler().fit_transform(data)
        detector = IsolationForest(
            n_estimators=250,
            contamination=0.10,
            random_state=RANDOM_SEED + int(week),
        )
        detector.fit(scaled_data)
        raw_scores = -detector.decision_function(scaled_data)
        minimum = float(np.min(raw_scores))
        maximum = float(np.max(raw_scores))

        if maximum - minimum < 1e-9:
            normalized_scores = np.zeros_like(raw_scores)
        else:
            normalized_scores = (raw_scores - minimum) / (maximum - minimum) * 100

        result.loc[week_mask, "population_anomaly"] = normalized_scores

    return result


def calculate_final_weekly_risk(risk_df: pd.DataFrame) -> pd.DataFrame:
    """Combine the model's three components into the final weekly score."""
    result = risk_df.copy()
    result["weekly_risk"] = (
        0.75 * result["personalized_risk"]
        + 0.15 * result["population_anomaly"]
        + 0.10 * result["corroboration"]
    ).clip(0, 100)
    result["severity"] = result["weekly_risk"].apply(severity_from_score)
    return result