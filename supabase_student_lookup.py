"""Upload and retrieve nine-week student analysis from Supabase."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from supabase import Client, create_client


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "wellbeing_outputs"
TABLE_NAME = "student_weekly_analysis"
EXPECTED_WEEKS = list(range(1, 10))


def create_supabase_client() -> Client:
    load_dotenv(ROOT / ".env")
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SECRET_KEY") or os.getenv(
        "SUPABASE_SERVICE_ROLE_KEY"
    )

    if not url or not key:
        raise RuntimeError(
            "Set SUPABASE_URL and SUPABASE_SECRET_KEY in the project .env file."
        )

    return create_client(url, key)


def load_weekly_records() -> list[dict[str, Any]]:
    raw_path = OUTPUT_DIR / "synthetic_student_behavior_9_weeks.csv"
    analysis_path = OUTPUT_DIR / "student_trajectory_engine.csv"
    cards_path = OUTPUT_DIR / "student_risk_cards.csv"

    if not raw_path.exists() or not analysis_path.exists() or not cards_path.exists():
        raise FileNotFoundError(
            "Generated CSVs are missing. Run student_wellbeing_ai.py first."
        )

    raw = pd.read_csv(raw_path)
    analysis = pd.read_csv(analysis_path)
    cards = pd.read_csv(cards_path)
    combined = raw.merge(
        analysis,
        on=["student_id", "week"],
        how="inner",
        validate="one_to_one",
        suffixes=("", "_analysis"),
    )
    card_fields = cards[
        [
            "student_id",
            "first_deviation_week",
            "earliest_signal",
            "deviation_detection_type",
        ]
    ]
    combined = combined.merge(
        card_fields,
        on="student_id",
        how="left",
        validate="many_to_one",
    )

    if len(combined) != len(raw) or len(combined) != len(analysis):
        raise ValueError("Raw and analysis CSVs do not contain matching weeks.")

    for student_id, group in combined.groupby("student_id"):
        actual_weeks = sorted(group["week"].astype(int).tolist())
        if actual_weeks != EXPECTED_WEEKS:
            raise ValueError(
                f"{student_id} has weeks {actual_weeks}; expected weeks 1 through 9."
            )

    combined["week"] = combined["week"].astype(int)
    return json.loads(combined.to_json(orient="records"))


def upload_weekly_records() -> None:
    client = create_supabase_client()
    records = load_weekly_records()
    batch_size = 100

    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        payload = [
            {
                "student_id": record["student_id"],
                "week": record["week"],
                "analysis": record,
            }
            for record in batch
        ]
        client.table(TABLE_NAME).upsert(
            payload,
            on_conflict="student_id,week",
        ).execute()

    print(f"Uploaded {len(records)} weekly records for {len(records) // 9} students.")


def normalize_student_id(value: str) -> str:
    match = re.fullmatch(r"(?:STU-)?(\d+)", value.strip(), re.IGNORECASE)
    if not match:
        raise ValueError("Enter a student number such as 7 or STU-007.")
    return f"STU-{int(match.group(1)):03d}"


def print_student_timeline(student_id: str, rows: list[dict[str, Any]]) -> None:
    print(f"\nNine-week wellbeing analysis: {student_id}")
    print("=" * 42)

    for result in rows:
        analysis = result["analysis"]
        week = int(result["week"])
        risk = float(analysis.get("weekly_risk", 0))
        signal_count = int(analysis.get("meaningful_signal_count", 0))
        first_deviation_week = int(analysis.get("first_deviation_week", -1))
        notes = []

        if week == first_deviation_week:
            signal = analysis.get("earliest_signal", "")
            note = "First behavioral deviation detected."
            if signal and signal != "NONE":
                note += f" Earliest signal: {signal}."
            notes.append(note)

        if signal_count >= 4:
            notes.append(f"{signal_count} signals simultaneously deviating.")
        elif signal_count >= 2:
            notes.append(f"{signal_count} independent signals converging.")

        trajectory = str(analysis.get("trajectory_state", "")).upper()
        if trajectory == "RAPIDLY ESCALATING":
            notes.append("Trajectory now rapidly escalating.")
        elif trajectory == "ESCALATING":
            notes.append("Trajectory now escalating.")

        anomaly = float(analysis.get("population_anomaly", 0))
        if signal_count >= 3 and anomaly >= 60:
            notes.append("HIGH-CONFIDENCE MULTI-SIGNAL DEVIATION")

        print(f"\nW{week}")
        print(f"Risk: {risk:.0f}")
        for note in notes:
            print(note)


def show_student_analysis(student_number: str) -> None:
    student_id = normalize_student_id(student_number)
    client = create_supabase_client()
    response = (
        client.table(TABLE_NAME)
        .select("week,analysis")
        .eq("student_id", student_id)
        .order("week")
        .execute()
    )
    rows = response.data

    if not rows:
        print(f"No records found for {student_id}.")
        return

    print_student_timeline(student_id, rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upload weekly analysis or look up a student's nine-week timeline."
    )
    parser.add_argument(
        "action",
        nargs="?",
        choices=("upload", "lookup"),
        default="lookup",
        help="Use 'upload' for the CSVs, or 'lookup' to query a student.",
    )
    parser.add_argument(
        "student_number",
        nargs="?",
        help="Student number such as 7 or STU-007 (lookup only).",
    )
    args = parser.parse_args()

    if args.action == "upload":
        upload_weekly_records()
        return

    student_number = args.student_number or input("Student number: ")
    show_student_analysis(student_number)


if __name__ == "__main__":
    main()