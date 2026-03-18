#!/usr/bin/env python3
"""Predict future trends for active market narratives."""

from __future__ import annotations

import argparse
from datetime import datetime

from narrativealpha.analysis.prediction import TrendPredictor
from narrativealpha.analysis.tracking import NarrativeTracker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict trends for active narratives")
    parser.add_argument(
        "--horizon", type=int, default=24, help="Forecast horizon in hours"
    )
    parser.add_argument("--db-path", type=str, default=None, help="Override SQLite path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    tracker = NarrativeTracker(db_path=args.db_path)
    predictor = TrendPredictor(forecast_horizon_hours=args.horizon)

    active_narratives = tracker.list_active()
    if not active_narratives:
        print("No active narratives found to predict.")
        return

    print(f"Predicting trends for {len(active_narratives)} narratives...")
    print(f"Forecast horizon: {args.horizon} hours")
    print("-" * 60)

    predictions = []
    for narrative in active_narratives:
        history = tracker.get_history(narrative.id)
        result = predictor.predict_next_score(narrative, history)
        predictions.append((narrative, result))

    # Sort by predicted score improvement or overall predicted score
    predictions.sort(key=lambda x: x[1].predicted_score, reverse=True)

    for narrative, res in predictions:
        trend_icon = "📈" if res.trend_direction == "up" else "📉" if res.trend_direction == "down" else "➡️"
        print(f"{trend_icon} {narrative.name}")
        print(f"   Current Score:   {res.current_score:.4f}")
        print(f"   Predicted:       {res.predicted_score:.4f} ({res.trend_direction})")
        print(f"   Confidence:      {res.confidence * 100:.1f}%")
        print(f"   Keywords:        {', '.join(narrative.keywords[:3])}")
        print()


if __name__ == "__main__":
    main()
