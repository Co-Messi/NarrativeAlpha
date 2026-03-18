"""Trend prediction algorithms for market narratives."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import List, Optional

import numpy as np
from pydantic import BaseModel, Field

from narrativealpha.models import Narrative


class PredictionResult(BaseModel):
    """Result of a trend prediction for a narrative."""

    narrative_id: str
    current_score: float
    predicted_score: float
    confidence: float
    trend_direction: str = Field(..., description="'up', 'down', or 'sideways'")
    forecast_horizon_hours: int


class TrendPredictor:
    """Predicts future narrative scores based on historical data."""

    def __init__(self, forecast_horizon_hours: int = 24):
        self.forecast_horizon_hours = forecast_horizon_hours

    def predict_next_score(
        self, narrative: Narrative, history: List[dict]
    ) -> PredictionResult:
        """
        Predicts the next overall_score using historical data points.
        history: List of dicts containing 'overall_score' and 'updated_at' (ISO strings).
        """
        if not history or len(history) < 2:
            return PredictionResult(
                narrative_id=narrative.id,
                current_score=narrative.overall_score,
                predicted_score=narrative.overall_score,
                confidence=0.0,
                trend_direction="sideways",
                forecast_horizon_hours=self.forecast_horizon_hours,
            )

        # Sort history by time
        history = sorted(history, key=lambda x: x["updated_at"])

        # Extract scores and relative times (in hours)
        base_time = datetime.fromisoformat(history[0]["updated_at"])
        times = []
        scores = []
        for point in history:
            dt = datetime.fromisoformat(point["updated_at"])
            times.append((dt - base_time).total_seconds() / 3600.0)
            scores.append(point["overall_score"])

        # Add the current narrative state as the latest point
        current_time = (datetime.now(timezone.utc) - base_time.replace(tzinfo=timezone.utc)).total_seconds() / 3600.0
        times.append(current_time)
        scores.append(narrative.overall_score)

        # Linear regression: y = mx + b
        x = np.array(times)
        y = np.array(scores)
        
        if len(x) < 2:
            return PredictionResult(
                narrative_id=narrative.id,
                current_score=narrative.overall_score,
                predicted_score=narrative.overall_score,
                confidence=0.0,
                trend_direction="sideways",
                forecast_horizon_hours=self.forecast_horizon_hours,
            )

        m, b = np.polyfit(x, y, 1)

        # Predict value at current_time + forecast_horizon_hours
        predicted_time = current_time + self.forecast_horizon_hours
        predicted_score = m * predicted_time + b
        
        # Clip predicted score to [0, 1] as per scoring engine logic (mostly)
        predicted_score = max(0.0, min(1.0, round(float(predicted_score), 4)))

        # Determine trend direction
        if m > 0.005: # Threshold for 'up'
            trend = "up"
        elif m < -0.005: # Threshold for 'down'
            trend = "down"
        else:
            trend = "sideways"

        # Confidence based on R-squared or simply data density
        # For simplicity, we use the number of points and the fit error
        residuals = np.sum((y - (m * x + b))**2)
        variance = np.sum((y - np.mean(y))**2)
        if variance == 0:
            r_squared = 1.0
        else:
            r_squared = 1 - (residuals / variance)
            
        confidence = round(max(0.0, min(1.0, float(r_squared) * (min(len(x), 10) / 10.0))), 2)

        return PredictionResult(
            narrative_id=narrative.id,
            current_score=narrative.overall_score,
            predicted_score=predicted_score,
            confidence=confidence,
            trend_direction=trend,
            forecast_horizon_hours=self.forecast_horizon_hours,
        )
