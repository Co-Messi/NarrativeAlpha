"""Unit tests for the trend prediction module."""

from datetime import datetime, timedelta, timezone

import pytest

from narrativealpha.analysis.prediction import TrendPredictor, PredictionResult
from narrativealpha.models import Narrative


@pytest.fixture
def sample_narrative():
    return Narrative(
        id="test-narrative-1",
        name="Test Narrative",
        description="A test narrative for prediction",
        overall_score=0.5,
        velocity_score=0.5,
        saturation_score=0.5,
        sentiment_score=0.5,
        confidence=0.8,
        keywords=["test", "crypto", "trend"],
    )


def test_prediction_no_history(sample_narrative):
    predictor = TrendPredictor(forecast_horizon_hours=24)
    result = predictor.predict_next_score(sample_narrative, [])

    assert isinstance(result, PredictionResult)
    assert result.narrative_id == sample_narrative.id
    assert result.current_score == 0.5
    assert result.predicted_score == 0.5
    assert result.trend_direction == "sideways"
    assert result.confidence == 0.0


def test_prediction_upward_trend(sample_narrative):
    predictor = TrendPredictor(forecast_horizon_hours=10)
    
    # Historical data points showing upward trend
    now = datetime.now(timezone.utc)
    history = [
        {"overall_score": 0.1, "updated_at": (now - timedelta(hours=30)).isoformat()},
        {"overall_score": 0.2, "updated_at": (now - timedelta(hours=20)).isoformat()},
        {"overall_score": 0.3, "updated_at": (now - timedelta(hours=10)).isoformat()},
    ]
    
    # Current score is 0.5 (set in fixture)
    result = predictor.predict_next_score(sample_narrative, history)

    assert result.predicted_score > 0.5
    assert result.trend_direction == "up"
    assert result.confidence >= 0.3


def test_prediction_downward_trend(sample_narrative):
    # Update fixture for downward trend
    sample_narrative = sample_narrative.model_copy(update={"overall_score": 0.1})
    predictor = TrendPredictor(forecast_horizon_hours=10)
    
    # Historical data points showing downward trend
    now = datetime.now(timezone.utc)
    history = [
        {"overall_score": 0.5, "updated_at": (now - timedelta(hours=30)).isoformat()},
        {"overall_score": 0.4, "updated_at": (now - timedelta(hours=20)).isoformat()},
        {"overall_score": 0.3, "updated_at": (now - timedelta(hours=10)).isoformat()},
    ]
    
    result = predictor.predict_next_score(sample_narrative, history)

    assert result.predicted_score < 0.1
    assert result.trend_direction == "down"
    assert result.confidence >= 0.3


def test_prediction_sideways_trend(sample_narrative):
    predictor = TrendPredictor(forecast_horizon_hours=10)
    
    # History with constant scores
    now = datetime.now(timezone.utc)
    history = [
        {"overall_score": 0.5, "updated_at": (now - timedelta(hours=20)).isoformat()},
        {"overall_score": 0.5, "updated_at": (now - timedelta(hours=10)).isoformat()},
    ]
    
    result = predictor.predict_next_score(sample_narrative, history)

    assert result.predicted_score == 0.5
    assert result.trend_direction == "sideways"
