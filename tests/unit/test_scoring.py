"""Unit tests for velocity and saturation scoring."""

import pytest
from datetime import datetime, timedelta, timezone
from narrativealpha.models import Narrative, SocialPost
from narrativealpha.analysis.scoring import ScoringEngine

@pytest.fixture
def scoring_engine():
    return ScoringEngine(time_window_hours=24)

@pytest.fixture
def sample_posts():
    now = datetime.now(timezone.utc)
    return [
        SocialPost(
            id=f"post_{i}",
            platform="twitter",
            author_id=f"user_{i % 5}", # 5 unique authors
            author_username=f"user_{i % 5}",
            text=f"Sample text for post {i}",
            created_at=now - timedelta(hours=i),
            likes=100 * i,
            replies=10 * i,
            reposts=5 * i
        )
        for i in range(10)
    ]

@pytest.fixture
def sample_narrative():
    return Narrative(
        id="narr_1",
        name="Test Narrative",
        description="Testing velocity and saturation",
        post_ids=[f"post_{i}" for i in range(5)] # First 5 posts
    )

def test_calculate_velocity(scoring_engine, sample_narrative, sample_posts):
    velocity = scoring_engine.calculate_velocity(sample_narrative, sample_posts)
    assert 0.0 <= velocity <= 1.0
    assert velocity > 0 # Should have some velocity as posts are recent

def test_calculate_saturation(scoring_engine, sample_narrative, sample_posts):
    saturation = scoring_engine.calculate_saturation(sample_narrative, sample_posts)
    assert 0.0 <= saturation <= 1.0
    # 5/10 posts = 50% share of voice. Share capped at 5%, so should be high.
    assert saturation > 0.5

def test_update_narrative_scores(scoring_engine, sample_narrative, sample_posts):
    narratives = [sample_narrative]
    updated = scoring_engine.update_narrative_scores(narratives, sample_posts)
    
    assert len(updated) == 1
    assert updated[0].velocity_score > 0
    assert updated[0].saturation_score > 0
    assert updated[0].id == sample_narrative.id
