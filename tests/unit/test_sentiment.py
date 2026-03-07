"""Tests for sentiment analysis module."""

from datetime import datetime, timezone

from narrativealpha.analysis import SentimentAnalyzer
from narrativealpha.models import Narrative, SocialPost


def _post(
    post_id: str, text: str, likes: int = 0, replies: int = 0, reposts: int = 0
) -> SocialPost:
    return SocialPost(
        id=post_id,
        platform="twitter",
        author_id=f"a-{post_id}",
        author_username=f"u-{post_id}",
        text=text,
        created_at=datetime.now(timezone.utc),
        likes=likes,
        replies=replies,
        reposts=reposts,
    )


class TestSentimentAnalyzer:
    def test_scores_bullish_text_positive(self):
        analyzer = SentimentAnalyzer()
        result = analyzer.score_text("Strong breakout and rally with huge inflow demand")

        assert result.score > 0
        assert result.label == "bullish"
        assert result.positive_hits >= 2

    def test_scores_bearish_text_negative(self):
        analyzer = SentimentAnalyzer()
        result = analyzer.score_text("Major crash risk with panic selling and heavy outflow")

        assert result.score < 0
        assert result.label == "bearish"
        assert result.negative_hits >= 2

    def test_handles_negation(self):
        analyzer = SentimentAnalyzer()
        result = analyzer.score_text("not bullish anymore, this setup looks weak")

        assert result.score < 0
        assert result.negative_hits >= 1

    def test_aggregates_narrative_sentiment_with_engagement_weight(self):
        analyzer = SentimentAnalyzer()
        posts = [
            _post("1", "Bullish breakout and strong demand", likes=300),
            _post("2", "panic crash and sell pressure", likes=0),
        ]
        narrative = Narrative(
            id="nar_1",
            name="Test",
            description="Test narrative",
            post_ids=["1", "2"],
        )

        ns = analyzer.aggregate_for_narrative(narrative, {p.id: p for p in posts})

        assert ns.sample_size == 2
        assert ns.score > 0  # high-engagement bullish post should dominate
        assert ns.label in {"bullish", "neutral"}

    def test_apply_to_narratives_sets_sentiment_and_overall_score(self):
        analyzer = SentimentAnalyzer()
        posts = [_post("1", "bullish rally"), _post("2", "bullish breakout")]
        narrative = Narrative(
            id="nar_2",
            name="BTC momentum",
            description="Cluster",
            post_ids=["1", "2"],
            velocity_score=0.4,
            saturation_score=0.2,
        )

        updated, sentiment_map = analyzer.apply_to_narratives([narrative], posts)

        assert len(updated) == 1
        assert updated[0].sentiment_score > 0
        assert updated[0].overall_score != 0
        assert sentiment_map["nar_2"].label in {"bullish", "neutral"}
