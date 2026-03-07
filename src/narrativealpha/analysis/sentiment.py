"""Sentiment analysis module for market narrative signals."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

from narrativealpha.models import Narrative, SocialPost

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]{1,}")

_POSITIVE_WORDS = {
    "accumulate",
    "adoption",
    "beat",
    "breakout",
    "bull",
    "bullish",
    "buy",
    "catalyst",
    "decentralized",
    "demand",
    "dominance",
    "gain",
    "growth",
    "higher",
    "inflow",
    "long",
    "moon",
    "outperform",
    "partnership",
    "pump",
    "rally",
    "record",
    "recover",
    "resilient",
    "reversal",
    "rise",
    "squeeze",
    "strong",
    "support",
    "surge",
    "upgrade",
    "uptrend",
    "win",
}

_NEGATIVE_WORDS = {
    "ban",
    "bear",
    "bearish",
    "breakdown",
    "crackdown",
    "crash",
    "cut",
    "dump",
    "fear",
    "fraud",
    "hack",
    "higher-for-longer",
    "lawsuit",
    "liquidation",
    "loss",
    "lower",
    "miss",
    "outflow",
    "panic",
    "plunge",
    "recession",
    "reject",
    "resistance",
    "risk-off",
    "rug",
    "sell",
    "short",
    "slump",
    "slowdown",
    "uncertain",
    "volatility",
    "weak",
    "wipeout",
}

_NEGATIONS = {"not", "never", "no", "isnt", "wasnt", "dont", "didnt", "cant", "wont"}


@dataclass(frozen=True)
class SentimentResult:
    """Result of sentiment scoring for a single text body."""

    score: float
    positive_hits: int
    negative_hits: int

    @property
    def label(self) -> str:
        if self.score > 0.2:
            return "bullish"
        if self.score < -0.2:
            return "bearish"
        return "neutral"


@dataclass(frozen=True)
class NarrativeSentiment:
    """Aggregate sentiment output for a clustered narrative."""

    narrative_id: str
    score: float
    label: str
    sample_size: int


class SentimentAnalyzer:
    """Rule-based sentiment scoring tuned for finance and crypto language."""

    def __init__(
        self, positive_words: set[str] | None = None, negative_words: set[str] | None = None
    ):
        self.positive_words = positive_words or _POSITIVE_WORDS
        self.negative_words = negative_words or _NEGATIVE_WORDS

    def score_text(self, text: str) -> SentimentResult:
        """Score free-form text on a normalized scale [-1, 1]."""
        tokens = _tokenize(text)
        if not tokens:
            return SentimentResult(score=0.0, positive_hits=0, negative_hits=0)

        pos_hits = 0
        neg_hits = 0

        for idx, token in enumerate(tokens):
            is_negated = idx > 0 and tokens[idx - 1] in _NEGATIONS
            if token in self.positive_words:
                if is_negated:
                    neg_hits += 1
                else:
                    pos_hits += 1
            elif token in self.negative_words:
                if is_negated:
                    pos_hits += 1
                else:
                    neg_hits += 1

        total_hits = pos_hits + neg_hits
        if total_hits == 0:
            return SentimentResult(score=0.0, positive_hits=0, negative_hits=0)

        raw_score = (pos_hits - neg_hits) / total_hits
        dampened = raw_score * min(1.0, total_hits / 3)
        return SentimentResult(
            score=round(dampened, 4), positive_hits=pos_hits, negative_hits=neg_hits
        )

    def score_posts(self, posts: list[SocialPost]) -> dict[str, SentimentResult]:
        """Score each post by id."""
        return {post.id: self.score_text(post.text) for post in posts}

    def aggregate_for_narrative(
        self, narrative: Narrative, post_lookup: dict[str, SocialPost]
    ) -> NarrativeSentiment:
        """Aggregate per-post sentiment into a narrative-level signal."""
        post_scores: list[float] = []
        post_weights: list[float] = []

        for post_id in narrative.post_ids:
            post = post_lookup.get(post_id)
            if not post:
                continue
            result = self.score_text(post.text)
            weight = 1.0 + _engagement_weight(post)
            post_scores.append(result.score * weight)
            post_weights.append(weight)

        if not post_scores:
            score = 0.0
        else:
            score = round(sum(post_scores) / sum(post_weights), 4)

        label = "bullish" if score > 0.2 else "bearish" if score < -0.2 else "neutral"
        return NarrativeSentiment(
            narrative_id=narrative.id,
            score=score,
            label=label,
            sample_size=len(post_scores),
        )

    def apply_to_narratives(
        self, narratives: list[Narrative], posts: list[SocialPost]
    ) -> tuple[list[Narrative], dict[str, NarrativeSentiment]]:
        """Return copied narratives with sentiment_score/overall_score populated."""
        post_lookup = {post.id: post for post in posts}
        sentiment_map: dict[str, NarrativeSentiment] = {}
        updated: list[Narrative] = []

        for narrative in narratives:
            ns = self.aggregate_for_narrative(narrative, post_lookup)
            sentiment_map[narrative.id] = ns
            overall_score = round(
                (ns.score * 0.4)
                + (narrative.velocity_score * 0.35)
                + (narrative.saturation_score * 0.25),
                4,
            )
            updated.append(
                narrative.model_copy(
                    update={
                        "sentiment_score": ns.score,
                        "overall_score": overall_score,
                    }
                )
            )

        return updated, sentiment_map

    def explain_top_terms(
        self, posts: list[SocialPost], limit: int = 8
    ) -> dict[str, list[tuple[str, int]]]:
        """Expose frequent positive/negative tokens for debugging and reports."""
        pos_counter: Counter[str] = Counter()
        neg_counter: Counter[str] = Counter()

        for post in posts:
            tokens = _tokenize(post.text)
            for idx, token in enumerate(tokens):
                is_negated = idx > 0 and tokens[idx - 1] in _NEGATIONS
                if token in self.positive_words:
                    (neg_counter if is_negated else pos_counter)[token] += 1
                elif token in self.negative_words:
                    (pos_counter if is_negated else neg_counter)[token] += 1

        return {
            "positive": pos_counter.most_common(limit),
            "negative": neg_counter.most_common(limit),
        }


def _tokenize(text: str) -> list[str]:
    return [token.lower() for token in _WORD_RE.findall(text)]


def _engagement_weight(post: SocialPost) -> float:
    engagement = max(0, post.likes) + max(0, post.replies) + max(0, post.reposts)
    return min(2.0, engagement / 100)
