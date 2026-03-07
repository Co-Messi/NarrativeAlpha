#!/usr/bin/env python3
"""Analyze sentiment for clustered narratives from recent ingested posts."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone

from narrativealpha.analysis import NarrativeClusteringEngine, SentimentAnalyzer
from narrativealpha.ingestion.storage import SocialPostStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze sentiment for detected narratives")
    parser.add_argument("--limit", type=int, default=200, help="Max posts to load")
    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=2,
        help="Minimum posts required for a narrative cluster",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    storage = SocialPostStore()
    posts = storage.get_unprocessed(limit=args.limit)
    if not posts:
        print("No posts found. Run ingestion first.")
        return 0

    clustering = NarrativeClusteringEngine(min_cluster_size=args.min_cluster_size)
    narratives = clustering.cluster_posts(posts, now=datetime.now(timezone.utc))
    if not narratives:
        print("No narratives met clustering threshold.")
        return 0

    analyzer = SentimentAnalyzer()
    scored_narratives, sentiment_map = analyzer.apply_to_narratives(narratives, posts)

    print(f"Scored {len(scored_narratives)} narratives from {len(posts)} posts")
    print("=" * 72)

    for narrative in scored_narratives:
        sentiment = sentiment_map[narrative.id]
        print(f"{narrative.name} [{sentiment.label.upper()} {sentiment.score:+.3f}]")
        cashtag_text = ", ".join(narrative.cashtags) or "N/A"
        print(f"  Posts: {len(narrative.post_ids)} | Cashtags: {cashtag_text}")
        print(f"  Keywords: {', '.join(narrative.keywords[:6]) or 'N/A'}")
        print(f"  Overall score: {narrative.overall_score:+.3f}")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
