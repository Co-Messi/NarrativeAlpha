#!/usr/bin/env python3
"""Cluster unprocessed posts and persist narrative lifecycle events."""

from __future__ import annotations

import argparse

from narrativealpha.analysis.clustering import NarrativeClusteringEngine
from narrativealpha.analysis.tracking import NarrativeTracker
from narrativealpha.ingestion.storage import SocialPostStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Track narrative persistence over time")
    parser.add_argument("--limit", type=int, default=200, help="Max unprocessed posts to load")
    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=2,
        help="Minimum posts required to form a narrative cluster",
    )
    parser.add_argument(
        "--stale-after-hours",
        type=int,
        default=48,
        help="Mark unseen active narratives inactive after N hours",
    )
    parser.add_argument("--db-path", type=str, default=None, help="Override SQLite path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    store = SocialPostStore(db_path=args.db_path)
    tracker = NarrativeTracker(db_path=args.db_path)
    engine = NarrativeClusteringEngine(min_cluster_size=args.min_cluster_size)

    posts = store.get_unprocessed(limit=args.limit)
    if not posts:
        print("No unprocessed posts found.")
        return

    narratives = engine.cluster_posts(posts)
    events = tracker.upsert_narratives(narratives, stale_after_hours=args.stale_after_hours)

    if posts:
        store.mark_processed([p.id for p in posts])

    print(f"Processed posts: {len(posts)}")
    print(f"Narratives clustered: {len(narratives)}")
    if not events:
        print("No lifecycle events generated.")
        return

    summary: dict[str, int] = {}
    for event in events:
        summary[event.event] = summary.get(event.event, 0) + 1

    print("Lifecycle events:")
    for name, count in sorted(summary.items()):
        print(f"  - {name}: {count}")


if __name__ == "__main__":
    main()
