from datetime import datetime, timedelta

from narrativealpha.analysis.tracking import NarrativeTracker
from narrativealpha.models import Narrative


def _nar(
    narrative_id: str,
    name: str,
    post_ids: list[str],
    now: datetime,
    is_active: bool = True,
) -> Narrative:
    return Narrative(
        id=narrative_id,
        name=name,
        description=f"{name} description",
        first_seen=now - timedelta(hours=2),
        last_seen=now,
        post_ids=post_ids,
        cashtags=["BTC"],
        keywords=["etf", "flows"],
        sentiment_score=0.4,
        velocity_score=0.5,
        saturation_score=0.3,
        overall_score=0.47,
        confidence=0.72,
        is_active=is_active,
    )


def test_tracker_creates_and_updates_narrative(tmp_path):
    db_path = tmp_path / "narrativealpha.db"
    tracker = NarrativeTracker(db_path=str(db_path))
    now = datetime(2026, 3, 10, 12, 0, 0)

    created = tracker.upsert_narratives([_nar("nar_1", "BTC ETF", ["p1", "p2"], now)], now=now)
    assert len(created) == 1
    assert created[0].event == "created"
    assert created[0].new_version == 1

    updated_nar = _nar("nar_1", "BTC ETF", ["p1", "p2", "p3"], now + timedelta(minutes=20))
    updated = tracker.upsert_narratives([updated_nar], now=now + timedelta(minutes=20))
    assert len(updated) == 1
    assert updated[0].event == "updated"
    assert updated[0].new_version == 2


def test_tracker_reactivates_inactive_narrative(tmp_path):
    db_path = tmp_path / "narrativealpha.db"
    tracker = NarrativeTracker(db_path=str(db_path))
    now = datetime(2026, 3, 10, 12, 0, 0)

    tracker.upsert_narratives([_nar("nar_2", "ETH narrative", ["a1", "a2"], now)], now=now)
    tracker.upsert_narratives([], now=now + timedelta(hours=72), stale_after_hours=24)

    events = tracker.upsert_narratives(
        [_nar("nar_2", "ETH narrative", ["a1", "a2", "a3"], now + timedelta(hours=73))],
        now=now + timedelta(hours=73),
    )
    assert any(e.event == "reactivated" for e in events)


def test_tracker_deactivates_stale_narrative(tmp_path):
    db_path = tmp_path / "narrativealpha.db"
    tracker = NarrativeTracker(db_path=str(db_path))
    now = datetime(2026, 3, 10, 12, 0, 0)

    tracker.upsert_narratives([_nar("nar_3", "Solana memecoins", ["s1", "s2"], now)], now=now)
    events = tracker.upsert_narratives([], now=now + timedelta(hours=50), stale_after_hours=24)

    assert any(e.event == "deactivated" for e in events)
    active = tracker.list_active()
    assert active == []
