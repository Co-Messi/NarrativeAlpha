"""Narrative persistence and lifecycle tracking."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from narrativealpha.models import Narrative


@dataclass(frozen=True)
class NarrativeTrackEvent:
    """Result of reconciling one narrative against persistence."""

    narrative_id: str
    event: str
    previous_version: int
    new_version: int


class NarrativeTracker:
    """Persist and track narrative lifecycle across runs."""

    def __init__(self, db_path: str | None = None):
        self.db_path = Path(db_path or "data/narrativealpha.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS narratives (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT NOT NULL,
                    first_seen TEXT NOT NULL,
                    last_seen TEXT NOT NULL,
                    post_ids TEXT NOT NULL,
                    cashtags TEXT NOT NULL,
                    keywords TEXT NOT NULL,
                    sentiment_score REAL NOT NULL DEFAULT 0.0,
                    velocity_score REAL NOT NULL DEFAULT 0.0,
                    saturation_score REAL NOT NULL DEFAULT 0.0,
                    overall_score REAL NOT NULL DEFAULT 0.0,
                    is_active INTEGER NOT NULL DEFAULT 1,
                    confidence REAL NOT NULL DEFAULT 0.0,
                    version INTEGER NOT NULL DEFAULT 1,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS narrative_events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    narrative_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    previous_version INTEGER NOT NULL,
                    new_version INTEGER NOT NULL,
                    occurred_at TEXT NOT NULL,
                    details TEXT,
                    FOREIGN KEY(narrative_id) REFERENCES narratives(id)
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_narratives_active ON narratives(is_active)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_narratives_last_seen ON narratives(last_seen)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_narrative_events_narrative ON narrative_events(narrative_id)"
            )
            conn.commit()

    def upsert_narratives(
        self,
        narratives: list[Narrative],
        now: datetime | None = None,
        stale_after_hours: int = 48,
    ) -> list[NarrativeTrackEvent]:
        """Persist cluster output and emit lifecycle events.

        Event types:
        - created: first time narrative appears
        - updated: known narrative changed materially
        - reactivated: was inactive and appears again
        - unchanged: no material changes
        - deactivated: not seen for configured stale window
        """
        events: list[NarrativeTrackEvent] = []
        now_dt = now or datetime.utcnow()
        now_iso = now_dt.isoformat()

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            seen_ids = {n.id for n in narratives}

            for narrative in narratives:
                row = conn.execute(
                    "SELECT * FROM narratives WHERE id = ?", (narrative.id,)
                ).fetchone()

                if row is None:
                    conn.execute(
                        """
                        INSERT INTO narratives (
                            id, name, description, first_seen, last_seen, post_ids,
                            cashtags, keywords, sentiment_score, velocity_score,
                            saturation_score, overall_score, is_active, confidence,
                            version, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        self._to_db_tuple(narrative, version=1, updated_at=now_iso),
                    )
                    self._insert_event(
                        conn,
                        narrative_id=narrative.id,
                        event_type="created",
                        previous_version=0,
                        new_version=1,
                        occurred_at=now_iso,
                        details={"name": narrative.name},
                    )
                    events.append(
                        NarrativeTrackEvent(
                            narrative_id=narrative.id,
                            event="created",
                            previous_version=0,
                            new_version=1,
                        )
                    )
                    continue

                prev_version = int(row["version"])
                was_active = bool(row["is_active"])
                changed = self._materially_changed(row, narrative)
                event_type = "unchanged"
                new_version = prev_version

                if not was_active:
                    event_type = "reactivated"
                    new_version = prev_version + 1
                elif changed:
                    event_type = "updated"
                    new_version = prev_version + 1

                conn.execute(
                    """
                    UPDATE narratives
                    SET name = ?, description = ?, first_seen = ?, last_seen = ?,
                        post_ids = ?, cashtags = ?, keywords = ?, sentiment_score = ?,
                        velocity_score = ?, saturation_score = ?, overall_score = ?,
                        is_active = 1, confidence = ?, version = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (
                        narrative.name,
                        narrative.description,
                        narrative.first_seen.isoformat(),
                        narrative.last_seen.isoformat(),
                        json.dumps(sorted(narrative.post_ids)),
                        json.dumps(sorted(narrative.cashtags)),
                        json.dumps(sorted(narrative.keywords)),
                        narrative.sentiment_score,
                        narrative.velocity_score,
                        narrative.saturation_score,
                        narrative.overall_score,
                        narrative.confidence,
                        new_version,
                        now_iso,
                        narrative.id,
                    ),
                )

                if event_type != "unchanged":
                    self._insert_event(
                        conn,
                        narrative_id=narrative.id,
                        event_type=event_type,
                        previous_version=prev_version,
                        new_version=new_version,
                        occurred_at=now_iso,
                        details={"name": narrative.name},
                    )

                events.append(
                    NarrativeTrackEvent(
                        narrative_id=narrative.id,
                        event=event_type,
                        previous_version=prev_version,
                        new_version=new_version,
                    )
                )

            stale_cutoff = (now_dt - timedelta(hours=stale_after_hours)).isoformat()
            stale_rows = conn.execute(
                """
                SELECT id, version FROM narratives
                WHERE is_active = 1 AND id NOT IN (
                    SELECT id FROM narratives WHERE id IN (%s)
                ) AND last_seen < ?
                """
                % (",".join("?" * len(seen_ids)) if seen_ids else "''"),
                ((*seen_ids, stale_cutoff) if seen_ids else (stale_cutoff,)),
            ).fetchall()

            for stale in stale_rows:
                prev_version = int(stale["version"])
                new_version = prev_version + 1
                conn.execute(
                    "UPDATE narratives SET is_active = 0, version = ?, updated_at = ? WHERE id = ?",
                    (new_version, now_iso, stale["id"]),
                )
                self._insert_event(
                    conn,
                    narrative_id=stale["id"],
                    event_type="deactivated",
                    previous_version=prev_version,
                    new_version=new_version,
                    occurred_at=now_iso,
                    details={},
                )
                events.append(
                    NarrativeTrackEvent(
                        narrative_id=stale["id"],
                        event="deactivated",
                        previous_version=prev_version,
                        new_version=new_version,
                    )
                )

            conn.commit()

        return events

    def list_active(self) -> list[Narrative]:
        """Return currently active narratives."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM narratives WHERE is_active = 1 ORDER BY overall_score DESC, confidence DESC"
            ).fetchall()
            return [self._row_to_narrative(row) for row in rows]

    def _to_db_tuple(self, narrative: Narrative, version: int, updated_at: str) -> tuple:
        return (
            narrative.id,
            narrative.name,
            narrative.description,
            narrative.first_seen.isoformat(),
            narrative.last_seen.isoformat(),
            json.dumps(sorted(narrative.post_ids)),
            json.dumps(sorted(narrative.cashtags)),
            json.dumps(sorted(narrative.keywords)),
            narrative.sentiment_score,
            narrative.velocity_score,
            narrative.saturation_score,
            narrative.overall_score,
            int(narrative.is_active),
            narrative.confidence,
            version,
            updated_at,
        )

    def _materially_changed(self, row: sqlite3.Row, narrative: Narrative) -> bool:
        if row["name"] != narrative.name or row["description"] != narrative.description:
            return True

        if set(json.loads(row["post_ids"])) != set(narrative.post_ids):
            return True
        if set(json.loads(row["cashtags"])) != set(narrative.cashtags):
            return True
        if set(json.loads(row["keywords"])) != set(narrative.keywords):
            return True

        metric_fields = [
            "sentiment_score",
            "velocity_score",
            "saturation_score",
            "overall_score",
            "confidence",
        ]
        for field in metric_fields:
            if abs(float(row[field]) - float(getattr(narrative, field))) > 1e-6:
                return True
        return False

    def _insert_event(
        self,
        conn: sqlite3.Connection,
        narrative_id: str,
        event_type: str,
        previous_version: int,
        new_version: int,
        occurred_at: str,
        details: dict,
    ) -> None:
        conn.execute(
            """
            INSERT INTO narrative_events (
                narrative_id, event_type, previous_version, new_version, occurred_at, details
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                narrative_id,
                event_type,
                previous_version,
                new_version,
                occurred_at,
                json.dumps(details),
            ),
        )

    def _row_to_narrative(self, row: sqlite3.Row) -> Narrative:
        return Narrative(
            id=row["id"],
            name=row["name"],
            description=row["description"],
            first_seen=datetime.fromisoformat(row["first_seen"]),
            last_seen=datetime.fromisoformat(row["last_seen"]),
            post_ids=json.loads(row["post_ids"]),
            cashtags=json.loads(row["cashtags"]),
            keywords=json.loads(row["keywords"]),
            sentiment_score=float(row["sentiment_score"]),
            velocity_score=float(row["velocity_score"]),
            saturation_score=float(row["saturation_score"]),
            overall_score=float(row["overall_score"]),
            is_active=bool(row["is_active"]),
            confidence=float(row["confidence"]),
        )
