"""Tests for narrative report generation."""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from narrativealpha.reports import NarrativeReport, ReportGenerator


@pytest.fixture
def test_db():
    """Create a temporary test database with sample data."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    # Create tables
    import sqlite3
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS narratives (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            description TEXT,
            first_seen TEXT NOT NULL,
            last_seen TEXT NOT NULL,
            cashtags TEXT,
            keywords TEXT,
            sentiment_score REAL DEFAULT 0,
            velocity_score REAL DEFAULT 0,
            saturation_score REAL DEFAULT 0,
            overall_score REAL DEFAULT 0,
            is_active INTEGER DEFAULT 1,
            confidence REAL DEFAULT 0
        )
    """)
    
    now = datetime.utcnow()
    
    # Insert test data
    narratives = [
        # Active high-scoring narrative
        (1, "Bitcoin ETF", "Bitcoin ETF approval", now.isoformat(), now.isoformat(),
         '["$BTC"]', '["etf", "approval"]', 0.8, 0.9, 0.7, 0.8, 1, 0.9),
        # Active medium-scoring narrative
        (2, "Ethereum Staking", "ETH staking yields", now.isoformat(), now.isoformat(),
         '["$ETH"]', '["staking", "yield"]', 0.5, 0.6, 0.5, 0.5, 1, 0.7),
        # Emerging narrative (recently created)
        (3, "Solana Memecoins", "SOL meme coin frenzy", 
         (now - timedelta(hours=2)).isoformat(), now.isoformat(),
         '["$SOL"]', '["memecoin", "solana"]', 0.9, 0.95, 0.8, 0.9, 1, 0.85),
        # Declining narrative (stale)
        (4, "DeFi Summer", "DeFi Summer 2020", 
         (now - timedelta(days=30)).isoformat(), (now - timedelta(days=2)).isoformat(),
         '["$DEFi"]', '["defi", "summer"]', 0.1, 0.1, 0.2, 0.1, 1, 0.3),
    ]
    
    conn.executemany("""
        INSERT INTO narratives VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, narratives)
    
    conn.commit()
    conn.close()
    
    yield db_path
    
    # Cleanup
    Path(db_path).unlink(missing_ok=True)


class TestNarrativeReport:
    """Tests for NarrativeReport dataclass."""

    def test_to_markdown_basic(self):
        """Test basic markdown generation."""
        report = NarrativeReport(
            generated_at=datetime.utcnow(),
            period_start=datetime.utcnow() - timedelta(hours=24),
            period_end=datetime.utcnow(),
            total_narratives=10,
            active_narratives=8,
            emerging_narratives=2,
            declining_narratives=1,
            top_narratives=[
                {"name": "Test", "overall_score": 0.9, "sentiment_score": 0.5, "velocity_score": 0.8}
            ],
        )
        
        markdown = report.to_markdown()
        
        assert "Narrative Intelligence Report" in markdown
        assert "**Total Narratives Tracked:** 10" in markdown
        assert "**Emerging (new/rapidly growing):** 2" in markdown
        assert "**Declining (losing momentum):** 1" in markdown

    def test_to_json_basic(self):
        """Test basic JSON generation."""
        report = NarrativeReport(
            generated_at=datetime.utcnow(),
            period_start=datetime.utcnow() - timedelta(hours=24),
            period_end=datetime.utcnow(),
            total_narratives=5,
            active_narratives=3,
            emerging_narratives=1,
            declining_narratives=1,
        )
        
        json_output = report.to_json()
        
        assert "generated_at" in json_output
        assert json_output["summary"]["total_narratives"] == 5
        assert json_output["summary"]["active_narratives"] == 3

    def test_to_markdown_with_narratives(self):
        """Test markdown with narrative data."""
        report = NarrativeReport(
            generated_at=datetime.utcnow(),
            period_start=datetime.utcnow() - timedelta(hours=24),
            period_end=datetime.utcnow(),
            total_narratives=2,
            active_narratives=2,
            emerging_narratives=1,
            declining_narratives=0,
            top_narratives=[
                {
                    "name": "Bitcoin ETF",
                    "overall_score": 0.85,
                    "sentiment_score": 0.7,
                    "velocity_score": 0.9,
                },
                {
                    "name": "Ethereum",
                    "overall_score": 0.65,
                    "sentiment_score": -0.2,
                    "velocity_score": 0.5,
                },
            ],
            emerging=[
                {
                    "name": "New Narrative",
                    "velocity_score": 0.95,
                    "overall_score": 0.8,
                }
            ],
            sentiment_distribution={"positive": 3, "negative": 1, "neutral": 2},
        )
        
        markdown = report.to_markdown()
        
        assert "Top Narratives by Score" in markdown
        assert "Bitcoin ETF" in markdown
        assert "Emerging Narratives" in markdown
        assert "New Narrative" in markdown
        assert "Sentiment Distribution" in markdown
        assert "Positive: 3" in markdown


class TestReportGenerator:
    """Tests for ReportGenerator class."""

    def test_generator_init(self, test_db):
        """Test report generator initialization."""
        generator = ReportGenerator(db_path=test_db)
        
        assert generator.db_path.name == Path(test_db).name

    def test_generate_report(self, test_db):
        """Test full report generation."""
        generator = ReportGenerator(db_path=test_db)
        
        report = generator.generate_report(hours=24, min_score=0.0)
        
        assert report.total_narratives > 0
        assert report.generated_at is not None
        assert report.period_start is not None
        assert report.period_end is not None

    def test_generate_report_with_min_score(self, test_db):
        """Test report with minimum score filter."""
        generator = ReportGenerator(db_path=test_db)
        
        report = generator.generate_report(hours=24, min_score=0.7)
        
        # All narratives should have score >= 0.7
        for n in report.top_narratives:
            assert n["overall_score"] >= 0.7

    def test_fetch_emerging(self, test_db):
        """Test emerging narrative detection."""
        generator = ReportGenerator(db_path=test_db)
        
        with generator._connect() as conn:
            emerging = generator._fetch_emerging(conn, datetime.utcnow() - timedelta(hours=24))
        
        assert len(emerging) > 0
        # Emerging should have high velocity
        for n in emerging:
            assert n.get("velocity_score", 0) > 0.5

    def test_fetch_declining(self, test_db):
        """Test declining narrative detection."""
        generator = ReportGenerator(db_path=test_db)
        
        with generator._connect() as conn:
            declining = generator._fetch_declining(conn, datetime.utcnow() - timedelta(hours=24))
        
        assert len(declining) > 0
        # Declining should be inactive/stale
        for n in declining:
            assert n.get("is_active", 1) == 1

    def test_sentiment_distribution(self, test_db):
        """Test sentiment distribution calculation."""
        generator = ReportGenerator(db_path=test_db)
        
        with generator._connect() as conn:
            narratives = generator._fetch_narratives(conn, 0.0)
        
        dist = generator._calculate_sentiment_distribution(narratives)
        
        assert "positive" in dist
        assert "negative" in dist
        assert "neutral" in dist
        assert dist["positive"] + dist["negative"] + dist["neutral"] == len(narratives)

    def test_parse_json_field(self, test_db):
        """Test JSON field parsing."""
        generator = ReportGenerator(db_path=test_db)
        
        # Test with valid JSON
        result = generator._parse_json_field('["$BTC", "$ETH"]')
        assert result == ["$BTC", "$ETH"]
        
        # Test with empty
        result = generator._parse_json_field(None)
        assert result == []
        
        # Test with invalid JSON
        result = generator._parse_json_field("not json")
        assert result == []
