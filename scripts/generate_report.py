#!/usr/bin/env python3
"""Generate narrative intelligence reports."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from narrativealpha.reports import ReportGenerator


def main():
    parser = argparse.ArgumentParser(
        description="Generate narrative intelligence reports"
    )
    parser.add_argument(
        "--db",
        type=str,
        default="data/narrativealpha.db",
        help="Path to SQLite database (default: data/narrativealpha.db)",
    )
    parser.add_argument(
        "--hours",
        type=int,
        default=24,
        help="Lookback period in hours (default: 24)",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.0,
        help="Minimum overall score to include (default: 0.0)",
    )
    parser.add_argument(
        "--format",
        choices=["markdown", "json"],
        default="markdown",
        help="Output format (default: markdown)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file (default: stdout)",
    )
    parser.add_argument(
        "--daily",
        action="store_true",
        help="Generate daily report (last 24 hours)",
    )
    parser.add_argument(
        "--weekly",
        action="store_true",
        help="Generate weekly report (last 7 days)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    args = parser.parse_args()

    # Adjust hours based on preset flags
    if args.weekly:
        args.hours = 24 * 7
    elif args.daily:
        args.hours = 24

    if args.verbose:
        print(f"Generating report for last {args.hours} hours...")
        print(f"Database: {args.db}")

    # Generate report
    generator = ReportGenerator(db_path=args.db)
    report = generator.generate_report(
        hours=args.hours,
        min_score=args.min_score,
    )

    # Format output
    if args.format == "json":
        output = report.to_json()
        import json
        content = json.dumps(output, indent=2)
    else:
        content = report.to_markdown()

    # Output
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(content)
        if args.verbose:
            print(f"Report written to: {args.output}")
    else:
        print(content)

    # Summary stats
    if args.verbose:
        print(f"\nSummary:")
        print(f"  Total narratives: {report.total_narratives}")
        print(f"  Active: {report.active_narratives}")
        print(f"  Emerging: {report.emerging_narratives}")
        print(f"  Declining: {report.declining_narratives}")


if __name__ == "__main__":
    main()
