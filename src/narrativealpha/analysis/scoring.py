"""Velocity and Saturation scoring module for narratives."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Dict, List

from narrativealpha.models import Narrative, SocialPost


class ScoringEngine:
    """Calculates engagement velocity and market saturation for narratives."""

    def __init__(self, time_window_hours: int = 24):
        self.time_window_hours = time_window_hours

    def calculate_velocity(
        self, narrative: Narrative, posts: List[SocialPost]
    ) -> float:
        """
        Calculates narrative velocity based on post frequency and engagement growth.
        Normalized score [0, 1].
        """
        if not posts:
            return 0.0

        now = datetime.now(timezone.utc)
        recent_threshold = now - timedelta(hours=self.time_window_hours)
        
        # Filter posts belonging to this narrative that are within the window
        narrative_post_ids = set(narrative.post_ids)
        recent_posts = [
            p for p in posts 
            if p.id in narrative_post_ids and p.created_at.replace(tzinfo=timezone.utc) > recent_threshold
        ]

        if not recent_posts:
            return 0.0

        # Frequency: posts per hour in the window
        post_count = len(recent_posts)
        frequency = post_count / self.time_window_hours

        # Engagement density: total engagement / number of posts
        total_engagement = sum(
            p.likes + p.replies + p.reposts for p in recent_posts
        )
        engagement_density = total_engagement / post_count

        # Combine: Frequency (60%) + Engagement density (40%)
        # Scaling: Frequency capped at 10 posts/hr, Density capped at 500
        norm_freq = min(1.0, frequency / 10.0)
        norm_density = min(1.0, engagement_density / 500.0)

        velocity = (norm_freq * 0.6) + (norm_density * 0.4)
        return round(velocity, 4)

    def calculate_saturation(
        self, narrative: Narrative, all_posts: List[SocialPost]
    ) -> float:
        """
        Calculates narrative saturation (mindshare). 
        How much of the total market conversation is this narrative occupying?
        Normalized score [0, 1].
        """
        if not all_posts:
            return 0.0

        total_market_posts = len(all_posts)
        narrative_post_count = len(narrative.post_ids)

        # Basic saturation: percentage of total posts
        share_of_voice = narrative_post_count / total_market_posts

        # Diversity: number of unique authors contributing to the narrative
        unique_authors = len(set(p.author_id for p in all_posts if p.id in narrative.post_ids))
        diversity_score = min(1.0, unique_authors / 50.0) # Cap at 50 unique voices

        # Saturation: Share of voice (70%) + Author Diversity (30%)
        # Scale share_of_voice: 5% of total market is considered high saturation (1.0)
        norm_sov = min(1.0, share_of_voice / 0.05)
        
        saturation = (norm_sov * 0.7) + (diversity_score * 0.3)
        return round(saturation, 4)

    def update_narrative_scores(
        self, narratives: List[Narrative], posts: List[SocialPost]
    ) -> List[Narrative]:
        """Calculates and updates velocity and saturation for all narratives."""
        updated_narratives = []
        for narrative in narratives:
            velocity = self.calculate_velocity(narrative, posts)
            saturation = self.calculate_saturation(narrative, posts)
            
            # Note: overall_score update is typically handled by the sentiment module 
            # in this project's architecture (as seen in sentiment.py), 
            # but we update the individual scores here.
            updated_narratives.append(
                narrative.model_copy(
                    update={
                        "velocity_score": velocity,
                        "saturation_score": saturation,
                    }
                )
            )
        return updated_narratives
