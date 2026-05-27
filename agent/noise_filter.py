"""
No-Slop Research — Noise Filter

Filters, deduplicates, and scores improvement points from the Challenger Team.
Prevents garbage improvement points from wasting rounds.
"""

import re
from difflib import SequenceMatcher


def filter_improvement_points(points: list, topic: str,
                               min_length: int = 20,
                               similarity_threshold: float = 0.7) -> dict:
    """
    Filter and score improvement points from Team B.

    Args:
        points: Raw improvement point strings from challenger
        topic: The research topic (for relevance scoring)
        min_length: Minimum character length for a valid point
        similarity_threshold: Above this, points are considered duplicates

    Returns:
        {
            "filtered_points": [...],      # Clean, deduplicated points
            "removed_noise": [...],         # Points that were filtered out
            "removed_reasons": {...],       # Why each was removed
            "quality_score": float,         # 0-1 overall quality of Team B output
        }
    """
    if not points:
        return {
            "filtered_points": [],
            "removed_noise": [],
            "removed_reasons": {},
            "quality_score": 0.0
        }

    removed = []
    removed_reasons = {}
    clean = []

    for i, point in enumerate(points):
        point = point.strip()

        # Check 1: Too short / empty
        if len(point) < min_length:
            removed.append(point)
            removed_reasons[point[:50]] = "Too short (< {min_length} chars)"
            continue

        # Check 2: Generic / vague noise
        if _is_generic_noise(point):
            removed.append(point)
            removed_reasons[point[:50]] = "Generic noise — no actionable content"
            continue

        # Check 3: Not actionable (no verb, no specific fix)
        if not _is_actionable(point):
            removed.append(point)
            removed_reasons[point[:50]] = "Not actionable — no specific fix described"
            continue

        # Check 4: Duplicate / near-duplicate
        is_dup = False
        for existing in clean:
            if _similarity(point, existing) > similarity_threshold:
                removed.append(point)
                removed_reasons[point[:50]] = f"Duplicate of: {existing[:60]}..."
                is_dup = True
                break
        if is_dup:
            continue

        clean.append(point)

    # Calculate quality score
    total = len(points)
    if total == 0:
        quality = 0.0
    else:
        quality = len(clean) / total

    return {
        "filtered_points": clean,
        "removed_noise": removed,
        "removed_reasons": removed_reasons,
        "quality_score": round(quality, 2)
    }


def _is_generic_noise(point: str) -> bool:
    """Check if a point is generic filler with no substance."""
    noise_patterns = [
        r"^(more research|further study|additional investigation) (is|are) needed",
        r"^(this|the) (claim|finding|conclusion) (needs|requires) (more|further)",
        r"^consider (adding|including|checking)",
        r"^it would be (useful|helpful|good) to",
        r"^the research (could|might|should) (benefit|improve)",
        r"^(overall|in general|basically)",
        r"^this is (not|barely) (a |an )?(significant|major|important)",
    ]

    point_lower = point.lower().strip()
    for pattern in noise_patterns:
        if re.match(pattern, point_lower):
            return True

    # Check if it's mostly filler words
    words = point_lower.split()
    filler_words = {"the", "a", "an", "is", "are", "was", "were", "be", "been",
                    "being", "have", "has", "had", "do", "does", "did", "will",
                    "would", "could", "should", "may", "might", "must", "shall",
                    "can", "need", "dare", "ought", "used", "to", "of", "in",
                    "for", "on", "with", "at", "by", "from", "as", "into",
                    "through", "during", "before", "after", "above", "below",
                    "between", "out", "off", "over", "under", "again", "further",
                    "then", "once", "more", "also", "very", "really", "quite",
                    "just", "only", "even", "still", "already"}
    filler_ratio = sum(1 for w in words if w in filler_words) / max(len(words), 1)
    if filler_ratio > 0.75 and len(words) > 15:
        return True

    return False


def _is_actionable(point: str) -> bool:
    """Check if a point contains an actionable suggestion."""
    actionable_signals = [
        r"\b(add|include|remove|replace|fix|address|verify|check)\b",
        r"\bcross[- ]?ref",
        r"\b(source|cite|evidence|data|statistic|study|research|figures?|numbers?|metrics?)\b",
        r"\b(missing|absent|lacking|weak|outdated|needs?|requires?|necessary)\b",
        r"\b(bias|framing|assumption|gap|flaw|weakness|contradiction|perspective)\b",
        r"\b(specify|clarify|quantify|measure|compare|contrast|support)\b",
        r"\b(improve|strengthen|update|revise|expand|narrow|back(?:ed)?)\b",
        r"\[IMPROVE-\d+\]",
        r"\b(SEC|FDA|Gartner|IDC|Forrester|McKinsey|report|filing)\b",
    ]

    point_lower = point.lower()
    match_count = sum(1 for pattern in actionable_signals if re.search(pattern, point_lower))

    # Need at least 2 signal matches for a point to be considered actionable
    # (one match could be coincidence, two suggests real substance)
    return match_count >= 2


def _similarity(a: str, b: str) -> float:
    """Calculate string similarity ratio."""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def severity_score(point: str) -> str:
    """
    Rate the severity of an improvement point.
    Returns: critical, major, minor
    """
    critical_signals = [
        r"\b(missing|absent|no |lack).{0,30}(source|evidence|citation|data)\b",
        r"\b(false|incorrect|wrong|inaccurate|misleading)\b",
        r"\b(contradicts?|conflicts?|opposes?)\b",
        r"\b(critically|severely|fundamentally)\b",
        r"\binvalidat",
    ]
    major_signals = [
        r"\b(significant|substantial|major|important)\b",
        r"\b(weak|flawed|biased|cherry.?pick)\b",
        r"\b(outdated|stale|old)\b",
        r"\b(missing|absent|lack).{0,20}(perspective|viewpoint|angle)\b",
    ]

    point_lower = point.lower()
    for pattern in critical_signals:
        if re.search(pattern, point_lower):
            return "critical"
    for pattern in major_signals:
        if re.search(pattern, point_lower):
            return "major"
    return "minor"
