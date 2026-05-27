"""
No-Slop Research — Evidence-Based Confidence Scorer

Replaces keyword-matching with structured scoring based on
source quality, evidence chains, and corroboration.
"""

import re
from typing import Optional


def score_confidence(validation_result: str, research_profile: str,
                     challenge_result: str = "") -> dict:
    """
    Calculate evidence-based confidence scores from validation output.

    Instead of keyword matching ("high confidence" → 0.85), this:
    1. Counts source citations and quality indicators
    2. Measures corroboration (multiple sources agreeing)
    3. Penalizes for known weaknesses from challenger
    4. Weighs evidence strength descriptors with context

    Returns:
        {
            "overall_score": float,        # 0.0 - 1.0
            "confidence_label": str,       # high/moderate/low
            "source_quality": float,       # 0.0 - 1.0
            "corroboration": float,        # 0.0 - 1.0
            "weakness_penalty": float,     # 0.0 - 1.0 (deduction)
            "claim_scores": [...],         # Per-claim breakdown
            "methodology": str,            # How score was calculated
        }
    """
    # 1. Source Quality Score
    source_score = _score_source_quality(validation_result, research_profile)

    # 2. Corroboration Score
    corroboration_score = _score_corroboration(validation_result)

    # 3. Weakness Penalty (from challenger findings)
    weakness_penalty = _score_weaknesses(challenge_result)

    # 4. Evidence Strength Descriptors (weighted, not just keyword matched)
    evidence_score = _score_evidence_strength(validation_result)

    # Combine scores with weights
    raw_score = (
        source_score * 0.30 +        # Source quality matters most
        corroboration_score * 0.25 +  # Multiple agreeing sources
        evidence_score * 0.30 +       # Evidence chain strength
        (1.0 - weakness_penalty) * 0.15  # Penalty for known weaknesses
    )

    # Clamp to 0-1
    overall = max(0.0, min(1.0, raw_score))

    # Label
    if overall >= 0.75:
        label = "high"
    elif overall >= 0.50:
        label = "moderate"
    else:
        label = "low"

    # Per-claim scores
    claim_scores = _extract_claim_scores(validation_result)

    return {
        "overall_score": round(overall, 3),
        "confidence_label": label,
        "source_quality": round(source_score, 3),
        "corroboration": round(corroboration_score, 3),
        "weakness_penalty": round(weakness_penalty, 3),
        "claim_scores": claim_scores,
        "methodology": (
            "Weighted composite: source_quality(30%) + corroboration(25%) "
            "+ evidence_strength(30%) + (1-weakness_penalty)(15%)"
        )
    }


def _score_source_quality(validation: str, profile: str) -> float:
    """Score based on number and quality of cited sources."""
    combined = (validation + " " + profile).lower()

    # Count URLs (proxy for source count)
    urls = re.findall(r'https?://[^\s\)\]\"\']+', combined)
    unique_domains = set()
    for url in urls:
        domain_match = re.search(r'https?://([^/]+)', url)
        if domain_match:
            unique_domains.add(domain_match.group(1).lower())

    # Source count score (diminishing returns)
    num_sources = len(unique_domains)
    if num_sources >= 10:
        source_count_score = 1.0
    elif num_sources >= 5:
        source_count_score = 0.7 + (num_sources - 5) * 0.06
    elif num_sources >= 2:
        source_count_score = 0.4 + (num_sources - 2) * 0.1
    elif num_sources >= 1:
        source_count_score = 0.2
    else:
        source_count_score = 0.0

    # Quality indicators
    quality_bonus = 0.0
    quality_patterns = [
        (r'\b(peer.?reviewed|journal|academic|university|institute)\b', 0.15),
        (r'\b(government|official|gov\.|\.gov|official data)\b', 0.10),
        (r'\b(primary source|first.?hand|original data|raw data)\b', 0.10),
        (r'\b(statistics|dataset|survey|study|research)\b.{0,30}\b(sample|n=|participants)\b', 0.10),
        (r'\b(published|report|whitepaper|paper)\b', 0.05),
    ]
    for pattern, bonus in quality_patterns:
        if re.search(pattern, combined):
            quality_bonus += bonus

    return min(1.0, source_count_score + quality_bonus)


def _score_corroboration(validation: str) -> float:
    """Score based on how many claims are corroborated by multiple sources."""
    v_lower = validation.lower()

    # Count corroboration signals
    multi_source = len(re.findall(
        r'\b(multiple|several|many|numerous|all)\s+(sources?|studies?|reports?|data)\s+(agree|confirm|support|show|indicate)\b',
        v_lower
    ))
    cross_ref = len(re.findall(
        r'\b(cross.?ref|corroborat|confirm|verif|consistent)\b',
        v_lower
    ))
    contradiction = len(re.findall(
        r'\b(contradict|conflict|disagree|inconsisten|discrepanc)\b',
        v_lower
    ))

    # Positive signals
    positive = min(1.0, (multi_source * 0.25) + (cross_ref * 0.1))

    # Negative signals (contradictions reduce confidence)
    negative = min(0.5, contradiction * 0.15)

    return max(0.0, positive - negative)


def _score_weaknesses(challenge: str) -> float:
    """Score weakness severity from challenger output. Higher = more weaknesses."""
    if not challenge:
        return 0.0

    c_lower = challenge.lower()

    # Count severity indicators
    critical = len(re.findall(r'\b(critical|critically|severe|fundamental|fatal)\b', c_lower))
    major = len(re.findall(r'\b(major|significant|substantial|important)\b', c_lower))
    minor = len(re.findall(r'\b(minor|small|trivial|cosmetic|nitpick)\b', c_lower))

    # Weighted penalty
    penalty = (critical * 0.20) + (major * 0.10) + (minor * 0.03)

    # Check overall rating
    if "critically weak" in c_lower:
        penalty += 0.4
    elif "significant flaw" in c_lower:
        penalty += 0.25
    elif "minor gap" in c_lower:
        penalty += 0.10
    elif "bulletproof" in c_lower:
        penalty = max(0, penalty - 0.2)

    return min(1.0, penalty)


def _score_evidence_strength(validation: str) -> float:
    """Score evidence strength from validation output."""
    v_lower = validation.lower()

    # Count strength indicators
    strong = len(re.findall(r'\b(strong|robust|solid|compelling|conclusive|definitive)\b.{0,20}\b(evidence|support|data|proof)\b', v_lower))
    moderate = len(re.findall(r'\b(moderate|reasonable|adequate|decent|partial)\b.{0,20}\b(evidence|support|data)\b', v_lower))
    weak = len(re.findall(r'\b(weak|insufficient|lacking|thin|no evidence|unsubstantiated)\b', v_lower))

    if strong + moderate + weak == 0:
        return 0.5  # Neutral if no strength descriptors

    total = strong + moderate + weak
    score = ((strong * 1.0) + (moderate * 0.6) + (weak * 0.1)) / total
    return score


def _extract_claim_scores(validation: str) -> list:
    """Extract per-claim confidence ratings from validation text."""
    claims = []
    lines = validation.split("\n")
    current_claim = None
    current_evidence = None

    for line in lines:
        line_stripped = line.strip()
        line_lower = line_stripped.lower()

        # Detect claim
        if "**claim:**" in line_lower or "claim:" in line_lower:
            current_claim = line_stripped.split(":", 1)[-1].strip().strip("*").strip()
        elif "evidence quality" in line_lower or "evidence strength" in line_lower:
            if "strong" in line_lower:
                current_evidence = "strong"
            elif "moderate" in line_lower:
                current_evidence = "moderate"
            elif "weak" in line_lower:
                current_evidence = "weak"
        elif "confidence rating" in line_lower and current_claim:
            pct_match = re.search(r'(\d+)%', line_stripped)
            if pct_match:
                score = int(pct_match.group(1)) / 100.0
            elif "high" in line_lower:
                score = 0.8
            elif "moderate" in line_lower:
                score = 0.55
            elif "low" in line_lower:
                score = 0.3
            else:
                score = 0.5

            claims.append({
                "claim": current_claim,
                "score": score,
                "evidence": current_evidence or "unspecified"
            })
            current_claim = None
            current_evidence = None

    return claims
