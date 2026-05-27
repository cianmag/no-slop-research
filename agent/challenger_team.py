"""
Phase 3b: Challenger Team (Team B) — Takes the research profile and actively
tries to BREAK it. Looks for missing perspectives, biased framing,
cherry-picked data, contradicting evidence, logical gaps, and unstated assumptions.

UPDATED: Now makes direct LLM API calls + noise filtering on output.
"""

import re
from .llm_client import LLMClient
from .noise_filter import filter_improvement_points, severity_score


class ChallengerTeam:
    """Team B — challenges research findings and identifies flaws."""

    def __init__(self, config: dict = None):
        self.config = config or {}

    def challenge(self, topic: str, profile: str, round_num: int,
                  llm_client: LLMClient = None) -> str:
        """
        Run adversarial challenge on a research profile.
        Returns challenge analysis text.
        """
        if llm_client:
            return self._challenge_with_llm(topic, profile, round_num, llm_client)
        else:
            return self._challenge_fallback(topic, profile, round_num)

    def _challenge_with_llm(self, topic: str, profile: str, round_num: int,
                             llm_client: LLMClient) -> str:
        """Use LLM to challenge the research profile."""
        system = (
            "You are a CHALLENGER — a professional skeptic and devil's advocate. "
            "Your job is to take a Research Profile and try to DESTROY it. "
            "Find every flaw, gap, bias, and weakness. Be ruthless but constructive. "
            "Every improvement point you suggest must be SPECIFIC and ACTIONABLE."
        )

        user_msg = f"""TOPIC: {topic}
CHALLENGE ROUND: {round_num}

RESEARCH PROFILE:
{profile[:12000]}

YOUR MISSION:
This research profile claims to be comprehensive and accurate. YOUR JOB IS TO PROVE IT WRONG.

Attack from these angles:

1. MISSING PERSPECTIVES
   - Whose voice is absent? Who benefits from this framing?
   - What geographic, demographic, or ideological viewpoints are missing?

2. BIASED FRAMING
   - Is the research cherry-picking favorable data?
   - Are conclusions stronger than the evidence warrants?
   - Is there confirmation bias in source selection?

3. CONTRADICTING EVIDENCE
   - What evidence exists that contradicts the profile's claims?
   - Are there credible sources that tell a different story?

4. LOGICAL GAPS
   - Where does the reasoning jump from evidence to conclusion?
   - What assumptions are being made without evidence?

5. STALENESS & CONTEXT
   - Is any data outdated?
   - Has the landscape changed since sources were published?

6. ADVERSARIAL SCENARIOS
   - What would a critic say about each major finding?
   - If you had to debunk this in a debate, what would you use?

OUTPUT FORMAT:

## CHALLENGE SUMMARY
Overall assessment: How vulnerable is this research?
Rate: BULLETPROOF / MINOR GAPS / SIGNIFICANT FLAWS / CRITICALLY WEAK

## ATTACK SURFACE
For each weakness found:
- **Weakness:** [what's wrong]
- **Severity:** [Critical / Major / Minor]
- **Evidence:** [what contradicts or undermines this]
- **Fix Required:** [specific action to address it]

## IMPROVEMENT POINTS
Numbered list of SPECIFIC, ACTIONABLE improvements needed.
Each point should be concrete enough that a researcher could execute it.

Format each improvement point as:
[IMPROVE-1] <description of what needs to be fixed and how>
[IMPROVE-2] <description>
...

IMPORTANT: Every improvement point must be SPECIFIC and ACTIONABLE.
Bad: "More research is needed" (too vague — this is noise)
Good: "Add Q3 2024 revenue data from SEC filings for Company X to support the growth claim"

## CHALLENGE VERDICT
What's the single biggest vulnerability in this research?
What would make this 10x stronger?

Be brutal. Be specific. Be constructive. Your goal is to make this research UNBREAKABLE."""

        result = llm_client.chat(
            messages=[{"role": "user", "content": user_msg}],
            system=system,
            temperature=0.4,
            max_tokens=4096
        )

        if result["success"]:
            return result["content"]
        else:
            return self._challenge_fallback(topic, profile, round_num)

    def _challenge_fallback(self, topic: str, profile: str,
                             round_num: int) -> str:
        """Fallback challenge without LLM."""
        return f"""## CHALLENGE SUMMARY
No LLM client configured — cannot perform automated adversarial challenge.
Rate: UNKNOWN

## IMPROVEMENT POINTS
[IMPROVE-1] Configure an LLM API key in the dashboard to enable adversarial challenge

## CHALLENGE VERDICT
The research cannot be validated without an LLM to interrogate it.
Round: {round_num}
"""

    def extract_improvement_points(self, challenge_result: str,
                                    filter_noise: bool = True) -> list:
        """
        Extract numbered improvement points from the challenge result.
        With noise filtering enabled (default), removes vague/duplicate points.

        Returns list of improvement point strings.
        """
        raw_points = self._extract_raw_points(challenge_result)

        if filter_noise and raw_points:
            filtered = filter_improvement_points(raw_points, topic="")
            return filtered["filtered_points"]

        return raw_points

    def _extract_raw_points(self, challenge_result: str) -> list:
        """Extract raw improvement points from text."""
        points = []

        # Look for [IMPROVE-N] pattern
        pattern = r'\[IMPROVE-\d+\]\s*(.+?)(?=\[IMPROVE-|\Z)'
        matches = re.findall(pattern, challenge_result, re.DOTALL)
        if matches:
            for m in matches:
                point = m.strip().split("\n")[0].strip()
                if point:
                    points.append(point)
            return points

        # Fallback: look for numbered improvement points
        lines = challenge_result.split("\n")
        in_improvements = False
        for line in lines:
            if "improvement point" in line.lower() or "improvements needed" in line.lower():
                in_improvements = True
                continue
            if in_improvements:
                line_stripped = line.strip()
                if line_stripped and (line_stripped[0].isdigit() or
                                       line_stripped.startswith("-") or
                                       line_stripped.startswith("*")):
                    point = line_stripped.lstrip("0123456789.-*) \t").strip()
                    if point and len(point) > 10:
                        points.append(point)
                elif line_stripped.startswith("##") and points:
                    break

        return points

    def assess_vulnerability(self, challenge_result: str) -> str:
        """Extract the overall vulnerability assessment."""
        result_lower = challenge_result.lower()
        if "bulletproof" in result_lower:
            return "bulletproof"
        elif "critically weak" in result_lower:
            return "critically_weak"
        elif "significant flaw" in result_lower:
            return "significant_flaws"
        elif "minor gap" in result_lower:
            return "minor_gaps"
        return "unknown"
