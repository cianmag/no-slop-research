"""
Phase 4: Synthesizer — Merges improvement points from the Challenger Team
back into the research profile, triggering targeted re-research to address weaknesses.

UPDATED: Now makes direct LLM API calls.
"""

from .llm_client import LLMClient


class Synthesizer:
    """Merges improvement points back into the research profile."""

    def merge(self, topic: str, original_profile: str, improvement_points: list,
              validation_result: str, challenge_result: str, round_num: int,
              llm_client: LLMClient = None) -> str:
        """
        Merge improvement points into the profile.

        Args:
            topic: Research topic
            original_profile: Current research profile
            improvement_points: List of improvement point strings from Team B
            validation_result: Team A's validation output
            challenge_result: Team B's challenge output
            round_num: Current adversarial round
            llm_client: LLM client for API calls

        Returns:
            Updated Research Profile text
        """
        if llm_client:
            return self._merge_with_llm(topic, original_profile,
                                         improvement_points, validation_result,
                                         challenge_result, round_num, llm_client)
        else:
            return self._merge_fallback(topic, original_profile,
                                         improvement_points, round_num)

    def _merge_with_llm(self, topic: str, profile: str, improvements: list,
                         validation: str, challenge: str, round_num: int,
                         llm_client: LLMClient) -> str:
        """Use LLM to merge improvements into the profile."""
        improvements_text = "\n".join(f"  {i+1}. {p}" for i, p in enumerate(improvements))

        system = (
            "You are a RESEARCH SYNTHESIZER. Your job is to take a Research Profile, "
            "the adversarial challenge results, and merge improvements back into a "
            "stronger profile. Preserve strong findings; fix or weaken claims that "
            "were successfully challenged."
        )

        user_msg = f"""TOPIC: {topic}
SYNTHESIS ROUND: {round_num}

CURRENT RESEARCH PROFILE:
{profile[:10000]}

VALIDATOR TEAM (Team A) FINDINGS:
{validation[:4000]}

CHALLENGER TEAM (Team B) ATTACK:
{challenge[:4000]}

IMPROVEMENT POINTS TO ADDRESS:
{improvements_text}

YOUR MISSION:
Rewrite the Research Profile to address EVERY improvement point while preserving
the strong findings that survived validation.

PROCESS:
1. Keep findings that the Validator rated as Strong/Moderate evidence
2. For each improvement point, either:
   a. Add missing information/address the gap
   b. Weaken claims that the Challenger proved were overstated
   c. Add caveats and nuance where needed
   d. Flag for further research if the gap can't be filled from existing data
3. Re-structure the profile to be more balanced and comprehensive
4. Update confidence ratings based on both teams' input

OUTPUT FORMAT:
Return the COMPLETE updated Research Profile in the same structure as the original:
- Executive Summary (updated)
- Key Findings (with updated evidence ratings)
- Data Points
- Stakeholder Map
- Contradictions & Tensions (updated)
- Knowledge Gaps (updated — what still needs research)
- Sources (updated)
- Confidence Assessment (updated)

Mark any NEW research areas with [NEEDS-RESEARCH] tags so they can be
investigated in the next round if needed."""

        result = llm_client.chat(
            messages=[{"role": "user", "content": user_msg}],
            system=system,
            temperature=0.3,
            max_tokens=6000
        )

        if result["success"]:
            return result["content"]
        else:
            return self._merge_fallback(topic, profile, improvements, round_num)

    def _merge_fallback(self, topic: str, profile: str, improvements: list,
                         round_num: int) -> str:
        """Fallback merge without LLM."""
        improvements_text = "\n".join(f"- {p}" for p in improvements)
        return f"""{profile}

---

## ADVERSARIAL ROUND {round_num} — IMPROVEMENT POINTS
The following improvement points were identified but NOT addressed
(no LLM client configured for synthesis):

{improvements_text}

[NEEDS-RESEARCH] Configure an API key to enable automatic synthesis.
"""

    def extract_needs_research(self, synthesized_profile: str) -> list:
        """Extract items tagged with [NEEDS-RESEARCH]."""
        import re
        pattern = r'\[NEEDS-RESEARCH\]\s*(.+?)(?=\n|$)'
        return re.findall(pattern, synthesized_profile)

    def calculate_improvement_delta(self, original_profile: str,
                                     updated_profile: str) -> dict:
        """Calculate how much the profile changed between rounds."""
        import re
        original_len = len(original_profile)
        updated_len = len(updated_profile)

        original_high = len(re.findall(r'high confidence', original_profile.lower()))
        updated_high = len(re.findall(r'high confidence', updated_profile.lower()))

        original_low = len(re.findall(r'low confidence', original_profile.lower()))
        updated_low = len(re.findall(r'low confidence', updated_profile.lower()))

        return {
            "size_change": updated_len - original_len,
            "size_pct": round((updated_len - original_len) / max(original_len, 1) * 100, 1),
            "high_confidence_change": updated_high - original_high,
            "low_confidence_change": original_low - updated_low,
        }
