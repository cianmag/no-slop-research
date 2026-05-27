"""
Phase 3a: Validator Team (Team A) — Takes the research profile and argues
WHY the findings are correct. Cross-references claims with sources,
identifies strong evidence chains, and rates confidence levels.

UPDATED: Now makes direct LLM API calls.
"""

from .llm_client import LLMClient


class ValidatorTeam:
    """Team A — validates research findings and rates confidence."""

    def __init__(self, config: dict = None):
        self.config = config or {}

    def validate(self, topic: str, profile: str, round_num: int,
                 llm_client: LLMClient = None) -> str:
        """
        Run validation on a research profile.
        Returns validation analysis text.
        """
        if llm_client:
            return self._validate_with_llm(topic, profile, round_num, llm_client)
        else:
            return self._validate_fallback(topic, profile, round_num)

    def _validate_with_llm(self, topic: str, profile: str, round_num: int,
                            llm_client: LLMClient) -> str:
        """Use LLM to validate the research profile."""
        system = (
            "You are a VALIDATOR — a rigorous fact-checker and evidence analyst. "
            "Your job is to assess how VALID and RELIABLE research findings are. "
            "Be RIGOROUS but FAIR. Do not rubber-stamp claims."
        )

        user_msg = f"""TOPIC: {topic}
VALIDATION ROUND: {round_num}

RESEARCH PROFILE:
{profile[:12000]}

YOUR MISSION:
Analyze every claim in this profile and assess its validity.

For each key finding, evaluate:
1. SOURCE QUALITY — Are the sources credible? Primary or secondary? Biased?
2. EVIDENCE CHAIN — Does the evidence logically support the claim?
3. CONSISTENCY — Do multiple sources agree? Are there contradictions?
4. COMPLETENESS — Is important context missing?
5. RECENCY — Is the information current or potentially outdated?

OUTPUT FORMAT:

## VALIDATION SUMMARY
Overall assessment: How reliable is this research profile?
Rate: HIGH CONFIDENCE / MODERATE CONFIDENCE / LOW CONFIDENCE

## CLAIM-BY-CLAIM ANALYSIS
For each key finding:
- **Claim:** [what the profile states]
- **Evidence Quality:** [Strong/Moderate/Weak]
- **Source Check:** [do sources actually support this?]
- **Confidence Rating:** [0-100]%
- **Notes:** [any caveats or concerns]

## STRONGEST FINDINGS
Which findings have the strongest evidence? Why?

## WEAKEST FINDINGS
Which findings are most questionable? Why?

## VALIDATION VERDICT
Is this research profile ready to be published as reliable?
If not, what specific improvements are needed?

Be honest. Your job is truth, not comfort."""

        result = llm_client.chat(
            messages=[{"role": "user", "content": user_msg}],
            system=system,
            temperature=0.2,
            max_tokens=4096
        )

        if result["success"]:
            return result["content"]
        else:
            return self._validate_fallback(topic, profile, round_num)

    def _validate_fallback(self, topic: str, profile: str,
                            round_num: int) -> str:
        """Fallback validation without LLM."""
        return f"""## VALIDATION SUMMARY
No LLM client configured — cannot perform automated validation.

## CLAIM-BY-CLAIM ANALYSIS
[Validation requires an LLM API key to cross-reference claims with sources]

## VALIDATION VERDICT
Configure an API key in the dashboard to enable automated validation.
Round: {round_num}
"""
