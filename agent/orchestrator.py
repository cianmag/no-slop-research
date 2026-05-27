"""
No-Slop Research — Main Pipeline Orchestrator

Coordinates the 5-phase adversarial research pipeline:
  Phase 1: Deep Research (web search + LLM synthesis)
  Phase 2: Profile Building (LLM synthesis)
  Phase 3: Adversarial Interrogation (Team A validates, Team B challenges)
  Phase 4: Synthesis & Re-test (loop until bulletproof)
  Phase 5: Final Report (clean output with confidence ratings)

UPDATED: Now uses direct LLM API calls, cost tracking, noise filtering,
and evidence-based confidence scoring. No external agent framework required.
"""

import json
import time
import uuid
import os
import sqlite3
from datetime import datetime, timezone
from typing import Optional

from .research_phase import ResearchPhase
from .profile_builder import ProfileBuilder
from .validator_team import ValidatorTeam
from .challenger_team import ChallengerTeam
from .synthesizer import Synthesizer
from .report_generator import ReportGenerator
from .llm_client import LLMClient, create_client_from_config, create_client_from_env
from .confidence_scorer import score_confidence

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dashboard", "research.db")


def get_db():
    """Get SQLite connection, creating tables if needed."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS research_runs (
            id TEXT PRIMARY KEY,
            topic TEXT NOT NULL,
            status TEXT DEFAULT 'pending',
            current_phase TEXT DEFAULT 'pending',
            current_round INTEGER DEFAULT 0,
            max_rounds INTEGER DEFAULT 3,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            final_report TEXT,
            research_profile TEXT,
            config TEXT DEFAULT '{}'
        );
        CREATE TABLE IF NOT EXISTS phase_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            phase TEXT NOT NULL,
            round_num INTEGER NOT NULL,
            team TEXT,
            result TEXT NOT NULL,
            improvement_points TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (run_id) REFERENCES research_runs(id)
        );
        CREATE TABLE IF NOT EXISTS subagent_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            phase TEXT NOT NULL,
            round_num INTEGER NOT NULL,
            agent_role TEXT NOT NULL,
            agent_index INTEGER DEFAULT 0,
            status TEXT DEFAULT 'running',
            output TEXT,
            tool_calls INTEGER DEFAULT 0,
            duration_ms INTEGER DEFAULT 0,
            started_at TEXT NOT NULL,
            completed_at TEXT,
            FOREIGN KEY (run_id) REFERENCES research_runs(id)
        );
        CREATE TABLE IF NOT EXISTS cost_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            phase TEXT NOT NULL,
            model TEXT,
            input_tokens INTEGER DEFAULT 0,
            output_tokens INTEGER DEFAULT 0,
            cost_usd REAL DEFAULT 0,
            duration_ms INTEGER DEFAULT 0,
            created_at TEXT NOT NULL,
            FOREIGN KEY (run_id) REFERENCES research_runs(id)
        );
    """)
    conn.commit()
    return conn


class ResearchPipeline:
    """
    Main orchestrator for the adversarial research pipeline.
    Coordinates all phases and manages the adversarial loop.

    Now makes direct LLM API calls — no external agent framework required.
    """

    def __init__(self, topic: str, config: dict = None, run_id: str = None):
        self.topic = topic
        self.run_id = run_id or str(uuid.uuid4())[:8]
        self.config = config or {}
        self.max_rounds = self.config.get("max_rounds", int(os.environ.get("MAX_ADVERSARIAL_ROUNDS", "3")))
        self.max_research_agents = self.config.get("max_research_agents", int(os.environ.get("MAX_RESEARCH_AGENTS", "4")))

        # Initialize LLM client
        self.llm_client = self._create_llm_client()

        # Initialize phases
        self.research = ResearchPhase(self.config)
        self.profile_builder = ProfileBuilder()
        self.validator = ValidatorTeam(self.config)
        self.challenger = ChallengerTeam(self.config)
        self.synthesizer = Synthesizer()
        self.reporter = ReportGenerator()

        # State
        self.research_data = ""
        self.profile = ""
        self.validation_result = ""
        self.challenge_result = ""
        self.improvement_points = []
        self.final_report = ""
        self.history = []

        # Cost tracking
        self.cost_log = []

    def _create_llm_client(self) -> Optional[LLMClient]:
        """Create LLM client from config or environment."""
        # Try config first (from dashboard API keys)
        client = create_client_from_config(self.config)
        if client:
            return client
        # Fall back to environment variables
        return create_client_from_env()

    def _log(self, msg: str):
        ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
        print(f"[{ts}] [{self.run_id}] {msg}")

    def _update_db(self, **kwargs):
        kwargs["updated_at"] = datetime.now(timezone.utc).isoformat()
        sets = ", ".join(f"{k} = ?" for k in kwargs)
        vals = list(kwargs.values()) + [self.run_id]
        conn = get_db()
        conn.execute(f"UPDATE research_runs SET {sets} WHERE id = ?", vals)
        conn.commit()
        conn.close()

    def _log_phase(self, phase: str, round_num: int, team: str, result: str,
                    improvement_points: list = None):
        conn = get_db()
        conn.execute(
            "INSERT INTO phase_results (run_id, phase, round_num, team, result, improvement_points, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (self.run_id, phase, round_num, team, result,
             json.dumps(improvement_points) if improvement_points else None,
             datetime.now(timezone.utc).isoformat())
        )
        conn.commit()
        conn.close()

    def _log_subagent(self, phase: str, round_num: int, agent_role: str,
                       agent_index: int, status: str, output: str = None,
                       tool_calls: int = 0, duration_ms: int = 0):
        conn = get_db()
        now = datetime.now(timezone.utc).isoformat()
        if status == "running":
            conn.execute(
                "INSERT INTO subagent_logs (run_id, phase, round_num, agent_role, agent_index, status, started_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (self.run_id, phase, round_num, agent_role, agent_index, status, now)
            )
        else:
            conn.execute(
                "UPDATE subagent_logs SET status = ?, output = ?, tool_calls = ?, duration_ms = ?, completed_at = ? "
                "WHERE run_id = ? AND phase = ? AND round_num = ? AND agent_role = ? AND agent_index = ? "
                "AND completed_at IS NULL",
                (status, output, tool_calls, duration_ms, now,
                 self.run_id, phase, round_num, agent_role, agent_index)
            )
        conn.commit()
        conn.close()

    def _log_cost(self, phase: str):
        """Log current LLM client costs to DB."""
        if not self.llm_client:
            return
        summary = self.llm_client.get_cost_summary()
        conn = get_db()
        conn.execute(
            "INSERT INTO cost_logs (run_id, phase, model, input_tokens, output_tokens, cost_usd, duration_ms, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (self.run_id, phase, summary["model"],
             summary["total_input_tokens"], summary["total_output_tokens"],
             summary["total_cost_usd"], 0,
             datetime.now(timezone.utc).isoformat())
        )
        conn.commit()
        conn.close()

    def run(self) -> dict:
        """Execute the full adversarial research pipeline."""
        # Create DB record
        conn = get_db()
        conn.execute(
            "INSERT OR REPLACE INTO research_runs (id, topic, status, current_phase, max_rounds, created_at, updated_at, config) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (self.run_id, self.topic, "running", "research", self.max_rounds,
             datetime.now(timezone.utc).isoformat(), datetime.now(timezone.utc).isoformat(),
             json.dumps(self.config))
        )
        conn.commit()
        conn.close()

        # Check LLM client
        if not self.llm_client:
            self._log("WARNING: No LLM client configured. Running in degraded mode (no API calls).")
            self._log("Configure an API key in the dashboard or set LLM_API_KEY in .env")

        try:
            # ===== PHASE 1: DEEP RESEARCH =====
            self._log("Phase 1: Deep Research — searching web + LLM synthesis...")
            self._update_db(current_phase="research")
            self.research_data = self.research.execute(
                topic=self.topic,
                num_agents=self.max_research_agents,
                run_id=self.run_id,
                log_fn=self._log_subagent,
                llm_client=self.llm_client
            )
            self._log_phase("research", 0, "research", self.research_data[:2000])
            self._log_cost("research")
            self._log(f"Research complete. Gathered {len(self.research_data)} chars of data.")

            # ===== PHASE 2: PROFILE BUILDING =====
            self._log("Phase 2: Building Research Profile...")
            self._update_db(current_phase="profile")
            self.profile = self.profile_builder.build(
                topic=self.topic,
                research_data=self.research_data,
                llm_client=self.llm_client
            )
            self._log_phase("profile", 0, "synthesis", self.profile[:2000])
            self._update_db(research_profile=self.profile)
            self._log_cost("profile")
            self._log(f"Profile built: {len(self.profile)} chars.")

            # ===== PHASE 3 & 4: ADVERSARIAL LOOP =====
            for round_num in range(1, self.max_rounds + 1):
                self._log(f"=== ADVERSARIAL ROUND {round_num}/{self.max_rounds} ===")
                self._update_db(current_phase="adversarial", current_round=round_num)

                # Team A: Validator
                self._log(f"Round {round_num}: Team A (Validators) interrogating...")
                self._log_subagent("adversarial", round_num, "validator", 0, "running")
                start = time.time()
                self.validation_result = self.validator.validate(
                    topic=self.topic,
                    profile=self.profile,
                    round_num=round_num,
                    llm_client=self.llm_client
                )
                duration = int((time.time() - start) * 1000)
                self._log_subagent("adversarial", round_num, "validator", 0, "completed",
                                   self.validation_result[:1000], duration_ms=duration)
                self._log_phase("adversarial", round_num, "validator", self.validation_result)
                self._log_cost(f"adversarial_r{round_num}_validator")
                self._log(f"Team A validation complete ({len(self.validation_result)} chars).")

                # Team B: Challenger
                self._log(f"Round {round_num}: Team B (Challengers) attacking...")
                self._log_subagent("adversarial", round_num, "challenger", 0, "running")
                start = time.time()
                self.challenge_result = self.challenger.challenge(
                    topic=self.topic,
                    profile=self.profile,
                    round_num=round_num,
                    llm_client=self.llm_client
                )
                duration = int((time.time() - start) * 1000)
                self._log_subagent("adversarial", round_num, "challenger", 0, "completed",
                                   self.challenge_result[:1000], duration_ms=duration)
                self._log_phase("adversarial", round_num, "challenger", self.challenge_result)
                self._log_cost(f"adversarial_r{round_num}_challenger")

                # Extract and FILTER improvement points (noise filtering!)
                raw_points = self.challenger._extract_raw_points(self.challenge_result)
                self.improvement_points = self.challenger.extract_improvement_points(
                    self.challenge_result, filter_noise=True
                )
                filtered_count = len(raw_points) - len(self.improvement_points)
                self._log(f"Team B found {len(raw_points)} raw points, "
                          f"{filtered_count} filtered as noise, "
                          f"{len(self.improvement_points)} actionable improvements.")

                # Store round history
                self.history.append({
                    "round": round_num,
                    "validation_summary": self.validation_result[:500],
                    "challenge_summary": self.challenge_result[:500],
                    "improvement_points": self.improvement_points,
                    "improvement_count": len(self.improvement_points),
                    "noise_filtered": filtered_count
                })

                # Check if challengers found no significant flaws
                if len(self.improvement_points) == 0:
                    self._log(f"★ Team B found no significant flaws in round {round_num}. Research validated!")
                    break

                # ===== PHASE 4: SYNTHESIS & RE-RESEARCH =====
                self._log(f"Phase 4: Synthesizing {len(self.improvement_points)} improvements...")
                self._update_db(current_phase="synthesis")
                self.profile = self.synthesizer.merge(
                    topic=self.topic,
                    original_profile=self.profile,
                    improvement_points=self.improvement_points,
                    validation_result=self.validation_result,
                    challenge_result=self.challenge_result,
                    round_num=round_num,
                    llm_client=self.llm_client
                )
                self._log_phase("synthesis", round_num, "synthesis", self.profile[:2000],
                                self.improvement_points)
                self._log_cost(f"synthesis_r{round_num}")
                self._log(f"Synthesis complete. Profile updated ({len(self.profile)} chars).")

            # ===== PHASE 5: FINAL REPORT =====
            self._log("Phase 5: Generating Final Report...")
            self._update_db(current_phase="report")
            self.final_report = self.reporter.generate(
                topic=self.topic,
                profile=self.profile,
                history=self.history,
                validation_result=self.validation_result,
                challenge_result=self.challenge_result,
                total_rounds=len(self.history),
                llm_client=self.llm_client
            )
            self._log_cost("report")

            # ===== CONFIDENCE SCORING =====
            confidence = score_confidence(
                validation_result=self.validation_result,
                research_profile=self.profile,
                challenge_result=self.challenge_result
            )

            self._update_db(status="completed", current_phase="completed",
                            final_report=self.final_report)

            # Final cost summary
            cost_summary = {}
            if self.llm_client:
                cost_summary = self.llm_client.get_cost_summary()
                self._log(f"★ Total cost: ${cost_summary['total_cost_usd']:.4f} "
                          f"({cost_summary['total_tokens']} tokens, "
                          f"{cost_summary['total_calls']} API calls)")

            self._log(f"★ Pipeline complete! Final report: {len(self.final_report)} chars.")
            self._log(f"★ Confidence: {confidence['confidence_label']} "
                      f"(score: {confidence['overall_score']:.2f})")

            return {
                "success": True,
                "run_id": self.run_id,
                "topic": self.topic,
                "rounds": len(self.history),
                "final_report": self.final_report,
                "history": self.history,
                "profile": self.profile,
                "confidence": confidence,
                "cost": cost_summary
            }

        except Exception as e:
            self._log(f"ERROR: {str(e)}")
            self._update_db(status="error", current_phase=f"error: {str(e)[:200]}")
            return {
                "success": False,
                "run_id": self.run_id,
                "error": str(e),
                "cost": self.llm_client.get_cost_summary() if self.llm_client else {}
            }

    def get_status(self) -> dict:
        """Get current pipeline status from DB."""
        conn = get_db()
        row = conn.execute("SELECT * FROM research_runs WHERE id = ?", (self.run_id,)).fetchone()
        conn.close()
        if row:
            return dict(row)
        return {"id": self.run_id, "status": "not_found"}
