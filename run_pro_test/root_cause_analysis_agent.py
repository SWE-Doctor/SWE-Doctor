"""Phase 3: LLM Root Cause Analysis Agent for bug localization refinement.

Uses mini-swe-agent's DefaultAgent + LitellmModel + LocalEnvironment scaffold
to give an LLM bash access within the source_snapshot directory. The agent
actively explores import chains, call graphs, and function definitions to
identify buggy files that static heuristics missed.
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml

# Add mini-swe-agent src to path for imports
_MINI_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_MINI_SRC) not in sys.path:
    sys.path.insert(0, str(_MINI_SRC))

from minisweagent.agents.default import AgentConfig, DefaultAgent
from minisweagent.environments.local import LocalEnvironment
from minisweagent.models.litellm_model import LitellmModel

# Local imports (run_pro_test/)
sys.path.insert(0, str(Path(__file__).parent))
from root_cause_analyzer import RootCauseCandidate
from statement_tracer import StatementTrace
from context_extractor import FailureContext

logger = logging.getLogger("root_cause_analysis_agent")

CONFIG_PATH = Path(__file__).parent / "config" / "root_cause_analysis_agent.yaml"


@dataclass
class AgentCandidate:
    """A candidate file from the RCA agent's output."""

    file: str
    line: int = 0
    confidence: float = 0.5
    reasoning: str = ""


def _load_config() -> dict:
    """Load the root cause analysis agent YAML config."""
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _format_traceback(trace: StatementTrace) -> str:
    """Format traceback frames as text for the agent prompt."""
    if not trace.traceback_frames:
        return "(no traceback available)"
    lines = []
    for frame in trace.traceback_frames:
        lines.append(f'  File "{frame.file}", line {frame.lineno}, in {frame.func_name}')
        if frame.code_line:
            lines.append(f"    {frame.code_line}")
    return "\n".join(lines)


def _format_candidates(candidates: list[RootCauseCandidate], limit: int = 10) -> list[dict]:
    """Format Phase 2 candidates as dicts for Jinja2 template."""
    result = []
    for c in candidates[:limit]:
        result.append(
            {
                "file": c.file,
                "line": c.line,
                "score": c.score,
                "signals": c.signals,
                "code_snippet": c.code_snippet or "",
            }
        )
    return result


def _parse_submission(text: str) -> list[AgentCandidate]:
    """Parse the agent's JSON submission into AgentCandidate objects.

    Handles common LLM output issues: markdown fences, trailing commas, etc.
    """
    text = text.strip()
    if not text:
        return []

    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.splitlines()
        # Remove first line (```json or ```) and last line (```)
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        logger.warning("Failed to parse agent submission as JSON: %s", text[:200])
        return []

    if not isinstance(data, list):
        logger.warning("Agent submission is not a list: %s", type(data))
        return []

    candidates = []
    for item in data:
        if not isinstance(item, dict) or "file" not in item:
            continue
        candidates.append(
            AgentCandidate(
                file=item["file"],
                line=int(item.get("line", 0)),
                confidence=max(0.0, min(1.0, float(item.get("confidence", 0.5)))),
                reasoning=str(item.get("reasoning", "")),
            )
        )
    return candidates


def _validate_candidates(candidates: list[AgentCandidate], repo_path: Path) -> list[AgentCandidate]:
    """Filter out candidates whose files don't exist in the repo."""
    valid = []
    for c in candidates:
        # Try exact path and common prefixes
        paths_to_check = [
            repo_path / c.file,
        ]
        # Strip leading slashes or /app/ prefix
        clean = c.file.lstrip("/")
        if clean.startswith("app/"):
            clean = clean[4:]
        paths_to_check.append(repo_path / clean)

        for p in paths_to_check:
            if p.is_file():
                # Normalize to relative path from repo root
                try:
                    c.file = str(p.relative_to(repo_path))
                except ValueError:
                    c.file = clean
                valid.append(c)
                break
        else:
            logger.debug("Agent candidate file not found: %s", c.file)
    return valid


class RootCauseAnalysisAgentRunner:
    """Runs the LLM root cause analysis agent on a single failing test.

    Uses mini-swe-agent's DefaultAgent with bash tool access to
    the source_snapshot directory.
    """

    def __init__(
        self,
        repo_path: Path,
        model_name: str | None = None,
        step_limit: int | None = None,
        cost_limit: float | None = None,
        trajectory_dir: Path | None = None,
    ):
        self.repo_path = Path(repo_path)
        self.config = _load_config()
        # Allow overrides
        if model_name:
            self.config["model"]["model_name"] = model_name
        if step_limit is not None:
            self.config["agent"]["step_limit"] = step_limit
        if cost_limit is not None:
            self.config["agent"]["cost_limit"] = cost_limit
        self.trajectory_dir = trajectory_dir
        self.extra_template_vars: dict[str, str] = {}

    def run(
        self,
        trace: StatementTrace,
        context: FailureContext | None,
        existing_candidates: list[RootCauseCandidate],
    ) -> list[RootCauseCandidate]:
        """Run the RCA agent and return refined candidates.

        Returns empty list on failure (timeout, parse error, etc.)
        - caller should fall back to Phase 2 results.
        """
        # Build template variables for the instance prompt
        template_vars = {
            "test_nodeid": trace.test_nodeid,
            "error_type": trace.error_type or "Unknown",
            "error_message": trace.error_message or "(no message)",
            "traceback_text": _format_traceback(trace),
            "existing_candidates": _format_candidates(existing_candidates),
            "focused_files": trace.focused_files[:50],  # limit to avoid prompt bloat
            **self.extra_template_vars,
        }

        # Set up trajectory path
        output_path = None
        if self.trajectory_dir:
            safe_name = trace.test_nodeid.replace("/", "_").replace("::", "__")[:100]
            output_path = self.trajectory_dir / f"rca_agent_{safe_name}.json"

        # Construct mini-swe-agent components
        agent_cfg = self.config["agent"].copy()
        agent_cfg["output_path"] = str(output_path) if output_path else None

        env_cfg = self.config.get("environment", {})
        env_cfg["cwd"] = str(self.repo_path)

        model_cfg = self.config.get("model", {})

        model = LitellmModel(**model_cfg)
        env = LocalEnvironment(**env_cfg)

        agent = DefaultAgent(model=model, env=env, **agent_cfg)
        # Inject template vars so Jinja2 can render them
        agent.extra_template_vars = template_vars

        try:
            result = agent.run(task="")  # task is embedded in instance_template via template_vars
            submission = result.get("submission", "")
            exit_status = result.get("exit_status", "")
        except Exception as e:
            logger.warning("RCA agent failed for %s: %s", trace.test_nodeid, e)
            return []

        if exit_status not in ("Submitted",):
            logger.info("Agent exited with status=%s for %s", exit_status, trace.test_nodeid)
            # Still try to parse if there's any submission
            if not submission:
                return []

        # Parse and validate
        agent_candidates = _parse_submission(submission)
        agent_candidates = _validate_candidates(agent_candidates, self.repo_path)

        if not agent_candidates:
            logger.info("Agent returned no valid candidates for %s", trace.test_nodeid)
            return []

        # Convert to RootCauseCandidate
        rca_candidates = []
        for ac in agent_candidates:
            rca_candidates.append(
                RootCauseCandidate(
                    file=ac.file,
                    line=ac.line,
                    score=ac.confidence,  # will be re-scored during merge
                    signals=[f"rca_agent({ac.confidence:.2f})"],
                    code_snippet="",
                    explanation=ac.reasoning,
                    func_name="",
                )
            )

        logger.info(
            "Agent returned %d candidates for %s (top: %s %.2f)",
            len(rca_candidates),
            trace.test_nodeid,
            rca_candidates[0].file if rca_candidates else "?",
            rca_candidates[0].score if rca_candidates else 0,
        )
        return rca_candidates
