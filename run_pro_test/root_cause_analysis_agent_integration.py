"""Integration layer for Phase 3 (LLM Root Cause Analysis Agent) into the RCA pipeline.

Always triggers the agent on every instance. The agent itself decides whether
to do a quick confirmation (1 step) or full investigation (up to step_limit).

Phase 2 confidence info is passed to the agent prompt so it can make an
informed decision. No ground-truth / patch data is used in the trigger.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from root_cause_analyzer import RCAResult, RootCauseCandidate, evaluate_result
from statement_tracer import StatementTrace
from context_extractor import FailureContext, SourceReader
from root_cause_analysis_agent import RootCauseAnalysisAgentRunner

logger = logging.getLogger("root_cause_analysis_agent_integration")

# -- Merge weights ------------------------------------------------------------

BOOST_EXISTING = 0.20
SCORE_NEW = 0.15


# -- Phase 2 confidence (passed to agent prompt, NOT used for trigger) --------

def _count_signal_types(candidate: RootCauseCandidate) -> int:
    types = set()
    for s in candidate.signals:
        paren = s.find("(")
        types.add(s[:paren] if paren > 0 else s)
    return len(types)


def phase2_confidence(
    trace: StatementTrace,
    candidates: list[RootCauseCandidate],
) -> tuple[float, str]:
    """Compute Phase 2 confidence score and a human-readable note.

    This is passed to the agent so it can decide whether to confirm or investigate.
    NOT used as a trigger gate.

    Returns (score 0.0-1.0, note string).
    """
    if not candidates:
        return 0.0, "No candidates from static analysis"

    conf = 0.0
    notes = []
    top1 = candidates[0]

    # Production traceback
    has_ptb = any(not f.is_test_code for f in trace.traceback_frames) if trace.traceback_frames else False
    if has_ptb:
        conf += 0.35
        notes.append("has production traceback")
    else:
        notes.append("no production traceback (test-only frames)")

    # Signal diversity
    n_sig = _count_signal_types(top1)
    if n_sig >= 3:
        conf += 0.25
        notes.append(f"top-1 has {n_sig} signal types (strong)")
    elif n_sig >= 2:
        conf += 0.10
        notes.append(f"top-1 has {n_sig} signal types")
    else:
        notes.append(f"top-1 has only {n_sig} signal type (weak)")

    # Score strength
    if top1.score >= 0.3:
        conf += 0.15
    elif top1.score >= 0.15:
        conf += 0.05

    # Score gap
    if len(candidates) >= 2:
        gap = top1.score - candidates[1].score
        if gap >= 0.15:
            conf += 0.15
            notes.append(f"clear score leader (gap={gap:.3f})")
        elif gap >= 0.05:
            conf += 0.05
        else:
            n_tied = sum(1 for c in candidates if abs(c.score - top1.score) < 0.01)
            if n_tied > 5:
                notes.append(f"{n_tied} candidates tied at score ~{top1.score:.3f}")

    return min(conf, 1.0), "; ".join(notes)


# -- Merge -------------------------------------------------------------------

def merge_rankings(
    phase2: list[RootCauseCandidate],
    phase3: list[RootCauseCandidate],
) -> list[RootCauseCandidate]:
    """Merge Phase 2 and Phase 3 candidate rankings.

    - Phase 3 candidates in Phase 2: boost score by BOOST_EXISTING * confidence
    - Phase 3 candidates NOT in Phase 2: add with score SCORE_NEW * confidence
    - Re-sort, deduplicate by (file, line)
    """
    if not phase3:
        return phase2

    p2_by_file: dict[str, list[RootCauseCandidate]] = {}
    for c in phase2:
        p2_by_file.setdefault(c.file, []).append(c)

    boosted: set[int] = set()
    new_candidates: list[RootCauseCandidate] = []

    for p3c in phase3:
        confidence = p3c.score
        matched = False

        for p2_file, p2_list in p2_by_file.items():
            if _files_match(p3c.file, p2_file):
                for p2c in p2_list:
                    if id(p2c) not in boosted:
                        p2c.score += BOOST_EXISTING * confidence
                        p2c.signals.append(f"rca_agent({confidence:.2f})")
                        if p3c.explanation and not p2c.explanation:
                            p2c.explanation = p3c.explanation
                        boosted.add(id(p2c))
                matched = True
                break

        if not matched:
            new_candidates.append(
                RootCauseCandidate(
                    file=p3c.file,
                    line=p3c.line,
                    score=SCORE_NEW * confidence,
                    signals=[f"rca_agent({confidence:.2f})"],
                    code_snippet="",
                    explanation=p3c.explanation,
                    func_name="",
                )
            )

    all_candidates = phase2 + new_candidates
    all_candidates.sort(key=lambda c: -c.score)

    seen: set[tuple[str, int]] = set()
    deduped: list[RootCauseCandidate] = []
    for c in all_candidates:
        key = (c.file, c.line)
        if key not in seen:
            seen.add(key)
            deduped.append(c)

    return deduped


def _files_match(a: str, b: str) -> bool:
    if a == b:
        return True
    a_clean = a.lstrip("/")
    b_clean = b.lstrip("/")
    if a_clean == b_clean:
        return True
    if a_clean.endswith("/" + b_clean) or b_clean.endswith("/" + a_clean):
        return True
    return False


def _find_snapshot_path(
    instance_dir: Path | None = None,
    workspace: Path | None = None,
) -> Path | None:
    for base in [workspace, instance_dir]:
        if base is None:
            continue
        snapshot = base / "source_snapshot"
        if snapshot.exists() and any(snapshot.rglob("*.py")):
            return snapshot
    return None


def run_phase3_refinement(
    results: list[RCAResult],
    traces: list[StatementTrace],
    contexts: list[FailureContext],
    snapshot_path: Path | None = None,
    instance_dir: Path | None = None,
    workspace: Path | None = None,
    model_name: str | None = None,
    trajectory_dir: Path | None = None,
    patch_files: list[str] | None = None,
    patch_text: str = "",
) -> list[RCAResult]:
    """Run Phase 3 RCA agent on all tests.

    Always triggers. The agent decides internally whether to confirm
    Phase 2 (fast-path, 1 step) or investigate further.

    patch_files/patch_text are only used AFTER merging for evaluation metrics.
    """
    repo_path = snapshot_path
    if repo_path is None:
        repo_path = _find_snapshot_path(instance_dir, workspace)
    if repo_path is None:
        logger.info("No source_snapshot found, skipping Phase 3")
        return results

    runner = RootCauseAnalysisAgentRunner(
        repo_path=repo_path,
        model_name=model_name,
        trajectory_dir=trajectory_dir,
    )

    # Run agent once per instance, reuse for all tests
    agent_cache: list[RootCauseCandidate] | None = None
    agent_ran = False

    refined: list[RCAResult] = []
    for rca, trace, context in zip(results, traces, contexts):
        # Compute confidence info for the agent prompt
        conf_score, conf_note = phase2_confidence(trace, rca.candidates)
        has_ptb = any(not f.is_test_code for f in trace.traceback_frames) if trace.traceback_frames else False
        top1_sigs = _count_signal_types(rca.candidates[0]) if rca.candidates else 0

        logger.info(
            "Phase 3 running for %s (confidence=%.2f: %s)",
            trace.test_nodeid, conf_score, conf_note,
        )

        if not agent_ran:
            # Inject confidence metadata into template vars
            runner.extra_template_vars = {
                "phase2_confidence_score": f"{conf_score:.2f}",
                "has_production_traceback": "Yes" if has_ptb else "No",
                "top1_signal_count": str(top1_sigs),
                "confidence_note": conf_note,
            }
            agent_cache = runner.run(trace, context, rca.candidates)
            agent_ran = True

        phase3_candidates = agent_cache or []

        if phase3_candidates:
            merged = merge_rankings(rca.candidates, phase3_candidates)

            if patch_files:
                from root_cause_analyzer import extract_patch_changed_lines

                patch_lines = extract_patch_changed_lines(patch_text) if patch_text else None
                new_rca = evaluate_result(merged, patch_files, patch_lines)
                new_rca.test_nodeid = rca.test_nodeid
                new_rca.error_type = rca.error_type
                new_rca.error_message = rca.error_message
                refined.append(new_rca)

                if new_rca.top5_hit and not rca.top5_hit:
                    logger.info(
                        "Phase 3 IMPROVED %s: top5_hit False->True (top1: %s)",
                        trace.test_nodeid,
                        merged[0].file if merged else "?",
                    )
                elif rca.top5_hit and not new_rca.top5_hit:
                    logger.warning(
                        "Phase 3 REGRESSED %s: top5_hit True->False",
                        trace.test_nodeid,
                    )
            else:
                rca.candidates = merged
                refined.append(rca)
        else:
            refined.append(rca)

    return refined
