"""LLM-driven debug loop.

Each round: send transcript to the LLM, parse exactly one action or
conclusion, dispatch, append tool output, repeat. Bounded by max_rounds
and MAX_HISTORY_CHARS (B10)."""
from __future__ import annotations

import hashlib
import json
import re

import litellm
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .actions import parse_action
from .gating import MIN_PDB_CMDS, _is_real_frame, can_conclude


_IDENTICAL_REJECTION_ESCALATE_AT = 3  # 3rd identical rejection in a row


def _conclusion_fingerprint(text: str) -> str:
    """md5 of the <conclusion>...</conclusion> root_cause_files +
    root_cause_functions (path-normalized + sorted), used to detect the LLM
    re-emitting an essentially-identical rejected conclusion. Path-prefix
    variants like `lib/foo.py` vs `/app/lib/foo.py` collapse to the same
    fingerprint so the streak counter doesn't get fooled."""
    m = re.search(r"<conclusion>(.*?)</conclusion>", text or "", re.DOTALL)
    if not m:
        return ""
    body = m.group(1)

    def _norm_list(tag: str) -> list[str]:
        mt = re.search(rf"<{tag}>(.*?)</{tag}>", body, re.DOTALL)
        if not mt:
            return []
        items = [x.strip() for x in mt.group(1).replace("\n", ",").split(",")]
        out = []
        for it in items:
            if not it:
                continue
            it = it.lstrip("/")            # /app/lib/foo.py -> app/lib/foo.py
            if it.startswith("app/"):
                it = it[len("app/"):]      # app/lib/foo.py  -> lib/foo.py
            out.append(it)
        return sorted(set(out))

    sig = "F=" + ",".join(_norm_list("root_cause_files"))
    sig += "|N=" + ",".join(_norm_list("root_cause_functions"))
    return hashlib.md5(sig.encode()).hexdigest()


def _identical_rejection_streak(trajectory, current_resp: str) -> int:
    """Count of prior trajectory turns (most recent first, contiguous) that
    are kind=='invalid' AND share `current_resp`'s conclusion fingerprint.
    Returns 0 when the streak is broken or `current_resp` has no conclusion."""
    fp = _conclusion_fingerprint(current_resp)
    if not fp:
        return 0
    streak = 0
    for t in reversed(trajectory):
        if t.kind != "invalid":
            break
        if _conclusion_fingerprint(t.llm_response) != fp:
            break
        streak += 1
    return streak


_LOOP_ESCALATION = (
    "SYSTEM: STOP RE-EMITTING <conclusion>. Your last 3 conclusions were "
    "rejected by the gate for the same reason. The gate needs runtime PDB "
    "evidence — re-emitting will not change that.\n"
    "Issue this NEXT TURN exactly:\n"
    '  <reason>collect pdb evidence at the failing frame</reason>\n'
    '  <action name="pdb_start">{{"run_args": ["pytest","-x","{repro}"]}}</action>\n'
    f"Then in the FOLLOWING turns issue at least {MIN_PDB_CMDS} "
    '<action name="pdb_cmd"> calls (e.g. `where`, `l`, `p <expr>`) at the '
    "failing frame BEFORE any new <conclusion>."
)


_LOOP_ESCALATION_GO = (
    "SYSTEM: STOP RE-EMITTING <conclusion>. Your last 3 conclusions were "
    "rejected by the gate for the same reason. The gate needs runtime dlv "
    "evidence — re-emitting will not change that.\n"
    "Issue this NEXT TURN exactly:\n"
    '  <reason>collect dlv evidence at the failing frame</reason>\n'
    '  <action name="dlv_start">{"run_args": ["./...", "-run", "<TestName>"]}</action>\n'
    f"Then in the FOLLOWING turns issue at least {MIN_PDB_CMDS} "
    '<action name="dlv_cmd"> calls (e.g. `break <pkg>.<Func>`, `continue`, '
    "`args`, `print <expr>`) at the failing frame BEFORE any new <conclusion>."
)


def _diagnose_gate(history_view: list[dict], pdb_log: list[dict], language: str = "python") -> str:
    from .gating import _had_pytest_evidence
    pytest_ok = _had_pytest_evidence(history_view, pdb_log)
    by_sid: dict[int, list[dict]] = {}
    for t in pdb_log:
        sid = t.get("session_id")
        if sid is not None:
            by_sid.setdefault(sid, []).append(t)
    best_cmds = 0
    any_real_stop = False
    for entries in by_sid.values():
        first = None
        for i, t in enumerate(entries):
            f = t.get("initial_frame") if t.get("kind") == "start" else t.get("current_frame")
            if _is_real_frame(f):
                first = i
                any_real_stop = True
                break
        if first is None:
            continue
        cmds = sum(1 for t in entries[first + 1:] if t.get("kind") == "cmd" and not t.get("ended"))
        best_cmds = max(best_cmds, cmds)
    parts: list[str] = ["SYSTEM: conclusion rejected by gate."]
    if language == "go":
        if not pytest_ok:
            parts.append(
                "- No successful gotest run yet. Emit "
                '<action name="gotest">…</action> and confirm it produced a runtime '
                "failure (build errors do NOT count)."
            )
        if not any_real_stop:
            parts.append(
                "- dlv never stopped at a real (non-stdlib, non-vendored) frame. "
                "Common cause: the breakpoint was off the test's execution path so "
                "`continue` ran to completion. Retry with "
                '`{"run_args":["./...","-run","<TestName>"]}`, set '
                "`break <pkg>.<Func>` on a function the test calls, then `continue`."
            )
        elif best_cmds < MIN_PDB_CMDS:
            parts.append(
                f"- You only issued {best_cmds} dlv_cmd turn(s) at a real frame; the gate "
                f"requires at least {MIN_PDB_CMDS}. Continue driving the session with "
                "dlv_cmd / dlv_script (e.g., `args`, `locals`, `print <expr>`, `next`, `step`)."
            )
    else:
        if not pytest_ok:
            parts.append(
                "- No successful pytest run yet. Emit "
                '<action name="pytest">…</action> and confirm it produced a runtime '
                "failure (collection errors do NOT count)."
            )
        if not any_real_stop:
            parts.append(
                "- PDB never stopped at a real (non-pdb-internal) frame. Common cause: "
                "pdb_start landed in post-mortem at <string>:1 because run_args was wrong. "
                'Retry with `{"run_args":["pytest","-x","<repro_path>"]}` (the harness '
                "prefixes `python -m pdb` for you — do NOT include `python` or `python -m`)."
            )
        elif best_cmds < MIN_PDB_CMDS:
            parts.append(
                f"- You only issued {best_cmds} pdb_cmd turn(s) at a real frame; the gate "
                f"requires at least {MIN_PDB_CMDS}. Continue driving the session with "
                "pdb_cmd / pdb_script (e.g., `where`, `l`, `p <expr>`, `n`, `s`)."
            )
    parts.append("Do NOT re-emit <conclusion> until the above is fixed.")
    return "\n".join(parts)

_PROMPTS = Path(__file__).parent / "prompts"
_MAX_HISTORY_CHARS = 160_000  # raised for stateful pdb (B10 successor)

_CONCL = re.compile(r"<conclusion>(.*?)</conclusion>", re.DOTALL)
_ACTION_OPEN = re.compile(
    r'<action\s+name="(?:bash|pytest|gotest|pdb|pdb_start|pdb_cmd|pdb_script|pdb_stop'
    r'|dlv_start|dlv_cmd|dlv_script|dlv_stop|probe|read)"\s*>'
)
_ALL_ACTIONS = re.compile(
    r'<action\s+name="(?:bash|pytest|gotest|pdb|pdb_start|pdb_cmd|pdb_script|pdb_stop'
    r'|dlv_start|dlv_cmd|dlv_script|dlv_stop|probe|read)"\s*>',
    re.DOTALL,
)
_REASON_RE = re.compile(r"<reason>(.*?)</reason>", re.DOTALL)

# Opening user turn for the role-separated chat loop. Reinforces the
# one-action-then-stop contract and forbids the model from writing the tool's
# turn itself — the failure mode that produced runaway responses.
_KICKOFF = (
    "Begin your root-cause investigation. Emit EXACTLY ONE "
    '<reason>…</reason><action name="…">…</action> and then STOP. Wait for the '
    "TOOL output to be returned to you before issuing the next action. Never "
    "write a `TOOL (...)` line or any tool output yourself."
)

# Per-turn history is truncated to this many chars when fed back as context
# (full responses/outputs are still recorded in the trajectory for the report).
_TURN_CONTEXT_CHARS = 4000


def _messages_chars(messages: list[dict]) -> int:
    return sum(len(m.get("content", "")) for m in messages)


def _action_start(resp: str) -> int:
    m = _ACTION_OPEN.search(resp)
    return m.start() if m else 10 ** 9  # +inf sentinel


def _tag(tag: str):
    return re.compile(rf"<{tag}>(.*?)</{tag}>", re.DOTALL)


def _pick(tag: str, text: str) -> str:
    m = _tag(tag).search(text)
    return m.group(1).strip() if m else ""


def _pick_list(tag: str, text: str) -> list[str]:
    raw = _pick(tag, text)
    return [x.strip() for x in raw.replace("\n", ",").split(",") if x.strip()]


@dataclass
class TrajectoryTurn:
    """One full round of the debug loop.

    kind:
      - "action"     — LLM emitted <action>, tool ran, `tool_output` populated.
      - "conclusion" — LLM emitted <conclusion>; loop exits after this turn.
      - "invalid"    — LLM emitted neither; we inject a SYSTEM reminder and retry.
    """
    kind: str
    llm_response: str              # raw, unredacted LLM reply (including any prose)
    action_name: str = ""
    action_payload: str = ""       # for structured actions, JSON of kwargs
    tool_output: str = ""          # present when kind == "action"

    def to_json(self) -> dict:
        return {
            "kind": self.kind,
            "llm_response": self.llm_response,
            "action_name": self.action_name,
            "action_payload": self.action_payload,
            "tool_output": self.tool_output,
        }


@dataclass
class DebugReport:
    root_cause_files: list[str] = field(default_factory=list)
    root_cause_functions: list[str] = field(default_factory=list)
    reasoning: str = ""
    suggested_fix: str = ""
    # transcript is the legacy (action, payload, tool_output) view kept for
    # Phase C templating; trajectory is the full round-by-round record.
    transcript: list[tuple[str, str, str]] = field(default_factory=list)
    trajectory: list[TrajectoryTurn] = field(default_factory=list)
    timed_out: bool = False

    def to_json(self) -> dict:
        return {
            "root_cause_files": self.root_cause_files,
            "root_cause_functions": self.root_cause_functions,
            "reasoning": self.reasoning,
            "suggested_fix": self.suggested_fix,
            "transcript": [list(t) for t in self.transcript],
            "trajectory": [t.to_json() for t in self.trajectory],
            "timed_out": self.timed_out,
        }


class DebugAnalyzer:
    def __init__(
        self,
        llm: Callable[[list], str],
        dispatch: Callable[[Any, Any, dict], str],
        container: Any,
        ctx: dict,
        max_rounds: int = 40,
        max_history_chars: int = _MAX_HISTORY_CHARS,
    ):
        self.llm = llm
        self.dispatch = dispatch
        self.container = container
        self.ctx = ctx
        self.max_rounds = max_rounds
        self.max_history_chars = max_history_chars

    def _system_prompt(self) -> str:
        workdir = self.ctx.get("workdir", "/app")
        repro_path = self.ctx.get("repro_path", "")
        if self.ctx.get("language") == "go":
            sys_txt = (_PROMPTS / "analyzer_system_go.j2").read_text()
            tut = (_PROMPTS / "action_tutorial_go.md").read_text().format(
                workdir=workdir,
                repro_path=repro_path,
            )
            checklist = (_PROMPTS / "dlv_checklist.md").read_text()
            return sys_txt.format(
                issue=self.ctx.get("issue", ""),
                workdir=workdir,
                repro_path=repro_path,
                repro_nodeid=self.ctx.get("repro_nodeid", ""),
                preflight_seed=self.ctx.get("preflight_seed", "(none)"),
                tutorial=tut,
                dlv_checklist=checklist,
            )
        sys_txt = (_PROMPTS / "analyzer_system.j2").read_text()
        tut = (_PROMPTS / "action_tutorial.md").read_text().format(
            workdir=workdir,
            repro_path=repro_path,
        )
        checklist = (_PROMPTS / "pdb_checklist.md").read_text()
        return sys_txt.format(
            issue=self.ctx.get("issue", ""),
            workdir=workdir,
            repro_path=repro_path,
            repro_nodeid=self.ctx.get("repro_nodeid", ""),
            preflight_seed=self.ctx.get("preflight_seed", "(none)"),
            tutorial=tut,
            pdb_checklist=checklist,
        )

    def run(self) -> DebugReport:
        report = DebugReport()
        # Role-separated chat history. The model's turn ends at the assistant
        # message boundary, so it cannot continue into (and fabricate) the
        # tool's turn — the flat-transcript form that caused runaway responses.
        messages: list[dict] = [
            {"role": "system", "content": self._system_prompt()},
            {"role": "user", "content": _KICKOFF},
        ]

        def push(resp_text: str, user_text: str) -> None:
            """Append the model's turn (truncated for context) + the next user
            turn. Full resp is recorded in the trajectory separately."""
            messages.append({"role": "assistant", "content": resp_text[:_TURN_CONTEXT_CHARS]})
            messages.append({"role": "user", "content": user_text})

        for _ in range(self.max_rounds):
            try:
                resp = self.llm(messages)
            except litellm.exceptions.ContextWindowExceededError:
                # Over the model context window — tu-zi only signals this by
                # hanging until timeout, so re-prompting would spin every round.
                # Stop the debug loop and surface the partial report.
                report.timed_out = True
                return report
            if not isinstance(resp, str):
                resp = ""

            concl = _CONCL.search(resp)
            action_matches = list(_ALL_ACTIONS.finditer(resp))
            reason_match = _REASON_RE.search(resp)

            # Path A: conclusion only (gated by pdb-driven evidence)
            if concl and not action_matches:
                history_view = [
                    {"action_name": t.action_name,
                     "tool_output_ok": bool(t.tool_output) and not t.tool_output.startswith("ERROR")}
                    for t in report.trajectory if t.kind == "action"
                ]
                pdb_log = self.ctx.get("_pdb_session_log", [])
                if not can_conclude(history_view, pdb_log):
                    streak = _identical_rejection_streak(report.trajectory, resp)
                    report.trajectory.append(TrajectoryTurn(kind="invalid", llm_response=resp))
                    _lang = self.ctx.get("language", "python")
                    if streak + 1 >= _IDENTICAL_REJECTION_ESCALATE_AT:
                        if _lang == "go":
                            correction = _LOOP_ESCALATION_GO
                        else:
                            correction = _LOOP_ESCALATION.format(
                                repro=self.ctx.get("repro_path", "<repro_path>"),
                            )
                    else:
                        correction = _diagnose_gate(history_view, pdb_log, language=_lang)
                    push(resp, correction)
                    continue
                body = concl.group(1)
                report.root_cause_files = _pick_list("root_cause_files", body)
                report.root_cause_functions = _pick_list("root_cause_functions", body)
                report.reasoning = _pick("reasoning", body)
                report.suggested_fix = _pick("suggested_fix", body)
                report.trajectory.append(TrajectoryTurn(kind="conclusion", llm_response=resp))
                return report

            # Path B: at least one action + reason, no conclusion.
            # If multiple <action> tags appear, execute only the FIRST and warn
            # — the model frequently batches read+read or read+probe and we
            # don't want to burn a turn on rejection.
            if action_matches and reason_match and not concl:
                multi = len(action_matches) > 1
                if multi:
                    first_end = action_matches[0].end()
                    close_idx = resp.find("</action>", first_end)
                    truncated = resp[: close_idx + len("</action>")] if close_idx != -1 else resp
                else:
                    truncated = resp
                action = parse_action(truncated)
                if action is None:
                    report.trajectory.append(TrajectoryTurn(kind="invalid", llm_response=resp))
                    push(resp, "SYSTEM: could not parse <action> body; check the tutorial.")
                    continue
                output = self.dispatch(action, self.container, self.ctx)
                if multi:
                    output = (
                        "[SYSTEM] You emitted multiple <action> tags; only the FIRST "
                        "was executed. Wait for tool output before issuing the next action.\n"
                        + output
                    )
                payload_repr = action.payload or json.dumps(action.kwargs, ensure_ascii=False)
                report.transcript.append((action.name, payload_repr, output))
                report.trajectory.append(TrajectoryTurn(
                    kind="action", llm_response=resp,
                    action_name=action.name, action_payload=payload_repr, tool_output=output,
                ))
                push(resp, f"TOOL ({action.name}):\n{output[:_TURN_CONTEXT_CHARS]}")
                if _messages_chars(messages) > self.max_history_chars * 0.7:
                    from .history_compaction import compact_pdb_turns
                    messages = compact_pdb_turns(messages, fold_first_n_pdb_turns=10)
                if _messages_chars(messages) > self.max_history_chars:
                    report.timed_out = True
                    return report
                continue

            # Path C: malformed — re-prompt, don't silently drop
            if action_matches and not reason_match:
                reason = "missing <reason>…</reason> before <action>"
            elif action_matches and concl:
                reason = "mixed <action> and <conclusion> in one turn"
            else:
                reason = "no <action> or <conclusion> tag"
            report.trajectory.append(TrajectoryTurn(kind="invalid", llm_response=resp))
            push(
                resp,
                f"SYSTEM: malformed turn ({reason}). Emit EXACTLY ONE of:\n"
                "  <reason>…</reason><action name=\"…\">…</action>\n"
                "OR (only after you have runtime evidence):\n"
                "  <conclusion>…</conclusion>"
            )
        report.timed_out = True
        return report
