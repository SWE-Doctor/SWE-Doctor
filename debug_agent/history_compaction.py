"""String-only summarization of older pdb_cmd turns.

Called by the analyzer when total history grows large, to keep stateful pdb
sessions from blowing the context budget without losing trace evidence.
Zero LLM calls — pure regex over the formatted history strings."""
from __future__ import annotations

import re

_FRAME_RE = re.compile(r"current_frame:\s*([^\n]+)")
_CMD_RE = re.compile(r'<action name="pdb_cmd">\s*\{"cmd"\s*:\s*"([^"]*)"\s*\}\s*</action>')
_TRANSCRIPT_TAIL_RE = re.compile(r"--- TRANSCRIPT ---\n(.*)$", re.DOTALL)


def _is_pdb_cmd_turn(h: str) -> bool:
    return "TOOL (pdb_cmd):" in h or "TOOL (pdb_script):" in h


def _summarize_one(h: str) -> str | None:
    m_cmd = _CMD_RE.search(h)
    m_frame = _FRAME_RE.search(h)
    if not m_cmd and not m_frame:
        return None
    parts = []
    if m_cmd:
        parts.append(f"cmd={m_cmd.group(1)!r}")
    if m_frame:
        parts.append(f"@{m_frame.group(1).strip()}")
    m_tx = _TRANSCRIPT_TAIL_RE.search(h)
    if m_tx and m_cmd:
        cmd_text = m_cmd.group(1).strip()
        if cmd_text.startswith(("p ", "pp ")):
            tail = m_tx.group(1).strip()
            lines = [l for l in tail.splitlines() if l.strip() and l.strip() != "(Pdb)"]
            if lines:
                parts.append(f"{cmd_text} = {lines[0].strip()[:120]}")
    return "  - " + "; ".join(parts)


def _summary_msg(folded_lines: list[str]) -> dict:
    return {"role": "user",
            "content": "<pdb_session_summary>\n" + "\n".join(folded_lines)
                       + "\n</pdb_session_summary>"}


def compact_pdb_turns(messages: list[dict], fold_first_n_pdb_turns: int) -> list[dict]:
    """Fold the first N pdb_cmd / pdb_script turns of a role-separated chat
    `messages` list into a single <pdb_session_summary> user message. The
    leading system + kickoff messages and all non-pdb turns are preserved in
    place. A "turn" is an (assistant, user-tool-output) message pair; the two
    contents are joined so the existing string-based summarizers apply."""
    if len(messages) <= 2:
        return list(messages)
    head, rest = list(messages[:2]), messages[2:]
    out: list[dict] = []
    folded_lines: list[str] = []
    folded = 0
    i = 0
    while i < len(rest):
        m = rest[i]
        nxt = rest[i + 1] if i + 1 < len(rest) else None
        is_pair = m.get("role") == "assistant" and nxt is not None and nxt.get("role") == "user"
        if is_pair:
            combined = m["content"] + "\n" + nxt["content"]
            if _is_pdb_cmd_turn(combined) and folded < fold_first_n_pdb_turns:
                line = _summarize_one(combined)
                if line:
                    folded_lines.append(line)
                folded += 1
                i += 2
                continue
            if folded_lines:
                out.append(_summary_msg(folded_lines))
                folded_lines = []
            out.append(m)
            out.append(nxt)
            i += 2
            continue
        # Defensive: a standalone message (shouldn't normally occur).
        if folded_lines:
            out.append(_summary_msg(folded_lines))
            folded_lines = []
        out.append(m)
        i += 1
    if folded_lines:
        out.append(_summary_msg(folded_lines))
    return head + out
