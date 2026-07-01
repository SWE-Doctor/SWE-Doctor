"""Split an issue description into independent behavioral aspects.

Each aspect becomes one reproduction test. Cap avoids fan-out runaway. Must
NOT read dataset golden patch (see tests/test_no_golden_leak.py)."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable


@dataclass
class IssueAspect:
    aspect_id: str
    description: str


_PROMPT = """\
Split the issue's EXPECTED BEHAVIOR into at most {max_aspects} independent clauses.
Each clause is a single assertion the fixed code must satisfy. Return a JSON
array of {{aspect_id, description}} objects. aspect_id is "a0","a1",....
Do not invent behaviors absent from the issue.

ISSUE:
---
{issue}
---
Return JSON only.
"""


def split_into_aspects(issue: str, max_aspects: int, llm: Callable[[str], str]) -> list[IssueAspect]:
    try:
        data = json.loads(llm(_PROMPT.format(issue=issue, max_aspects=max_aspects)))
    except json.JSONDecodeError:
        data = []
    out = [IssueAspect(d["aspect_id"], d["description"]) for d in data[:max_aspects]]
    if not out:
        out = [IssueAspect("a0", issue[:200])]
    return out
