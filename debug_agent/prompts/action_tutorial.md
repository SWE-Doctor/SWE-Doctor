**bash** — arbitrary shell. Cost: ~instant. Example:
`<action name="bash">ls {workdir} && grep -rn "def foo" {workdir}</action>`

**pytest** — run pytest. Empty payload reruns the repro. Cost: ~10s. Example:
`<action name="pytest">-x -s -vv --tb=long {repro_path}</action>`

**pdb** — one-shot scripted pdb. Payload is newline-separated pdb commands ending in `q`. Cost: ~15s. Example:
`<action name="pdb">b {workdir}/pkg/mod.py:42
c
p self.state
q</action>`

**probe** — inject `print("PROBE", repr(<expr>))` before a line, run a command, auto-revert. Cost: ~5s. Structured payload:
`<action name="probe">{{"file":"{workdir}/pkg/mod.py","before_line":42,"expr":"self.state","run_cmd":"pytest -x {repro_path}"}}</action>`

**read** — source inspection. Either range-read or grep. Cost: ~instant. Examples:
`<action name="read">{{"path":"{workdir}/pkg/mod.py","start":30,"n":40}}</action>`
`<action name="read">{{"grep":"def handle_event","path":"{workdir}/pkg/mod.py"}}</action>`

Rule: prefer `read` + `probe` before `pdb`. Use `pdb` only when you need to
step through control flow or inspect locals that probes can't reach.

**pdb_start** — start a long-lived pdb subprocess. Cost: ~3s. Subsequent
pdb_cmd turns share this session.

When `run_args` invokes pytest (the common case), the harness uses
`python -m pytest --pdb` and pytest **auto-drops into pdb at the failing
assertion's real frame**. You will land directly on the failing
assertion — do NOT manually `b <file>:<line>` then `c`; that just runs
pytest to completion. Use `where` / `pp <var>` / `l` from the first
pdb_cmd onward to inspect the failure.

For non-pytest scripts/modules the harness uses `python -m pdb` and
you will land at the script's first executable line; set a breakpoint
and `c` as usual.

`<action name="pdb_start">{{"run_args":["pytest","-x","{repro_path}"]}}</action>`
`<action name="pdb_start">{{"run_args":["unittest","discover","-s","{workdir}"]}}</action>`
`<action name="pdb_start">{{"run_args":["{workdir}/scripts/repro.py"]}}</action>`

**pdb_cmd** — single pdb command in the running session. Example:
`<action name="pdb_cmd">{{"cmd":"b {workdir}/pkg/mod.py:42"}}</action>`
`<action name="pdb_cmd">{{"cmd":"c"}}</action>`
`<action name="pdb_cmd">{{"cmd":"p self.state"}}</action>`

**pdb_script** — multiple pdb commands in one turn (each line individually
stepped). Example:
`<action name="pdb_script">{{"commands":["b mod.py:42","c","p x","pp self.__dict__","n","p x"]}}</action>`

**pdb_stop** — close the session early. Usually unnecessary; the analyzer
cleans up automatically.
`<action name="pdb_stop">{{}}</action>`

The old one-shot `pdb` action is **deprecated** — prefer pdb_start + pdb_cmd
because the new gating predicate explicitly counts commands inside a stateful
session.
