**bash** — arbitrary shell. Cost: ~instant. Example:
`<action name="bash">ls {workdir} && grep -rn "func Foo" {workdir}</action>`

**gotest** — run `go test`. Empty payload reruns every package; a payload is used as the `-run <pattern>` test selector. Cost: ~10s+ (compiles first). Example:
`<action name="gotest">TestEvaluate_FlagDisabled</action>`

The output includes `rc=<code>`, a summary of failed test names (`--- FAIL: ...`), and the go test output. Use this to SEE the failure before you start the debugger.

**probe** is NOT available for Go — use `read` + `gotest` to orient, then drive the debugger.

**read** — source inspection. Either range-read or grep. Cost: ~instant. Examples:
`<action name="read">{{"path":"{workdir}/server/evaluator.go","start":30,"n":40}}</action>`
`<action name="read">{{"grep":"func (s \\*Server) Evaluate","path":"{workdir}/server/evaluator.go"}}</action>`

Rule: prefer `read` + `gotest` before `dlv_start`. Use the debugger when you need to
step through control flow or inspect locals that static reading can't reveal. Budget
reads tightly — ~2-3 to orient, then open `dlv_start`; do not keep reading turn after turn.

**dlv_start** — start a long-lived **Delve (`dlv test`)** debug session. Cost: ~3s. The
`run_args` name a go test target and the harness builds
`dlv test <pkg> -- -test.run '^<TestName>$'` for you. Subsequent `dlv_cmd` turns share
this session. `run_args` shape: `["./<pkg>", "-run", "<TestName>"]` (package path +
test name). `dlv test` does NOT auto-stop at the failure — it halts at the REPL with
the program not yet running, so YOU set a breakpoint and `continue` to reach the bug.

`<action name="dlv_start">{{"run_args":["./server","-run","TestEvaluate_FlagDisabled"]}}</action>`
`<action name="dlv_start">{{"run_args":["./...","-run","TestRepro"]}}</action>`

**dlv_cmd** — single **dlv** command in the running session:
`<action name="dlv_cmd">{{"cmd":"break server.(*Server).Evaluate"}}</action>`  (break on a function — PREFERRED)
`<action name="dlv_cmd">{{"cmd":"break ./server/evaluator.go:20"}}</action>`  (break on file:line — use a repo-RELATIVE path)
(NEVER use an absolute path like `/app/server/evaluator.go:20` — dlv rejects it. Use the `<pkg>.<Func>` form or a `./`-relative path.)
`<action name="dlv_cmd">{{"cmd":"continue"}}</action>`  (run to the breakpoint)
`<action name="dlv_cmd">{{"cmd":"next"}}</action>`  (step over)
`<action name="dlv_cmd">{{"cmd":"step"}}</action>`  (step into)
`<action name="dlv_cmd">{{"cmd":"stepout"}}</action>`  (step out)
`<action name="dlv_cmd">{{"cmd":"print s.store"}}</action>`  (evaluate an expression)
`<action name="dlv_cmd">{{"cmd":"locals"}}</action>`  (dump all locals at the current frame)
`<action name="dlv_cmd">{{"cmd":"args"}}</action>`  (dump the current function's arguments)
`<action name="dlv_cmd">{{"cmd":"stack"}}</action>`  (print the call stack)

**dlv_script** — multiple dlv commands in one turn (each stepped individually). Example:
`<action name="dlv_script">{{"commands":["break server.(*Server).Evaluate","continue","args","locals","print s.store","next","print result"]}}</action>`

**dlv_stop** — close the session early. Usually unnecessary; the analyzer cleans up
automatically.
`<action name="dlv_stop">{{}}</action>`

Workflow (most reliable): `gotest` to see the failure → `dlv_start` with the failing
test → `break <pkg>.<Func>` (or `break <file>:<line>`) on an executable line on the
test's execution path → `continue` to land there → `args` / `locals` / `print <expr>`
to inspect, then `next` / `step` / `stepout` to follow execution. If `continue` runs to
completion WITHOUT stopping, your breakpoint line is not on the test's path — pick a
function the test actually calls.
