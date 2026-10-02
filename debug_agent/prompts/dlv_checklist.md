When you stop in dlv, before stepping further, ACTIVELY check the state against
this checklist. The bug almost always sits in one of these. You don't need to
report all of them — just look (`locals`, `args`, `print <expr>`, `stack`), and let
the anomalies guide your next step.

## dlv command reference
- `break <pkg>.<Func>` — break on a function, e.g. `break server.(*Server).Evaluate`
- `break <file>:<line>` — break on an **executable** source line, e.g. `break ./server/evaluator.go:20`
- `continue` — run to the next breakpoint or program end
- `next` — step to the next line (step over)
- `step` — step into the next function call
- `stepout` — step out of the current function
- `print <expr>` — evaluate an expression in the current frame (e.g. `print s.store`, `print len(items)`)
- `locals` — dump all local variables at the current frame (fastest way to survey state)
- `args` — dump the current function's arguments
- `stack` — print the call stack
- `quit` — end the session

**Important:** set breakpoints on executable source lines, then `continue` to reach
them. Verify you are in the right frame with `stack`, then survey state with `locals` /
`args` before stepping. Prefer `locals` over many individual `print` calls.

## What to look for once stopped

1. Nil pointer dereference (`panic: runtime error: invalid memory address or nil pointer dereference`)
   - A pointer/interface field is `nil` where an object is expected — `print x`, `print x == nil`
   - A method called on a nil receiver; a map/slice field never initialized
   - dlv: `args` to see the nil receiver, `print <field>` to find which is nil

2. Interface type assertion failure (`panic: interface conversion: ... is not ...`)
   - `v.(T)` where the dynamic type differs — `print v` shows the concrete type
   - Missing comma-ok form (`x, ok := v.(T)`); wrong concrete type wired in

3. Goroutine / channel deadlock (`fatal error: all goroutines are asleep - deadlock`)
   - Send on a channel with no receiver (or vice versa); unbuffered channel never drained
   - `WaitGroup` count mismatch; `<-ch` after the producer already returned
   - dlv: `goroutines` and `stack` to see who is blocked on what

4. Concurrent map access (`fatal error: concurrent map read and map write`)
   - A `map` written from one goroutine while read from another without a lock
   - Missing `sync.Mutex` / `sync.RWMutex` around shared state

5. Slice / array bounds (`panic: runtime error: index out of range [i] with length n`)
   - `s[i]` with `i >= len(s)`; `s[a:b]` with `b > cap(s)`; empty slice indexed at 0
   - dlv: `print len(s)`, `print i` right before the access

6. Branch / loop took the wrong arm
   - The `if` arm you expected was never entered; `else` was the default
   - `range` over an empty / nil slice or map iterated 0 times
   - `&&` / `||` short-circuited; the right side never ran
   - An `err != nil` check swallowed or ignored the real error

7. Off-by-one / boundary
   - Loop bound `< len` vs `<= len`; inclusive vs exclusive slice bounds
   - Index 0 vs last element; empty string/slice treated as a valid value

8. defer / recover / error handling
   - A `defer` mutated a named return value unexpectedly
   - `recover()` swallowed a panic and returned a zero value
   - `err` shadowed by `:=` in an inner scope; the outer `err` stays nil
   - Wrapped error compared with `==` instead of `errors.Is` / `errors.As`

9. Contract drift across the call site
   - Return value / error class / signature changed; caller not updated
   - A struct field added/renamed; a zero value silently used

10. Environment dependence
   - cwd, env var, `GOFLAGS`, build tags, `GOOS`/`GOARCH`
   - Time/locale; map iteration order assumed stable (it is not)
