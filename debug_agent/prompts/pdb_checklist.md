When you stop in PDB, before stepping further, ACTIVELY check the locals against
this checklist. The bug almost always sits in one of these. You don't need to
report all of them — just look, and let the anomalies guide your next step.

1. Abnormal values
   - None where an object is expected; empty container (len == 0); 0 / -1 / NaN / inf
   - Type mismatch: str vs bytes, int vs float, Path vs str
   - Mutable default arg already polluted by previous call

2. Branch / loop took the wrong arm
   - The `if` arm you expected was never entered; `else` was the default
   - Loop iterated 0 times (empty iterable / off-by-one range)
   - `and` / `or` short-circuited and the right side never ran
   - `except` swallowed a real exception (bare except, log-and-continue)

3. Test oracle failure (the assertion itself may be wrong)
   - assertEqual on objects whose __eq__ compares the wrong thing
   - assertRaises caught — but raised by setup, not by the bug under test
   - Float compared without tolerance; dict / set order dependence
   - Fixture itself is broken; test never reached the target code

4. Off-by-one / boundary
   - Slice inclusive / exclusive bounds; range upper bound dropped n
   - Index 0 / -1 / out-of-range; empty string / list treated as falsy

5. State / side-effect leakage (very Pythonic)
   - Module- or class-level mutable state shared across cases
   - Generator already consumed once; second iteration yields nothing
   - Stale cache / lru_cache / memoization
   - In-place mutation vs returning a new object

6. Contract drift across the call site
   - Return type / raised exception class / signature changed; caller not updated
   - kwargs name drift; *args swallowed positional args

7. Import / patch target mislocation
   - mock.patch("a.b.foo") patched the reference, not the definition site
   - Import-time side effects; partial state from circular import

8. Environment dependence
   - cwd, env var, locale, timezone, path separator
   - Local cache dir leftovers; tmp_path not cleaned
