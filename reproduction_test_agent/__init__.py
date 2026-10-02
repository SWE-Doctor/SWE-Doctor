"""e-Otter++ style reproduction test generator for SWE-bench Pro.

Pipeline: localize → heterogeneous prompting (morphs + masks) → generate tests
→ execution-augmented repair (critic loop) → filter by fail-for-right-reason.
"""
