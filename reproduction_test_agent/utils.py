"""Shared utilities for the reproduction test agent."""

# Fence-language tags we strip from a leading ```<lang> line. Language-agnostic
# so go/js repro code extracts cleanly the same way python does.
_FENCE_LANGS = {
    "python", "py", "go", "golang", "javascript", "js",
    "typescript", "ts", "jsx", "tsx", "mjs", "cjs", "",
}


def extract_code(response: str) -> str:
    """Extract code from an LLM response, handling markdown fences.

    Language-agnostic: keeps the original ```python fast-path, and otherwise
    strips a leading fence-language tag line (go/js/ts/...) before returning."""
    if "```python" in response:
        blocks = response.split("```python")
        if len(blocks) > 1:
            code = blocks[1].split("```")[0]
            return code.strip()
    if "```" in response:
        blocks = response.split("```")
        if len(blocks) >= 3:
            code = blocks[1]
            lines = code.splitlines()
            if lines and lines[0].strip().lower() in _FENCE_LANGS:
                lines = lines[1:]
            return "\n".join(lines).strip()
    return response.strip()
