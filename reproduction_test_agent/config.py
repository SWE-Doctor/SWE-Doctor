from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ReproTestConfig:
    # LLM settings
    model_name: str = "openai/gpt-5.4"
    language: str = "python"   # "python" | "go"
    generation_temperature: float = 0.0
    repair_temperature: float = 0.8
    morph_temperature: float = 0.0

    # Repair loop
    max_repair_attempts: int = 10

    # Execution
    test_timeout: int = 60
    repo_path: str = ""

    # Which prompts to use (subset of all 10)
    morphs: list[str] = field(default_factory=lambda: [
        "standard", "simple", "dropCode", "initTest", "initPatch",
    ])
    masks: list[str] = field(default_factory=lambda: [
        "planner", "full", "testLoc", "patchLoc", "none",
    ])

    # Aspect mode
    aspects: bool = True
    max_aspects: int = 4
    aspect_masks: tuple = ("none", "patchLoc")

    # Output
    output_dir: str = "results"
