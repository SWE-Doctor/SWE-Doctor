# SWE-Doctor

This repository is the research artifact for our ICSE 2027 submission. SWE-Doctor is a software issue resolution agent that guides patch generation with runtime diagnoses derived from bug reproduction test executions. Given an issue report and a buggy repository, it generates one bug reproduction test per behavior the issue states, executes and debugs those tests to build runtime-grounded diagnosis records, and hands the diagnoses together with localization evidence to a patch-generation agent. This reframes a bug reproduction test as a debugging entry point rather than a pass/fail target the agent must satisfy, which reduces partial patches that fix one stated behavior and leave others unaddressed.

The artifact implements this method as a pipeline that turns a natural-language issue report into a patch and then grades that patch with an official benchmark harness. The pipeline runs on SWE-bench Verified and SWE-bench Pro. The debugging stage is instantiated with the Python debugger pdb for Python issues and with Delve for Go issues, following the paper's design that the same debugging interface can be instantiated per language.

The agent scaffold is built on mini-swe-agent, whose source lives under `src/minisweagent/`. The base agent gives a language model bash access inside a Docker container. Documentation for that base agent is at https://mini-swe-agent.com. Everything outside `src/minisweagent/` is our own contribution.

## Requirements

- Python 3.10 or newer.
- Docker, for every stage that runs code inside a benchmark container. Stages 1, 2, and 4 launch one container per instance.
- A model endpoint reachable through litellm, plus the matching API key. See the Configuration section.

## Setup

Create an isolated environment and install the package together with its development tools. The commands below use `uv`, but any tool that reads `pyproject.toml` works.

```bash
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[dev]" swebench
```

This installs the four-stage pipeline, the `src/minisweagent/` scaffold in editable mode, the test tools, and the SWE-bench grading harness used in Stage 4.

## Vocabulary

- Instance: one bug-fix task from a benchmark. Each instance has a failing repository state, an issue description, and a hidden set of tests that the correct fix must pass.
- Bug reproduction test, abbreviated BRT: a test that we generate, which fails on the buggy code and is expected to pass once the bug is fixed.
- Behavioral requirement: one observable behavior stated in the issue report that a correct patch must satisfy, such as an expected output, an exception condition, or a compatibility requirement. SWE-Doctor generates one targeted BRT per requirement, which is what "multi-faceted" refers to. In the code this term appears as an "aspect".
- Diagnosis record: the structured output of the debugging stage for one BRT. It holds the suspected fault location, the runtime failure symptom, the failure propagation path, the patch impact on related code, and a suggested fix direction, together with the runtime evidence that grounds them. It is written as `<instance_id>_rca.json`.
- Run directory: one timestamped folder that holds the output of every stage for a single pipeline run. Each stage reads the previous stage's output from this folder and writes its own output back into it.

## Pipeline stages

| Stage | Purpose | Module |
|-------|---------|--------|
| 1 | Multi-faceted BRT generation: extract behavioral requirements and generate one targeted BRT per requirement | `reproduction_test_agent/` |
| 2a | Stage the accepted BRTs inside the instance container and record an execution trace | `run_pro_test/`, `run_verified_test/` |
| 2b | Runtime-grounded bug diagnosis: debug each BRT and write a diagnosis record | `debug_agent/` |
| 3 | Diagnosis-guided patch generation: cross-reference diagnoses with localization and generate the patch | `repair_agent/` |
| 4 | Apply the patch and grade it with the official benchmark harness | `swebench_pro_baseline/`, `swebench_verified_eval/` |

### Stage 1: multi-faceted BRT generation

`reproduction_test_agent/` first decomposes the issue report into behavioral requirements, the observable behaviors a correct patch must satisfy. For each requirement it localizes a ranked list of source files and focal functions, then generates a targeted BRT through a generate, execute, and refine loop: the model writes a candidate test, runs it inside the instance container, and revises it until the test fails for the behavior the requirement states. A screening step keeps a BRT only when it fails after reaching the localized target code, through an assertion failure or a program exception, and discards tests that abort on environment or dependency errors before reaching that code. The output is a set of BRTs, each linked to a behavioral requirement and paired with its localized files and functions. Requirement extraction lives in `aspect_extractor.py` and requirement-level localization in `localizer.py`.

### Stage 2: runtime-grounded bug diagnosis

Stage 2a stages each accepted BRT and produces an execution trace. Stage 2b runs an LLM debugging agent that drives a runtime debugger over the failing BRT, inspects the live program state, and fills a diagnosis record. For Python it drives the Python debugger pdb. For Go it drives Delve, the Go debugger, invoked as `dlv`. The agent moves between stack frames, sets breakpoints, re-runs the BRT, and reads runtime values, then works backward from the failure symptom toward a suspected fault location. A runtime-grounding check accepts a diagnosis only when the agent has collected runtime evidence through execution feedback, debugger interaction, or runtime probing, so a diagnosis cannot rest on the issue text and source code alone. The result is written as `<instance_id>_rca.json`.

### Stage 3: diagnosis-guided patch generation

`repair_agent/` builds on mini-swe-agent and leaves its agent loop unchanged, altering only the context the agent reads before patch generation. It combines two sources: the behavioral requirements with their localization from Stage 1, and the diagnosis records from Stage 2. When both point to the same location, SWE-Doctor presents it as a strong edit candidate; when they diverge, it presents both and asks the agent to compare them against the issue report, the diagnoses, and the code before deciding where to edit. Before submission, a completeness check asks the agent to revisit every extracted requirement and diagnosis field and confirm the patch addresses each, which reduces partial patches. If no diagnosis record is produced, the stage falls back to the default mini-swe-agent workflow. The output is a `preds.json` file holding one predicted patch per instance.

### Stage 4: evaluation

Stage 4 applies each predicted patch and runs the benchmark's own grading harness. `swebench_verified_eval/` wraps the official SWE-bench harness for Verified. `swebench_pro_baseline/` wraps the SWE-bench Pro harness and also provides the bash-only baseline agent we compare against.

### Language support

The pipeline handles Python and Go issues. The language is not inferred from the instance; it is chosen by an explicit `--language` flag that takes `python` or `go` and defaults to `python`. Each stage reads this flag and routes to the language-specific implementation. The two driver scripts fix the language for you: `run_pipeline_verified.sh` runs the Python-only Verified modules, and `run_pipeline_pro_go.sh` passes `--language go` to Stages 1 and 2b and uses the Go repair config.

Each stage branches as follows:

- Stage 1 reads a per-language `LangPack` from `reproduction_test_agent/langpack.py`, the single source of language-specific settings such as the test filename, the generation prompt, source-file globs, and error-type patterns. All other Stage 1 modules consult this pack rather than branching on the language themselves.
- Stage 2b keeps one debugger action interface for the LLM agent, so the agent emits the same commands regardless of language. The debugger backend behind those commands is swapped by injection: the Python path uses `PdbSession` to drive pdb, and the Go path installs Delve and injects a `GoDlvSession` that drives `dlv test`. The Go path also locates `*_test.go` targets instead of `*.py` and skips the Python-only steps.
- Stage 3 selects a per-language repair config, `repair_config_pro_go.yaml` for Go and `repair_config_verified.yaml` for Python.

## Repository layout

| Path | Contents |
|------|----------|
| `reproduction_test_agent/` | Stage 1 multi-faceted BRT generation and its batch runners |
| `run_pro_test/` | Stage 2a staging and tracing for SWE-bench Pro |
| `run_verified_test/` | Stage 2a and 2b dispatchers for SWE-bench Verified |
| `debug_agent/` | Stage 2b debugger-driven diagnosis for Python and Go |
| `repair_agent/` | Stage 3 diagnosis-guided patch generation, prompts, and config |
| `swebench_pro_baseline/` | Stage 4 grading for SWE-bench Pro, plus the bash-only baseline |
| `swebench_verified_eval/` | Stage 4 grading for SWE-bench Verified |
| `pipeline_run/` | End-to-end driver scripts that chain all four stages |
| `src/minisweagent/` | Upstream mini-swe-agent scaffold |
| `verified_common.py` | Shared constants for the Verified container environment |
| `tests/` | Unit tests for the modules above |
| `docs/`, `pipeline_docs/` | Design notes and the mini-swe-agent documentation site |

## Quick check without Docker

Before launching real runs, you can confirm the wiring with three dry-run commands. Each one loads the benchmark dataset, selects the requested instances, and exits before any container starts. They need network access to download the dataset, but no Docker and no API key. Put one instance id per line in an ids file first.

```bash
# Stage 1 selection: prints each instance and the Docker image it would use.
python -m reproduction_test_agent.run_batch \
  --issue-ids-file ids.txt --output /tmp/s1 --language go --dry-run

# Subset selection for the SWE-bench Pro baseline.
python swebench_pro_baseline/run_subset.py \
  --dataset ScaleAI/SWE-bench_Pro --issue-ids-file ids.txt --dry-run

# Stage 4 selection: prints which predictions would be graded.
python swebench_pro_baseline/evaluate.py \
  -o <stage3-output-dir> --dataset ScaleAI/SWE-bench_Pro --dry-run
```

To confirm that every entry point imports and parses its arguments, append `--help` to any module or script listed in the Repository layout section.

## Running the pipeline

Two driver scripts chain all four stages. Both take an issue-id list with one instance id per line.

SWE-bench Verified:

```bash
bash pipeline_run/run_pipeline_verified.sh <run-dir> <issue-ids-file>
```

SWE-bench Pro, Go track:

```bash
PIPELINE_ENV=pipeline_run/backend_env.local.sh \
  bash pipeline_run/run_pipeline_pro_go.sh <issue-ids-file> [run-dir]
```

### Running one stage at a time

The driver scripts call these entry points in order. Run them by hand when you want to inspect a single stage. Each `--help` lists the full set of options.

| Stage | SWE-bench Pro | SWE-bench Verified |
|-------|---------------|--------------------|
| 1 | `python -m reproduction_test_agent.run_batch` | `python -m reproduction_test_agent.run_batch_verified` |
| 2a | `python run_pro_test/stage_go_repro.py` for Go, otherwise `python run_pro_test/run_repro_trace.py` | `python -m run_verified_test.run_repro_trace_verified` |
| 2b | `python -m debug_agent.run_debug` | `python -m run_verified_test.run_rca_verified` |
| 3 | `python -m repair_agent.run_repair` | `python -m repair_agent.run_repair_verified` |
| 4 | `python swebench_pro_baseline/evaluate.py` | `python -m swebench_verified_eval.run_eval_verified` |

## Configuration

Some inputs live outside the repository, such as the benchmark dataset checkout and your API credentials. Every such input is read from an environment variable so the scripts stay portable. Set the variables that match your run.

| Variable | Meaning |
|----------|---------|
| `SWEBENCH_PRO_OS_ROOT` | Path to your SWE-bench Pro dataset checkout. Defaults to `SWE-bench_Pro-os` next to the repository. |
| `SWEBENCH_PRO_ISSUE_IDS` | Path to a default issue-id list for the Pro runners. |
| `UTA_DATA_ROOT` | Parent directory that holds external datasets. Defaults to the repository's parent directory. |
| `PIPELINE_ENV` | Path to a shell file that exports the backend model name and provider keys for the Pro Go pipeline. See `pipeline_run/backend_env.example.sh`. |
| `DEEPSEEK_ENV` | Optional credentials file to run the Verified pipeline on a DeepSeek endpoint. |

Credential files are not committed. For the Verified pipeline, place Anthropic credentials in `claude.env` at the repository root. For the Pro Go pipeline, copy `pipeline_run/backend_env.example.sh` to a local file and fill in your keys.

## Tests

Tests that need a live container are marked `docker`. Skip them when Docker is absent:

```bash
python -m pytest tests/ -m "not docker"
```

Two notes for a fresh checkout:

- The tests under `tests/run/test_cli_integration.py` invoke the `mini` console script through a subprocess, so the script must be on `PATH`. Activate the virtual environment from the Setup section before running them.
- `tests/debug_agent/test_container_go.py` copies a precompiled Delve binary from `debug_agent/assets/`. That binary is not committed, so this one test fails until you place the binary there. The pipeline still builds Delve inside the container when the binary is missing.

The two collection errors under `tests/environments/extra/` belong to the upstream scaffold and need the optional `modal` and `contree` packages. Deselect that directory if you have not installed those extras.

## License

See `LICENSE.md`. The upstream mini-swe-agent scaffold under `src/minisweagent/` retains its original license.
