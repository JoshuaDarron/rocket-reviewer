# CLAUDE.md

## Project Overview

**rocketride-reviewer** is a GitHub Action (composite action) that performs multi-agent AI code reviews on pull requests using RocketRide's pipeline platform. Three independent AI reviewer agents (Claude, GPT, Gemini) run in parallel, each posting reviews under its own GitHub App identity. A Claude-based aggregator deduplicates overlapping comments before results are posted. The tool supports recursive, scoped conversation where developers reply to a specific agent and only that agent responds.

For full requirements, architecture decisions, and rationale, see `docs/PRD.md`.

---

## Repository Structure

```
rocketride-reviewer/
├── action.yml                      # Composite action definition (entry point)
├── Dockerfile                      # RocketRide engine container
├── docs/
│   ├── PRD.md                      # Product requirements document
│   ├── PLANNING.md                 # Implementation plan and task breakdown
│   └── SETUP.md                    # GitHub App setup guide for users
├── pipelines/
│   ├── full-review.pipe.json                # RocketRide pipeline: 3-agent parallel review
│   ├── conversation-reply-claude.pipe.json  # RocketRide pipeline: Claude conversation reply
│   ├── conversation-reply-openai.pipe.json  # RocketRide pipeline: GPT conversation reply
│   └── conversation-reply-gemini.pipe.json  # RocketRide pipeline: Gemini conversation reply
├── src/
│   ├── __init__.py
│   ├── main.py                     # Entry point: event detection, gating, orchestration
│   ├── github_client.py            # GitHub API wrapper (PR data, diff, comments, reviews)
│   ├── engine.py                   # RocketRide engine lifecycle (start, connect, teardown)
│   ├── pipeline.py                 # Pipeline execution (full review and conversation modes)
│   ├── aggregator.py               # Post-processing: parse agent output, route to posting
│   ├── reviewer.py                 # Review posting logic (per GitHub App)
│   ├── chunker.py                  # Large PR diff chunking and line number remapping
│   ├── config.py                   # Configuration loading and defaults
│   ├── models.py                   # Pydantic models (ReviewComment, AgentReview, etc.)
│   ├── filters.py                  # File filtering logic (ignore patterns)
│   └── errors.py                   # Exception hierarchy (BaseReviewerError and subclasses)
├── tests/
│   ├── __init__.py
│   ├── conftest.py                 # Shared fixtures (mock PR data, mock agent responses)
│   ├── test_main.py                # Gating logic tests
│   ├── test_github_client.py       # GitHub API interaction tests
│   ├── test_engine.py              # Engine lifecycle tests
│   ├── test_pipeline.py            # Pipeline execution tests
│   ├── test_aggregator.py          # Deduplication logic tests
│   ├── test_reviewer.py            # Review posting tests
│   ├── test_chunker.py             # Diff chunking and remapping tests
│   ├── test_config.py              # Configuration loading tests
│   ├── test_models.py              # Schema validation tests
│   └── test_filters.py             # File filtering tests
├── pyproject.toml                  # Project config, dependencies, tool settings
├── requirements.txt                # Pinned production dependencies
├── requirements-dev.txt            # Dev/test dependencies
├── CLAUDE.md                       # This file
├── README.md                       # User-facing documentation
├── LICENSE                         # MIT
└── .github/
    └── workflows/
        ├── ci.yml                  # Lint, type check, test on push/PR
        └── example-usage.yml       # Example workflow showing action usage
```

### Key boundaries

- `pipelines/` contains RocketRide pipeline JSON files only. No Python logic lives here.
- `src/` contains all Python orchestration. Each module has a single responsibility.
- `tests/` mirrors `src/` one-to-one. Every source module has a corresponding test module.
- `action.yml` is the public interface. It defines inputs, calls Docker, and invokes `src/main.py`.

---

## Tech Stack

| Layer | Technology | Version |
|-------|-----------|---------|
| Language | Python | 3.12+ |
| Pipeline Engine | RocketRide (Docker) | v1.0.1 |
| RocketRide SDK | `rocketride` (Python) | Latest |
| GitHub API | PyGithub | Latest |
| Data Validation | Pydantic v2 | Latest |
| HTTP | httpx | Latest |
| Linting | Ruff | Latest |
| Formatting | Black | Latest (88 char line length) |
| Type Checking | mypy (strict mode) | Latest |
| Testing | pytest + pytest-asyncio | Latest |
| CI | GitHub Actions | N/A |

---

## Python Conventions

### Style

- **Formatter**: Black with default settings (88 char line length).
- **Linter**: Ruff. Follow all default rules plus `I` (isort), `UP` (pyupgrade), `B` (bugbear), `SIM` (simplify), and `PTH` (pathlib).
- **Type hints**: Required on all function signatures. Use `from __future__ import annotations` at the top of every module. Prefer `str | None` over `Optional[str]`.
- **Imports**: Sorted by Ruff/isort. Standard library, third-party, then local. No wildcard imports.
- **Naming**: snake_case for functions and variables. PascalCase for classes. UPPER_SNAKE for constants. Prefix private helpers with underscore.
- **Docstrings**: Required on all public functions and classes. Use Google-style docstrings.
- **f-strings**: Preferred over `.format()` or `%` formatting in all cases.

### General patterns

- Use Pydantic models for all structured data (review comments, agent responses, config). Never pass raw dicts across module boundaries.
- Use `pathlib.Path` over `os.path` for all file operations.
- Use `httpx` for any HTTP calls outside of PyGithub. Prefer async where possible.
- Use `asyncio` for RocketRide SDK calls (the Python SDK is async context manager based).
- Keep functions short. If a function exceeds 40 lines, consider breaking it up. Use judgment; don't split for the sake of splitting if the logic is linear and readable.

---

## Error Handling

This is a CI tool. Errors are reported gracefully (PR comments where possible), and the action exits with code 1 on failure so the GitHub check accurately reflects the result. Whether a failed review blocks merging is controlled by the repo's branch protection settings, not by the action itself.

### Core principles

1. **Exit honestly.** The top-level entry point in `main.py` catches all exceptions, logs the error, posts a summary comment on the PR if possible, and exits with code 1. This ensures the GitHub check shows red on failure. Repo owners control whether this check is required for merging via branch protection rules.

2. **Agent failures are independent.** If one LLM agent fails (API timeout, rate limit, malformed response), the other two must still complete. Use individual try/except blocks per agent, not a shared one. Log the failure. Post a comment on the PR noting which agent was unavailable (e.g., "GPT reviewer was unavailable for this review due to an API timeout.").

3. **Distinguish retryable from terminal errors.** API rate limits and transient network errors get up to 3 retries with exponential backoff. Authentication failures, invalid configuration, and schema validation errors are terminal and reported immediately.

4. **RocketRide engine failures are terminal for the run, not for CI.** If the Docker container fails to start or the engine is unreachable, log the error, post a PR comment explaining the review could not run, and exit cleanly.

5. **GitHub API errors during posting are logged but not fatal.** If posting a review comment fails, log it and continue posting the remaining comments. Summarize any posting failures at the end.

### Error hierarchy

```
BaseReviewerError                # Base for all project exceptions
├── ConfigurationError           # Invalid config, missing secrets
├── EngineError                  # RocketRide engine startup/connection failures
├── PipelineError                # Pipeline execution failures
│   ├── AgentError               # Individual agent failure (contains agent name)
│   └── AggregatorError          # Aggregator failure
├── GitHubClientError            # GitHub API interaction failures
│   ├── DiffRetrievalError       # Failed to fetch PR diff
│   ├── CommentPostingError      # Failed to post a comment
│   └── ReviewSubmissionError    # Failed to submit review status
├── FilterError                  # File filtering configuration errors
└── ChunkingError                # Diff chunking failures (split, overlap, remap)
```

Define these in `src/errors.py` (add this file to the repo structure). All exceptions inherit from `BaseReviewerError`. Use specific exceptions, not generic ones. Every except block should catch the most specific exception possible.

### Retry pattern

```python
async def with_retry(
    fn: Callable,
    max_retries: int = 3,
    backoff_base: float = 1.0,
    retryable: tuple[type[Exception], ...] = (httpx.TimeoutException, httpx.NetworkError),
) -> Any:
    """Execute fn with exponential backoff on retryable errors."""
    for attempt in range(max_retries + 1):
        try:
            return await fn()
        except retryable as e:
            if attempt == max_retries:
                raise
            wait = backoff_base * (2 ** attempt)
            logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {wait}s.")
            await asyncio.sleep(wait)
```

---

## Testing

### Expectations

- **Every source module gets a test module.** No exceptions.
- **Unit tests are the primary layer.** Mock external dependencies (GitHub API, RocketRide SDK, LLM APIs). Tests should run fast and offline.
- **Integration tests are separate.** If we add tests that actually hit the RocketRide engine or GitHub API, they go in `tests/integration/` and are not run in standard CI. Mark them with `@pytest.mark.integration`.
- **Test the gating logic thoroughly.** The trigger conditions (target branch, event type, comment author detection, loop prevention) are the most critical path. Every branch in the gating logic needs a test case.
- **Test agent failure isolation.** Verify that one agent failing does not prevent the other two from completing.
- **Test the deduplication logic with real-world-like data.** The aggregator's dedup is a subtle area. Include test cases for exact duplicates, near-duplicates (same file/line, slightly different wording), and non-duplicates that happen to be on the same line.
- **Test Pydantic models with invalid data.** Verify that malformed agent responses are caught and handled, not silently passed through.
- **Test diff chunking thoroughly.** Verify file-boundary splits, function-boundary splits within large files, overlap context inclusion, and line number remapping after merge. Include edge cases: single-file PRs that exceed chunk limits, files with no detectable function boundaries, PRs at the exact threshold limits.
- **Test agent routing.** Verify that bot username to node ID mapping works for all three agents, and that unknown bot usernames are handled gracefully.

### Running tests

```bash
# All unit tests
pytest tests/ -v

# With coverage
pytest tests/ -v --cov=src --cov-report=term-missing

# Specific module
pytest tests/test_aggregator.py -v
```

### Fixtures

Define shared fixtures in `tests/conftest.py`:

- `mock_pr_diff`: A realistic multi-file diff string
- `mock_pr_comments`: A list of existing PR comments with thread structure
- `mock_agent_response_clean`: A valid agent response with zero critical/high comments
- `mock_agent_response_issues`: A valid agent response with mixed severity comments
- `mock_agent_response_malformed`: An invalid agent response (missing fields, wrong types)
- `mock_config_default`: Default configuration object
- `mock_config_custom`: Configuration with custom ignore patterns and diff-only mode
- `mock_large_diff`: A diff exceeding 500 lines for chunking tests
- `mock_oversized_pr`: A diff exceeding the max files/lines thresholds

---

## RocketRide Pipeline Conventions

### Model Versions and Agent Routing

These constants are defined in `src/config.py`:

```python
# Model identifiers (not user-configurable in v1)
MODELS = {
    "claude-reviewer": "claude-sonnet-4-6",
    "gpt-reviewer": "openai-5-2",
    "gemini-reviewer": "gemini-3-pro",
    "aggregator": "claude-sonnet-4-6",
}

# Pipeline file paths
FULL_REVIEW_PIPELINE_FILE = "full-review.pipe.json"

# Maps agent node IDs to per-agent conversation reply pipeline filenames
CONVERSATION_PIPELINE_FILES = {
    "claude-reviewer": "conversation-reply-claude.pipe.json",
    "gpt-reviewer": "conversation-reply-openai.pipe.json",
    "gemini-reviewer": "conversation-reply-gemini.pipe.json",
}

# Maps response lane names to reviewer node IDs
LANE_TO_REVIEWER = {
    "claude": "claude-reviewer",
    "openai": "gpt-reviewer",
    "gemini": "gemini-reviewer",
}

# Maps GitHub App bot username to RocketRide pipeline node ID
AGENT_ROUTING = {
    "claude-reviewer[bot]": "claude-reviewer",
    "gpt-reviewer[bot]":    "gpt-reviewer",
    "gemini-reviewer[bot]": "gemini-reviewer",
}

# All bot usernames (for loop prevention)
BOT_USERNAMES = set(AGENT_ROUTING.keys())
```

Never hardcode model names or bot usernames outside of `src/config.py`. Always import from config.

### Pipeline JSON files

- Live in `pipelines/` directory. Four files: `full-review.pipe.json` and three per-agent conversation reply files (`conversation-reply-claude.pipe.json`, `conversation-reply-openai.pipe.json`, `conversation-reply-gemini.pipe.json`).
- Node IDs should be descriptive: `claude-reviewer`, `gpt-reviewer`, `gemini-reviewer`, `claude-aggregator`, not `node-1`, `node-2`.
- Each reviewer node receives the same input schema. The pipeline definition handles the fan-out, not the Python orchestration.
- The aggregator node's output schema must match the per-agent JSON payload structure defined in the PRD (see Section 5.2 and 5.3).

### Interacting with the engine

- Start the engine via Docker before pipeline execution. Poll `localhost:5565` every 2 seconds for up to 30 seconds. If the engine does not respond within this window, abort with a graceful failure.
- Use the RocketRide Python SDK async context manager pattern for all engine interactions.
- Always call `terminate()` on the pipeline token after execution, even on failure. Use try/finally.
- Engine teardown (Docker stop) happens in a finally block in `main.py`, regardless of success or failure.

### Large PR chunking

The chunking logic lives in `src/chunker.py`. Key rules:

- Split diffs at file boundaries first.
- If a single file diff exceeds `max_chunk_lines` (default 500), split at function/class boundaries where detectable, otherwise at blank line boundaries.
- Include `chunk_overlap_lines` (default 20) lines of overlap between segments for context continuity.
- After all chunks are reviewed, merge comment lists and remap line numbers back to the original diff coordinates.
- If a PR exceeds `max_files` (50) or `max_total_lines` (5,000) after filtering, post a summary comment and exit without reviewing.
- All thresholds are configurable via `.rocketride-review.yml`.

---

## GitHub App Conventions

- Each of the three GitHub Apps (`claude-reviewer[bot]`, `gpt-reviewer[bot]`, `gemini-reviewer[bot]`) authenticates independently using its own App ID and private key.
- Authentication tokens are generated per-run using the GitHub App installation token flow. Tokens are held in memory only.
- When posting reviews, each app posts only its own comments. Never post another agent's comments under the wrong app identity.
- For loop prevention: before processing an `issue_comment` event, check if the comment author is any of the three app bot usernames. If so, exit immediately.

---

## Git Conventions

- Branch from `main` for all work.
- Commit messages are short and imperative: "Add aggregator dedup logic", "Fix agent timeout handling", "Update pipeline schema".
- No commit message format enforcement tooling in v1, but keep them clean and descriptive.
- PRs should target `main` and include a description of what changed and why.
- **Do NOT add `Co-Authored-By` lines to commit messages.** Commits should contain only the message itself.

### Pre-commit checks

Before every commit, run **all four checks** in order. If any check fails, fix the issue, then re-run **that check** until it passes before moving to the next. Only commit once all four pass.

```bash
# 1. Tests
pytest tests/ -v

# 2. Lint
ruff check src/ tests/

# 3. Type check
mypy src/

# 4. Format check
black --check src/ tests/
```

If `black --check` fails, run `black src/ tests/` to auto-format, then re-run the check. For `ruff` auto-fixable errors, run `ruff check --fix src/ tests/`.

---

## What Not to Do

- **Do not put orchestration logic in the pipeline JSON.** The pipelines handle LLM execution and data flow. Gating, GitHub API calls, comment posting, and approval logic live in Python.
- **Do not catch bare `Exception` anywhere except `main.py`'s top-level handler.** Catch specific exceptions everywhere else.
- **Do not log or print API keys, tokens, or private keys.** Ever. Not even partially masked.
- **Do not add persistent state (databases, files written to the repo, external services).** v1 is stateless. Every run is independent.
- **Exit honestly.** The action exits with code 1 on failure so checks reflect reality. Whether the check blocks merging is controlled by branch protection settings, not the action.
- **Do not hardcode model names or API endpoints.** These should be configurable or at minimum defined as constants in a single location, not scattered through the codebase.
