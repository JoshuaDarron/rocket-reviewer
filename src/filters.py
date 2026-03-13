"""File filtering logic for excluding files from review.

Matches files against configurable ignore patterns (fnmatch/glob style).
Supports default patterns, user extensions, and full overrides.
"""

from __future__ import annotations

import fnmatch
from pathlib import PurePosixPath

from src.config import DEFAULT_IGNORE_PATTERNS


def should_ignore(file_path: str, patterns: list[str]) -> bool:
    """Check if a file path matches any of the ignore patterns.

    Supports simple ``fnmatch`` patterns (e.g., ``*.lock``) and recursive
    glob patterns with ``**`` (e.g., ``dist/**``).

    Args:
        file_path: Relative file path to check. Backslashes are
            normalized to forward slashes.
        patterns: List of glob/fnmatch patterns.

    Returns:
        True if the file should be ignored.
    """
    # Normalize to forward slashes
    normalized = file_path.replace("\\", "/")
    path = PurePosixPath(normalized)

    for pattern in patterns:
        if "/" in pattern or "**" in pattern:
            # Directory pattern — match against the full path using fnmatch
            # Convert ** to fnmatch-compatible wildcard
            fnmatch_pattern = pattern.replace("**/", "*/")
            if fnmatch.fnmatch(normalized, fnmatch_pattern):
                return True
            # Also try matching with any prefix for nested paths
            # e.g., "node_modules/**" should match "node_modules/foo/bar.js"
            if "**" in pattern:
                prefix = pattern.split("**")[0]
                if normalized.startswith(prefix):
                    return True
        else:
            # Simple filename pattern — match against the file name only
            if fnmatch.fnmatch(path.name, pattern):
                return True

    return False


def get_effective_patterns(
    extra: list[str] | None = None,
    override: list[str] | None = None,
) -> list[str]:
    """Build the effective ignore pattern list.

    If override is provided, it replaces the defaults entirely.
    Otherwise, extra patterns are appended to the defaults.

    Args:
        extra: Additional patterns to append to defaults.
        override: If set, replaces all default patterns.

    Returns:
        The effective list of ignore patterns.
    """
    if override is not None:
        return list(override)

    patterns = list(DEFAULT_IGNORE_PATTERNS)
    if extra:
        patterns.extend(extra)
    return patterns
