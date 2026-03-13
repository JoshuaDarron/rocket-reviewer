"""Tests for file filtering logic."""

from __future__ import annotations

from src.config import DEFAULT_IGNORE_PATTERNS
from src.filters import get_effective_patterns, should_ignore


class TestShouldIgnore:
    """Tests for should_ignore()."""

    def test_simple_extension_match(self) -> None:
        assert should_ignore("package-lock.json", ["package-lock.json"])

    def test_wildcard_extension(self) -> None:
        assert should_ignore("styles.min.css", ["*.min.css"])

    def test_recursive_glob(self) -> None:
        assert should_ignore("dist/bundle.js", ["dist/**"])

    def test_nested_recursive_glob(self) -> None:
        assert should_ignore("node_modules/foo/bar.js", ["node_modules/**"])

    def test_no_match(self) -> None:
        assert not should_ignore("src/main.py", ["*.lock", "dist/**"])

    def test_lock_file_default(self) -> None:
        assert should_ignore("poetry.lock", DEFAULT_IGNORE_PATTERNS)

    def test_image_file_default(self) -> None:
        assert should_ignore("logo.png", DEFAULT_IGNORE_PATTERNS)
        assert should_ignore("icon.svg", DEFAULT_IGNORE_PATTERNS)

    def test_source_file_not_ignored(self) -> None:
        assert not should_ignore("src/config.py", DEFAULT_IGNORE_PATTERNS)

    def test_windows_path_normalization(self) -> None:
        """Backslashes are normalized to forward slashes."""
        assert should_ignore("dist\\bundle.js", ["dist/**"])
        assert should_ignore("node_modules\\foo\\bar.js", ["node_modules/**"])

    def test_generated_file_pattern(self) -> None:
        assert should_ignore("api.generated.ts", ["*.generated.*"])

    def test_font_files(self) -> None:
        assert should_ignore("fonts/roboto.woff2", ["*.woff2"])


class TestGetEffectivePatterns:
    """Tests for get_effective_patterns()."""

    def test_defaults_when_no_args(self) -> None:
        patterns = get_effective_patterns()
        assert patterns == DEFAULT_IGNORE_PATTERNS

    def test_extra_extends_defaults(self) -> None:
        patterns = get_effective_patterns(extra=["*.sql", "migrations/**"])
        assert "*.sql" in patterns
        assert "migrations/**" in patterns
        # Defaults are still included
        assert "*.lock" in patterns

    def test_override_replaces_defaults(self) -> None:
        override = ["*.custom"]
        patterns = get_effective_patterns(override=override)
        assert patterns == ["*.custom"]
        assert "*.lock" not in patterns

    def test_override_takes_precedence_over_extra(self) -> None:
        """When both override and extra are given, override wins."""
        patterns = get_effective_patterns(extra=["*.sql"], override=["*.custom"])
        assert patterns == ["*.custom"]
        assert "*.sql" not in patterns

    def test_empty_extra(self) -> None:
        patterns = get_effective_patterns(extra=[])
        assert patterns == DEFAULT_IGNORE_PATTERNS

    def test_empty_override(self) -> None:
        patterns = get_effective_patterns(override=[])
        assert patterns == []
