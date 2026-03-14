"""Tests for diff chunking and line number remapping."""

from __future__ import annotations

import pytest

from src.chunker import (
    ChunkResult,
    _split_into_file_diffs,
    chunk_diff,
    chunk_diff_detailed,
    remap_line_numbers,
)
from src.errors import ChunkingError

# ---------------------------------------------------------------------------
# Helpers to build synthetic diffs
# ---------------------------------------------------------------------------


def _make_file_diff(filename: str, num_lines: int, prefix: str = "+") -> str:
    """Build a synthetic single-file diff with *num_lines* changed lines."""
    header = (
        f"diff --git a/{filename} b/{filename}\n"
        f"index 0000000..1111111 100644\n"
        f"--- a/{filename}\n"
        f"+++ b/{filename}\n"
        f"@@ -1,{num_lines} +1,{num_lines} @@\n"
    )
    body = "".join(f"{prefix}    line_{i} = {i}\n" for i in range(num_lines))
    return header + body


def _make_file_diff_with_functions(filename: str, funcs: int, lines_per: int) -> str:
    """Build a diff with function definitions.

    Each of *funcs* functions is followed by *lines_per* body lines.
    """
    header = (
        f"diff --git a/{filename} b/{filename}\n"
        f"index 0000000..1111111 100644\n"
        f"--- a/{filename}\n"
        f"+++ b/{filename}\n"
        f"@@ -1,1000 +1,1000 @@\n"
    )
    body_parts: list[str] = []
    for f in range(funcs):
        body_parts.append(f"+def func_{f}():\n")
        for j in range(lines_per):
            body_parts.append(f"+    x_{f}_{j} = {j}\n")
    return header + "".join(body_parts)


def _make_file_diff_with_blanks(filename: str, num_lines: int, blank_every: int) -> str:
    """Build a diff that has a blank line every *blank_every* lines."""
    header = (
        f"diff --git a/{filename} b/{filename}\n"
        f"index 0000000..1111111 100644\n"
        f"--- a/{filename}\n"
        f"+++ b/{filename}\n"
        f"@@ -1,{num_lines} +1,{num_lines} @@\n"
    )
    body_parts: list[str] = []
    for i in range(num_lines):
        if i > 0 and i % blank_every == 0:
            body_parts.append("+\n")
        else:
            body_parts.append(f"+    line_{i} = {i}\n")
    return header + "".join(body_parts)


# ---------------------------------------------------------------------------
# _split_into_file_diffs
# ---------------------------------------------------------------------------


class TestSplitIntoFileDiffs:
    """Tests for the internal file-splitting helper."""

    def test_empty_diff(self) -> None:
        """Empty string produces empty list."""
        assert _split_into_file_diffs("") == []

    def test_whitespace_only(self) -> None:
        """Whitespace-only string produces empty list."""
        assert _split_into_file_diffs("   \n\n  ") == []

    def test_single_file(self) -> None:
        """Single file diff returns one pair."""
        diff = _make_file_diff("foo.py", 5)
        result = _split_into_file_diffs(diff)
        assert len(result) == 1
        assert result[0][0] == "foo.py"
        assert "foo.py" in result[0][1]

    def test_multiple_files(self) -> None:
        """Multi-file diff returns one pair per file."""
        diff = _make_file_diff("a.py", 3) + _make_file_diff("b.py", 4)
        result = _split_into_file_diffs(diff)
        assert len(result) == 2
        assert result[0][0] == "a.py"
        assert result[1][0] == "b.py"

    def test_no_git_header(self) -> None:
        """Diff with no 'diff --git' header falls back to unknown filename."""
        result = _split_into_file_diffs("@@ -1,3 +1,3 @@\n+ hello\n")
        assert len(result) == 1
        assert result[0][0] == "unknown"


# ---------------------------------------------------------------------------
# chunk_diff — basic cases
# ---------------------------------------------------------------------------


class TestChunkDiffBasic:
    """Tests for chunk_diff with small and simple diffs."""

    def test_empty_diff_returns_empty(self) -> None:
        """An empty diff string should return an empty list."""
        assert chunk_diff("") == []

    def test_whitespace_only_returns_empty(self) -> None:
        """A whitespace-only diff should return an empty list."""
        assert chunk_diff("  \n\n  ") == []

    def test_single_file_under_threshold(self) -> None:
        """A single file under max_chunk_lines returns one chunk."""
        diff = _make_file_diff("small.py", 10)
        result = chunk_diff(diff, max_chunk_lines=500)
        assert len(result) == 1
        assert "small.py" in result[0]

    def test_multi_file_under_threshold(self) -> None:
        """Multiple files each under threshold return one chunk per file."""
        diff = _make_file_diff("a.py", 10) + _make_file_diff("b.py", 15)
        result = chunk_diff(diff, max_chunk_lines=500)
        assert len(result) == 2
        assert "a.py" in result[0]
        assert "b.py" in result[1]

    def test_single_line_change(self) -> None:
        """A PR with a single line change produces exactly one chunk."""
        diff = _make_file_diff("tiny.py", 1)
        result = chunk_diff(diff, max_chunk_lines=500)
        assert len(result) == 1

    def test_binary_file_diff(self) -> None:
        """Binary file markers in a diff are handled without error."""
        diff = (
            "diff --git a/image.png b/image.png\n"
            "index 0000000..1111111 100644\n"
            "Binary files /dev/null and b/image.png differ\n"
        )
        result = chunk_diff(diff, max_chunk_lines=500)
        assert len(result) == 1
        assert "Binary files" in result[0]


# ---------------------------------------------------------------------------
# chunk_diff — file boundary splitting
# ---------------------------------------------------------------------------


class TestChunkDiffFileBoundary:
    """Tests for splitting at file boundaries in multi-file diffs."""

    def test_three_files_produce_three_chunks(self) -> None:
        """Three small files produce three separate chunks."""
        diff = (
            _make_file_diff("x.py", 5)
            + _make_file_diff("y.py", 5)
            + _make_file_diff("z.py", 5)
        )
        result = chunk_diff(diff, max_chunk_lines=500)
        assert len(result) == 3

    def test_file_boundary_content_integrity(self) -> None:
        """Each chunk contains only its own file's content."""
        diff = _make_file_diff("first.py", 3) + _make_file_diff("second.py", 3)
        result = chunk_diff(diff, max_chunk_lines=500)
        assert "first.py" in result[0]
        assert "second.py" not in result[0]
        assert "second.py" in result[1]
        assert "first.py" not in result[1]


# ---------------------------------------------------------------------------
# chunk_diff — large file splitting
# ---------------------------------------------------------------------------


class TestChunkDiffLargeFile:
    """Tests for sub-splitting large single-file diffs."""

    def test_large_file_splits_at_function_boundaries(self) -> None:
        """A large file with function defs should split at them."""
        # 10 functions * 60 lines each = 600 body lines + headers > 500
        diff = _make_file_diff_with_functions("big.py", funcs=10, lines_per=60)
        result = chunk_diff(diff, max_chunk_lines=200, overlap_lines=10)
        assert len(result) > 1
        # Each chunk should be manageable in size
        for chunk in result:
            # Allow some slack for overlap
            assert len(chunk.splitlines()) <= 250

    def test_large_file_no_functions_splits_at_blank_lines(self) -> None:
        """A large file with no function boundaries splits at blank lines."""
        diff = _make_file_diff_with_blanks("data.py", num_lines=600, blank_every=50)
        result = chunk_diff(diff, max_chunk_lines=200, overlap_lines=10)
        assert len(result) > 1

    def test_large_file_no_boundaries_hard_split(self) -> None:
        """A file with no detectable boundaries or blanks gets a hard split."""
        # Build a diff with no blank lines and no function/class keywords.
        diff = _make_file_diff("dense.py", 600)
        result = chunk_diff(diff, max_chunk_lines=200, overlap_lines=10)
        assert len(result) > 1

    def test_overlap_context_included(self) -> None:
        """Overlap lines appear in adjacent chunks."""
        diff = _make_file_diff("overlap.py", 600)
        result = chunk_diff(diff, max_chunk_lines=200, overlap_lines=20)
        assert len(result) >= 2
        # Get the last lines of chunk 0 and first lines of chunk 1
        last_of_first = result[0].splitlines()[-10:]
        first_of_second = result[1].splitlines()[:30]
        # There should be some shared content
        overlap_found = any(
            line in first_of_second for line in last_of_first if line.strip()
        )
        assert overlap_found, "Expected overlap context between adjacent chunks"

    def test_exact_threshold_no_split(self) -> None:
        """A file with exactly max_chunk_lines does not get split."""
        # 500 body lines + 5 header lines = 505 lines, over 500
        # So use 495 body lines + 5 header lines = 500 exactly
        diff = _make_file_diff("exact.py", 495)
        lines = diff.splitlines(keepends=True)
        # Adjust to exactly 500 lines
        diff_exact = "".join(lines[:500])
        result = chunk_diff(diff_exact, max_chunk_lines=500)
        assert len(result) == 1

    def test_one_over_threshold_splits(self) -> None:
        """A file with max_chunk_lines + 1 lines gets split."""
        diff = _make_file_diff("over.py", 600)
        result = chunk_diff(diff, max_chunk_lines=100, overlap_lines=5)
        assert len(result) > 1


# ---------------------------------------------------------------------------
# chunk_diff — edge cases
# ---------------------------------------------------------------------------


class TestChunkDiffEdgeCases:
    """Edge cases and error conditions for chunk_diff."""

    def test_invalid_max_chunk_lines(self) -> None:
        """max_chunk_lines < 1 should raise ChunkingError."""
        with pytest.raises(ChunkingError, match="max_chunk_lines must be >= 1"):
            chunk_diff("diff --git a/f b/f\n+hello\n", max_chunk_lines=0)

    def test_negative_overlap_lines(self) -> None:
        """Negative overlap_lines should raise ChunkingError."""
        with pytest.raises(ChunkingError, match="overlap_lines must be >= 0"):
            chunk_diff("diff --git a/f b/f\n+hello\n", overlap_lines=-1)

    def test_zero_overlap_lines(self) -> None:
        """Zero overlap is valid and produces no shared context."""
        diff = _make_file_diff("no_overlap.py", 300)
        result = chunk_diff(diff, max_chunk_lines=100, overlap_lines=0)
        assert len(result) > 1

    def test_mixed_small_and_large_files(self) -> None:
        """A diff with both small and large files handles each appropriately."""
        small = _make_file_diff("small.py", 10)
        large = _make_file_diff("large.py", 600)
        diff = small + large
        result = chunk_diff(diff, max_chunk_lines=200, overlap_lines=10)
        # At least 1 chunk for small + multiple for large
        assert len(result) >= 3
        assert "small.py" in result[0]


# ---------------------------------------------------------------------------
# chunk_diff_detailed
# ---------------------------------------------------------------------------


class TestChunkDiffDetailed:
    """Tests for chunk_diff_detailed which returns ChunkResult objects."""

    def test_empty_returns_empty(self) -> None:
        """Empty diff returns empty list."""
        assert chunk_diff_detailed("") == []

    def test_single_file_metadata(self) -> None:
        """Single small file returns one ChunkResult with correct metadata."""
        diff = _make_file_diff("meta.py", 10)
        results = chunk_diff_detailed(diff, max_chunk_lines=500)
        assert len(results) == 1
        assert isinstance(results[0], ChunkResult)
        assert results[0].filename == "meta.py"
        assert results[0].start_offset == 0
        assert "meta.py" in results[0].chunk_text

    def test_multi_file_offsets(self) -> None:
        """Multiple files have increasing start_offset values."""
        diff = _make_file_diff("a.py", 10) + _make_file_diff("b.py", 15)
        results = chunk_diff_detailed(diff, max_chunk_lines=500)
        assert len(results) == 2
        assert results[0].start_offset == 0
        # Second file offset should equal line count of first file
        first_line_count = len(results[0].chunk_text.splitlines(keepends=True))
        assert results[1].start_offset == first_line_count


# ---------------------------------------------------------------------------
# remap_line_numbers
# ---------------------------------------------------------------------------


class TestRemapLineNumbers:
    """Tests for remapping chunk-local line numbers to global coordinates."""

    def test_empty_comments(self) -> None:
        """No comments returns empty list."""
        assert remap_line_numbers([], [0, 100]) == []

    def test_single_chunk_offset_zero(self) -> None:
        """Comments in chunk 0 with offset 0 keep their line numbers."""
        comments = [{"file": "a.py", "line": 10, "chunk_index": 0, "body": "issue"}]
        result = remap_line_numbers(comments, [0])
        assert result[0]["line"] == 10
        assert "chunk_index" not in result[0]

    def test_nonzero_offset(self) -> None:
        """Comments get the correct offset added."""
        comments = [
            {"file": "a.py", "line": 5, "chunk_index": 0, "body": "first"},
            {"file": "a.py", "line": 3, "chunk_index": 1, "body": "second"},
            {"file": "b.py", "line": 7, "chunk_index": 2, "body": "third"},
        ]
        offsets = [0, 100, 250]
        result = remap_line_numbers(comments, offsets)
        assert result[0]["line"] == 5  # 5 + 0
        assert result[1]["line"] == 103  # 3 + 100
        assert result[2]["line"] == 257  # 7 + 250

    def test_no_chunk_index_passed_through(self) -> None:
        """Comments without chunk_index are returned as-is."""
        comments = [{"file": "a.py", "line": 42, "body": "global"}]
        result = remap_line_numbers(comments, [0, 100])
        assert result[0]["line"] == 42

    def test_out_of_range_chunk_index(self) -> None:
        """Out-of-range chunk_index raises ChunkingError."""
        comments = [{"file": "a.py", "line": 1, "chunk_index": 5}]
        with pytest.raises(ChunkingError, match="out of range"):
            remap_line_numbers(comments, [0, 100])

    def test_non_int_chunk_index(self) -> None:
        """Non-integer chunk_index raises ChunkingError."""
        comments = [{"file": "a.py", "line": 1, "chunk_index": "zero"}]
        with pytest.raises(ChunkingError, match="chunk_index must be an int"):
            remap_line_numbers(comments, [0])

    def test_non_int_line(self) -> None:
        """Non-integer line raises ChunkingError."""
        comments = [{"file": "a.py", "line": "bad", "chunk_index": 0}]
        with pytest.raises(ChunkingError, match="line must be an int"):
            remap_line_numbers(comments, [0])

    def test_original_comments_not_mutated(self) -> None:
        """The function should not mutate the input list."""
        comments = [{"file": "a.py", "line": 5, "chunk_index": 0, "body": "test"}]
        original_line = comments[0]["line"]
        remap_line_numbers(comments, [100])
        assert comments[0]["line"] == original_line
        assert "chunk_index" in comments[0]


# ---------------------------------------------------------------------------
# Integration-style: uses mock_large_diff fixture from conftest
# ---------------------------------------------------------------------------


class TestChunkDiffWithFixtures:
    """Tests that use shared fixtures from conftest.py."""

    def test_mock_large_diff_produces_multiple_chunks(
        self, mock_large_diff: str
    ) -> None:
        """The mock_large_diff fixture (600 lines) should be chunked."""
        result = chunk_diff(mock_large_diff, max_chunk_lines=200, overlap_lines=10)
        assert len(result) > 1

    def test_mock_large_diff_default_threshold(self, mock_large_diff: str) -> None:
        """At default 500-line threshold, 600-line diff gets split."""
        result = chunk_diff(mock_large_diff, max_chunk_lines=500, overlap_lines=20)
        assert len(result) >= 2

    def test_mock_pr_diff_no_chunking_needed(self, mock_pr_diff: str) -> None:
        """The small mock_pr_diff should not need chunking (one chunk per file)."""
        result = chunk_diff(mock_pr_diff, max_chunk_lines=500)
        # The fixture has 2 files, each small
        assert len(result) == 2
