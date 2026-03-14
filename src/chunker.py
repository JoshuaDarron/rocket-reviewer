"""Large PR diff chunking and line number remapping.

Splits diffs at file boundaries, then at function/class boundaries for
oversized files. Includes overlap context between segments and remaps
line numbers back to original diff coordinates after merge.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from src.errors import ChunkingError

logger = logging.getLogger(__name__)

# Regex to detect the start of a new file in a unified diff.
_FILE_HEADER_RE = re.compile(r"^diff --git ", re.MULTILINE)

# Regex to detect function/class boundaries (Python, JS/TS, Java, Go, Rust, etc.)
_BOUNDARY_RE = re.compile(
    r"^[+-]?\s*"
    r"(?:def |class |function |async function "
    r"|const \w+ = |export |impl |fn |func "
    r"|public |private |protected )",
    re.MULTILINE,
)


@dataclass
class ChunkResult:
    """Metadata and content for a single diff chunk.

    Attributes:
        filename: The file this chunk belongs to.
        start_offset: The line offset of this chunk within the original diff.
        chunk_text: The raw diff text for this chunk.
    """

    filename: str
    start_offset: int
    chunk_text: str


@dataclass
class _FileDiff:
    """Internal representation of a single file's diff.

    Attributes:
        filename: Extracted filename from the diff header.
        text: Full diff text for this file (including headers).
        lines: The text split into individual lines.
    """

    filename: str
    text: str
    lines: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.lines = self.text.splitlines(keepends=True)


def _split_into_file_diffs(diff: str) -> list[tuple[str, str]]:
    """Split a unified diff into per-file sections.

    Args:
        diff: The full unified diff string.

    Returns:
        List of ``(filename, file_diff)`` pairs. The filename is extracted
        from the ``diff --git a/... b/...`` header. If parsing fails the
        filename defaults to ``"unknown"``.

    Raises:
        ChunkingError: If the diff cannot be split.
    """
    if not diff or not diff.strip():
        return []

    try:
        # Find all "diff --git" header positions.
        positions = [m.start() for m in _FILE_HEADER_RE.finditer(diff)]

        if not positions:
            # No file headers found — treat entire diff as a single file.
            return [("unknown", diff)]

        results: list[tuple[str, str]] = []
        for idx, pos in enumerate(positions):
            end = positions[idx + 1] if idx + 1 < len(positions) else len(diff)
            section = diff[pos:end]

            # Extract filename from "diff --git a/<path> b/<path>".
            header_line = section.split("\n", 1)[0]
            match = re.search(r" b/(.+)$", header_line)
            filename = match.group(1) if match else "unknown"
            results.append((filename, section))

        return results
    except Exception as exc:
        raise ChunkingError(f"Failed to split diff into files: {exc}") from exc


def _find_split_points(
    lines: list[str],
    max_chunk_lines: int,
    overlap_lines: int,
) -> list[int]:
    """Find indices at which to split a large file diff.

    Tries function/class boundaries first, then blank lines, and finally
    falls back to hard splits at ``max_chunk_lines``.

    Args:
        lines: The diff lines for a single file.
        max_chunk_lines: Maximum lines per chunk.
        overlap_lines: Overlap context between segments.

    Returns:
        Sorted list of line indices where splits should occur.
    """
    if len(lines) <= max_chunk_lines:
        return []

    split_points: list[int] = []
    current_start = 0

    while current_start + max_chunk_lines < len(lines):
        window_end = current_start + max_chunk_lines
        search_start = max(current_start + max_chunk_lines // 2, current_start + 1)

        # 1. Try function/class boundary (search backwards from window_end).
        best: int | None = None
        for i in range(window_end - 1, search_start - 1, -1):
            if _BOUNDARY_RE.match(lines[i]):
                best = i
                break

        # 2. Fall back to blank line boundary.
        if best is None:
            for i in range(window_end - 1, search_start - 1, -1):
                stripped = lines[i].strip()
                if stripped == "" or stripped in ("+", "-"):
                    best = i + 1  # split *after* the blank line
                    break

        # 3. Hard split at max_chunk_lines.
        if best is None:
            best = window_end

        split_points.append(best)
        # Next segment starts overlap_lines before the split point.
        current_start = max(best - overlap_lines, current_start + 1)

    return sorted(set(split_points))


def chunk_diff(
    diff: str,
    max_chunk_lines: int = 500,
    overlap_lines: int = 20,
) -> list[str]:
    """Split a diff into reviewable chunks.

    The algorithm works in two passes:

    1. Split the diff at file boundaries (each ``diff --git`` header starts
       a new file).
    2. If a single file's diff exceeds *max_chunk_lines*, sub-split at
       function/class boundaries where detectable, otherwise at blank line
       boundaries. Include *overlap_lines* lines of overlap between segments.

    Args:
        diff: The full unified diff string.
        max_chunk_lines: Maximum lines per chunk.
        overlap_lines: Lines of overlap context between segments.

    Returns:
        List of diff chunk strings. An empty diff returns an empty list.

    Raises:
        ChunkingError: If the diff cannot be chunked.
    """
    if not diff or not diff.strip():
        return []

    if max_chunk_lines < 1:
        raise ChunkingError("max_chunk_lines must be >= 1")
    if overlap_lines < 0:
        raise ChunkingError("overlap_lines must be >= 0")

    try:
        file_diffs = _split_into_file_diffs(diff)
        chunks: list[str] = []

        for _filename, file_text in file_diffs:
            file_lines = file_text.splitlines(keepends=True)

            if len(file_lines) <= max_chunk_lines:
                chunks.append(file_text)
                continue

            # Need to sub-split this file.
            split_points = _find_split_points(
                file_lines, max_chunk_lines, overlap_lines
            )

            if not split_points:
                chunks.append(file_text)
                continue

            # Build sub-chunks with overlap.
            boundaries = [0, *split_points, len(file_lines)]
            for i in range(len(boundaries) - 1):
                start = boundaries[i]
                end = boundaries[i + 1]

                # Add overlap from the next chunk's beginning (already handled
                # by the split point calculation which accounts for overlap).
                # For the first sub-chunk, also include overlap into the next.
                if i > 0:
                    start = max(start - overlap_lines, 0)

                chunk_text = "".join(file_lines[start:end])
                if chunk_text.strip():
                    chunks.append(chunk_text)

        return chunks

    except ChunkingError:
        raise
    except Exception as exc:
        raise ChunkingError(f"Failed to chunk diff: {exc}") from exc


def chunk_diff_detailed(
    diff: str,
    max_chunk_lines: int = 500,
    overlap_lines: int = 20,
) -> list[ChunkResult]:
    """Split a diff into chunks and return detailed metadata.

    Like :func:`chunk_diff` but returns :class:`ChunkResult` objects that
    include the filename and starting offset for each chunk.

    Args:
        diff: The full unified diff string.
        max_chunk_lines: Maximum lines per chunk.
        overlap_lines: Lines of overlap context between segments.

    Returns:
        List of :class:`ChunkResult` instances.

    Raises:
        ChunkingError: If the diff cannot be chunked.
    """
    if not diff or not diff.strip():
        return []

    try:
        file_diffs = _split_into_file_diffs(diff)
        results: list[ChunkResult] = []
        global_offset = 0

        for filename, file_text in file_diffs:
            file_lines = file_text.splitlines(keepends=True)

            if len(file_lines) <= max_chunk_lines:
                results.append(
                    ChunkResult(
                        filename=filename,
                        start_offset=global_offset,
                        chunk_text=file_text,
                    )
                )
                global_offset += len(file_lines)
                continue

            split_points = _find_split_points(
                file_lines, max_chunk_lines, overlap_lines
            )

            if not split_points:
                results.append(
                    ChunkResult(
                        filename=filename,
                        start_offset=global_offset,
                        chunk_text=file_text,
                    )
                )
                global_offset += len(file_lines)
                continue

            boundaries = [0, *split_points, len(file_lines)]
            for i in range(len(boundaries) - 1):
                start = boundaries[i]
                end = boundaries[i + 1]

                effective_start = max(start - overlap_lines, 0) if i > 0 else start

                chunk_text = "".join(file_lines[effective_start:end])
                if chunk_text.strip():
                    results.append(
                        ChunkResult(
                            filename=filename,
                            start_offset=global_offset + effective_start,
                            chunk_text=chunk_text,
                        )
                    )

            global_offset += len(file_lines)

        return results

    except ChunkingError:
        raise
    except Exception as exc:
        raise ChunkingError(f"Failed to chunk diff: {exc}") from exc


def remap_line_numbers(
    comments: list[dict[str, object]],
    chunk_offsets: list[int],
) -> list[dict[str, object]]:
    """Remap comment line numbers from chunk-local to original diff coordinates.

    Each comment must have a ``chunk_index`` key indicating which chunk it
    came from, and a ``line`` key with the chunk-local line number. The
    function adds the corresponding offset from *chunk_offsets* to produce
    the line number in the original (un-chunked) diff.

    Args:
        comments: Comments with chunk-local line numbers. Each dict must
            contain ``"line"`` (int) and ``"chunk_index"`` (int) keys.
        chunk_offsets: Starting line offset for each chunk.

    Returns:
        A new list of comment dicts with corrected ``line`` values. The
        ``chunk_index`` key is removed from each comment.

    Raises:
        ChunkingError: If a chunk_index is out of range or line is not an int.
    """
    if not comments:
        return []

    try:
        remapped: list[dict[str, object]] = []
        for comment in comments:
            new_comment = dict(comment)

            chunk_index = new_comment.get("chunk_index")
            if chunk_index is None:
                # No chunk_index means the comment is already in global coords.
                remapped.append(new_comment)
                continue

            if not isinstance(chunk_index, int):
                raise ChunkingError(
                    f"chunk_index must be an int, got {type(chunk_index).__name__}"
                )

            if chunk_index < 0 or chunk_index >= len(chunk_offsets):
                raise ChunkingError(
                    f"chunk_index {chunk_index} out of range "
                    f"(0..{len(chunk_offsets) - 1})"
                )

            line = new_comment.get("line")
            if not isinstance(line, int):
                raise ChunkingError(f"line must be an int, got {type(line).__name__}")

            new_comment["line"] = line + chunk_offsets[chunk_index]
            new_comment.pop("chunk_index", None)
            remapped.append(new_comment)

        return remapped

    except ChunkingError:
        raise
    except Exception as exc:
        raise ChunkingError(f"Failed to remap line numbers: {exc}") from exc
