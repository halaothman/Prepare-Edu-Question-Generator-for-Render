from __future__ import annotations

import re

from .config import CHUNK_OVERLAP, CHUNK_SIZE, LOGICAL_SEGMENT_MAX_CHARS, MAX_LOGICAL_SEGMENTS

# Topic / chapter heading lines (Arabic + English, common slide/PDF patterns)
_HEADING_LINE = re.compile(
    r"^(?:"
    r"(?:Chapter|CHAPTER|Section|SECTION|Part|PART|Appendix|APPENDIX|Unit|UNIT|Lesson|LESSON)"
    r"\s+[\dIVXLCivxlc\.]+"
    r"|(?:الفصل|الوحدة|الباب|المحاضرة|الدرس|الموضوع)\s*[\d٠-٩0-9\.:\-]+"
    r"|\d{1,2}(?:\.\d+){0,3}[\.\):\-]\s+\S"
    r"|[IVXLC]{1,6}\.\s+\S"
    r")\s*.+$",
    re.IGNORECASE,
)


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    chunks: list[str] = []
    step = max(size - overlap, 1)
    for i in range(0, len(text), step):
        piece = text[i : i + size].strip()
        if piece:
            chunks.append(piece)
    return chunks


def build_segments(
    text: str,
    *,
    size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
    max_segment_chars: int = LOGICAL_SEGMENT_MAX_CHARS,
) -> list[str]:
    """Merge small chunks into segments capped at max_segment_chars (legacy fixed-size path)."""
    chunks = chunk_text(text, size=size, overlap=overlap)
    if not chunks:
        return []

    segments: list[str] = []
    current: list[str] = []
    current_len = 0

    for chunk in chunks:
        extra = 2 if current else 0
        if current and current_len + extra + len(chunk) > max_segment_chars:
            segments.append("\n\n".join(current))
            current = [chunk]
            current_len = len(chunk)
            continue
        current.append(chunk)
        current_len += extra + len(chunk)

    if current:
        segments.append("\n\n".join(current))

    return segments


def _split_by_headings(text: str) -> list[str]:
    lines = text.splitlines()
    if not lines:
        return []

    blocks: list[str] = []
    current: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped and _HEADING_LINE.match(stripped) and current:
            block = "\n".join(current).strip()
            if block:
                blocks.append(block)
            current = [line]
        else:
            current.append(line)

    if current:
        block = "\n".join(current).strip()
        if block:
            blocks.append(block)
    return blocks


def _split_into_equal_parts(text: str, parts: int) -> list[str]:
    text = text.strip()
    if not text:
        return []
    parts = max(1, parts)
    if parts == 1:
        return [text]

    length = len(text)
    step = length // parts
    segments: list[str] = []
    for index in range(parts):
        start = index * step
        end = length if index == parts - 1 else (index + 1) * step
        if index > 0:
            # Prefer breaking at paragraph boundary near the cut.
            window = text[start : min(length, start + 400)]
            break_at = window.find("\n\n")
            if break_at > 40:
                start = start + break_at + 2
        piece = text[start:end].strip()
        if piece:
            segments.append(piece)
    return segments


def _merge_smallest_adjacent(blocks: list[str]) -> list[str]:
    if len(blocks) <= 1:
        return blocks
    best_index = 0
    best_size = len(blocks[0]) + len(blocks[1])
    for i in range(len(blocks) - 1):
        size = len(blocks[i]) + len(blocks[i + 1])
        if size < best_size:
            best_size = size
            best_index = i
    merged = blocks[:best_index] + [
        blocks[best_index].rstrip() + "\n\n" + blocks[best_index + 1].lstrip()
    ] + blocks[best_index + 2 :]
    return merged


def _cap_block_size(block: str, max_chars: int) -> list[str]:
    if len(block) <= max_chars:
        return [block]
    return build_segments(block, max_segment_chars=max_chars)


def build_logical_segments(
    text: str,
    *,
    max_segments: int = MAX_LOGICAL_SEGMENTS,
    max_segment_chars: int = LOGICAL_SEGMENT_MAX_CHARS,
    min_segment_chars: int = 800,
) -> list[str]:
    """Split by headings when possible; otherwise equal parts; cap segment count."""
    cleaned = text.strip()
    if not cleaned:
        return []

    blocks = _split_by_headings(cleaned)
    if len(blocks) < 2:
        target = min(
            max_segments,
            max(2, len(cleaned) // max(min_segment_chars, 1)),
        )
        blocks = _split_into_equal_parts(cleaned, target)

    while len(blocks) > max_segments:
        blocks = _merge_smallest_adjacent(blocks)

    segments: list[str] = []
    for block in blocks:
        if len(block) <= max_segment_chars:
            if len(block) >= min_segment_chars or not segments:
                segments.append(block)
            elif segments:
                segments[-1] = segments[-1].rstrip() + "\n\n" + block.lstrip()
            else:
                segments.append(block)
        else:
            segments.extend(_cap_block_size(block, max_segment_chars))

    while len(segments) > max_segments:
        segments = _merge_smallest_adjacent(segments)

    return [segment.strip() for segment in segments if segment.strip()]
