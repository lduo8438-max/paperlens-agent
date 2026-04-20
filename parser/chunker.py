"""
Structure-aware chunker.

Splits each Section into chunks that respect sentence boundaries.
Each chunk carries: chunk_id, text, section, page.
"""

from models.schemas import Chunk, Section

MAX_CHARS = 1000  # max characters per chunk


def _split_text(text: str, max_chars: int) -> list[str]:
    """Split text into parts of at most max_chars, breaking on sentence ends."""
    if len(text) <= max_chars:
        return [text]

    parts = []
    while len(text) > max_chars:
        split_at = text.rfind(". ", 0, max_chars)
        if split_at == -1:
            split_at = max_chars
        else:
            split_at += 1  # include the period
        parts.append(text[:split_at].strip())
        text = text[split_at:].strip()

    if text:
        parts.append(text)
    return parts


def chunk_sections(sections: list[Section]) -> list[Chunk]:
    """Convert sections into chunks with structural metadata."""
    chunks: list[Chunk] = []
    for sec_idx, section in enumerate(sections):
        if not section.text.strip():
            continue
        parts = _split_text(section.text, MAX_CHARS)
        for i, part in enumerate(parts):
            chunks.append(Chunk(
                chunk_id=f"{sec_idx}_{section.name}_{i}",
                text=part,
                section=section.name,
                page=section.page,
            ))
    return chunks
