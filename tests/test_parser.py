"""Tests for parser modules (no PDF file needed)."""
import pytest
from models.schemas import Section
from parser.chunker import chunk_sections, _split_text


def test_split_text_short():
    assert _split_text("Hello world.", 1000) == ["Hello world."]


def test_split_text_long():
    text = ("This is a sentence. " * 60).strip()
    parts = _split_text(text, 200)
    assert all(len(p) <= 200 for p in parts)
    assert " ".join(parts).replace("  ", " ") != ""


def test_chunk_sections_ids_unique():
    sections = [
        Section(name="Introduction", page=1, text="This is intro. " * 10),
        Section(name="Method", page=2, text="This is method. " * 10),
    ]
    chunks = chunk_sections(sections)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))


def test_chunk_metadata():
    sections = [Section(name="Results", page=3, text="Short text.")]
    chunks = chunk_sections(sections)
    assert chunks[0].section == "Results"
    assert chunks[0].page == 3
