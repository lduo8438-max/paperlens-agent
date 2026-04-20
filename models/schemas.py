from pydantic import BaseModel
from typing import Optional


class Section(BaseModel):
    """A detected section in the paper."""
    name: str
    page: int
    text: str


class Chunk(BaseModel):
    """A text chunk with structural metadata."""
    chunk_id: str
    text: str
    section: str
    page: int


class Source(BaseModel):
    """Source attribution for an answer."""
    section: str
    page: int


class QueryResult(BaseModel):
    """Result returned by the RAG pipeline."""
    answer: str
    sources: list[Source]
    route: str  # summary | section | qa
