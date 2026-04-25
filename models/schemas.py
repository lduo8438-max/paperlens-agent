from datetime import datetime
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
    doc_id: str = ""         # 關聯到文檔
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


class Document(BaseModel):
    """文檔元數據"""
    doc_id: str              # 唯一標識符（UUID）
    filename: str            # 原始文件名
    upload_time: datetime    # 上傳時間
    num_chunks: int          # chunk 數量
