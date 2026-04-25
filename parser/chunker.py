"""
結構感知的分塊器，使用 LangChain 的 RecursiveCharacterTextSplitter。

將每個 Section 分割為 chunks，保留結構元數據。
"""

from langchain.text_splitter import RecursiveCharacterTextSplitter

from config import settings
from models.schemas import Chunk, Section


def chunk_sections(sections: list[Section]) -> list[Chunk]:
    """
    將論文章節轉換為文本塊，保留結構元數據。

    使用 LangChain 的 RecursiveCharacterTextSplitter 進行智能分割，
    支持中英文混合，並保留 chunk overlap。

    Args:
        sections: 論文章節列表

    Returns:
        包含結構元數據的文本塊列表
    """
    # 分隔符優先級：段落 > 句子 > 詞
    separators = [
        "\n\n",           # 段落
        "\n",             # 換行
        "。", "！", "？",  # 中文句子結束
        ". ", "! ", "? ", # 英文句子結束
        "；", "; ",       # 分號
        "，", ", ",       # 逗號
        " ",              # 空格
        ""                # 字符
    ]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.rag.chunk_size,
        chunk_overlap=settings.rag.chunk_overlap,
        length_function=len,
        separators=separators
    )

    chunks: list[Chunk] = []
    for sec_idx, section in enumerate(sections):
        if not section.text.strip():
            continue

        # 使用智能分割
        chunk_texts = splitter.split_text(section.text)

        for i, text in enumerate(chunk_texts):
            chunks.append(Chunk(
                chunk_id=f"{sec_idx}_{section.name}_{i}",
                text=text,
                section=section.name,
                page=section.page
            ))

    return chunks


# 保留舊的類接口以兼容現有代碼
class Chunker:
    """分塊器類（兼容接口）"""

    def chunk(self, sections: list[Section]) -> list[Chunk]:
        """將章節分塊"""
        return chunk_sections(sections)
