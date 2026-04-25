"""
配置管理模組

使用 pydantic-settings 實現配置管理，支持：
- YAML 配置文件加載
- 環境變量覆蓋
- 類型驗證
- 配置分組
"""

import os
import re
from pathlib import Path
from typing import Optional

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings


class OpenAIConfig(BaseSettings):
    """OpenAI API 配置"""

    api_key: str = Field(default="", description="OpenAI API 密鑰")
    model: str = Field(default="gpt-4o-mini", description="使用的模型名稱")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="生成溫度")
    max_tokens: int = Field(default=1000, gt=0, description="最大生成 token 數")

    class Config:
        env_prefix = "OPENAI_"


class RAGConfig(BaseSettings):
    """RAG 檢索配置"""

    chunk_size: int = Field(default=1000, gt=0, description="每個 chunk 的最大字符數")
    chunk_overlap: int = Field(default=200, ge=0, description="chunk 之間的重疊字符數")
    top_k: int = Field(default=5, gt=0, description="檢索返回的 chunk 數量")
    similarity_threshold: float = Field(
        default=0.7, ge=0.0, le=1.0, description="相似度閾值"
    )

    class Config:
        env_prefix = "RAG_"


class VectorStoreConfig(BaseSettings):
    """向量存儲配置"""

    path: str = Field(default="./data/chroma", description="向量數據庫存儲路徑")
    collection_name: str = Field(default="paperlens", description="集合名稱")

    class Config:
        env_prefix = "VECTOR_STORE_"


class EmbeddingConfig(BaseSettings):
    """Embedding 模型配置"""

    model_name: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Embedding 模型名稱"
    )
    device: str = Field(default="cpu", description="運行設備 (cpu/cuda)")

    class Config:
        env_prefix = "EMBEDDING_"


class UIConfig(BaseSettings):
    """UI 界面配置"""

    page_title: str = Field(default="PaperLens - 學術論文分析器", description="頁面標題")
    max_upload_size: int = Field(default=50, gt=0, description="最大上傳文件大小 (MB)")

    class Config:
        env_prefix = "UI_"


class Settings(BaseSettings):
    """主配置類"""

    openai: OpenAIConfig = Field(default_factory=OpenAIConfig)
    rag: RAGConfig = Field(default_factory=RAGConfig)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    ui: UIConfig = Field(default_factory=UIConfig)

    @classmethod
    def load_from_yaml(cls, config_path: Optional[Path] = None) -> "Settings":
        """
        從 YAML 文件加載配置

        Args:
            config_path: 配置文件路徑，默認為 config/config.yaml

        Returns:
            Settings 實例
        """
        if config_path is None:
            # 默認配置文件路徑
            config_path = Path(__file__).parent / "config.yaml"

        if not config_path.exists():
            # 如果配置文件不存在，返回默認配置
            return cls()

        # 讀取 YAML 文件
        with open(config_path, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f)

        if config_data is None:
            return cls()

        # 解析環境變量
        config_data = cls._resolve_env_vars(config_data)

        # 創建配置對象
        openai_config = OpenAIConfig(**config_data.get("openai", {}))
        rag_config = RAGConfig(**config_data.get("rag", {}))
        vector_store_config = VectorStoreConfig(**config_data.get("vector_store", {}))
        embedding_config = EmbeddingConfig(**config_data.get("embedding", {}))
        ui_config = UIConfig(**config_data.get("ui", {}))

        return cls(
            openai=openai_config,
            rag=rag_config,
            vector_store=vector_store_config,
            embedding=embedding_config,
            ui=ui_config,
        )

    @staticmethod
    def _resolve_env_vars(config_data: dict) -> dict:
        """
        解析配置中的環境變量引用

        支持格式：${VAR_NAME} 或 ${VAR_NAME:default_value}

        Args:
            config_data: 配置字典

        Returns:
            解析後的配置字典
        """
        def resolve_value(value):
            if isinstance(value, str):
                # 匹配 ${VAR_NAME} 或 ${VAR_NAME:default}
                pattern = r'\$\{([^}:]+)(?::([^}]*))?\}'

                def replacer(match):
                    var_name = match.group(1)
                    default_value = match.group(2) if match.group(2) is not None else ""
                    return os.environ.get(var_name, default_value)

                return re.sub(pattern, replacer, value)
            elif isinstance(value, dict):
                return {k: resolve_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [resolve_value(item) for item in value]
            else:
                return value

        return resolve_value(config_data)

    class Config:
        env_nested_delimiter = "__"


# 全局配置單例
settings = Settings.load_from_yaml()


# 便捷訪問函數
def get_settings() -> Settings:
    """獲取全局配置實例"""
    return settings


def reload_settings(config_path: Optional[Path] = None) -> Settings:
    """
    重新加載配置

    Args:
        config_path: 配置文件路徑

    Returns:
        新的 Settings 實例
    """
    global settings
    settings = Settings.load_from_yaml(config_path)
    return settings


if __name__ == "__main__":
    # 測試配置加載
    print("=== 配置加載測試 ===")
    print(f"OpenAI Model: {settings.openai.model}")
    print(f"OpenAI Temperature: {settings.openai.temperature}")
    print(f"RAG Chunk Size: {settings.rag.chunk_size}")
    print(f"RAG Top K: {settings.rag.top_k}")
    print(f"Vector Store Path: {settings.vector_store.path}")
    print(f"Embedding Model: {settings.embedding.model_name}")
    print(f"UI Page Title: {settings.ui.page_title}")
    print(f"UI Max Upload Size: {settings.ui.max_upload_size} MB")

    # 測試環境變量
    if settings.openai.api_key:
        print(f"OpenAI API Key: {'*' * 10}{settings.openai.api_key[-4:]}")
    else:
        print("OpenAI API Key: Not set")
