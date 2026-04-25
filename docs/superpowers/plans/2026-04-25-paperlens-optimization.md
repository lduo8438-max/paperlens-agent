# PaperLens-Agent 優化實施計劃

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 通過 5 項關鍵改進提升 PaperLens-Agent 的穩定性、可維護性和用戶體驗

**Architecture:** 漸進式優化，保持現有架構。按依賴順序實施：配置管理 → 錯誤處理 → 分塊優化 → Prompt 優化 → 多文檔支持 → 中文化

**Tech Stack:** Python, Streamlit, LangChain, ChromaDB, OpenAI, Pydantic Settings, PyYAML

---

## 文件結構映射

### 新增文件
- `config/__init__.py` - 配置模塊入口，導出 settings 對象
- `config/settings.py` - 配置加載邏輯，使用 pydantic-settings
- `config/config.yaml` - YAML 配置文件

### 修改文件
- `requirements.txt` - 添加新依賴
- `parser/chunker.py` - 使用 LangChain splitter
- `rag/pipeline.py` - 錯誤處理、優化 prompt、支持 doc_ids
- `rag/vector_store.py` - 添加文檔管理方法
- `ui/app.py` - 重寫文檔管理界面
- `models/schemas.py` - 添加 Document 模型，擴展 Chunk
- `README.md` - 更新為簡體中文
- `progress/progress.md` - 更新進度

### 不修改文件
- `parser/pdf_parser.py` - PDF 解析邏輯保持不變
- `agent/router.py` - 查詢路由保持不變
- `rag/embedder.py` - Embedder 封裝保持不變

---

## Task 1: 配置管理 - 更新依賴

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: 添加新依賴到 requirements.txt**

```bash
cd /tmp/paperlens-agent
```

在 requirements.txt 末尾添加：
```
pyyaml>=6.0
pydantic-settings>=2.0.0
langchain>=0.1.0
langchain-text-splitters>=0.0.1
```

- [ ] **Step 2: 驗證依賴文件格式**

Run: `cat requirements.txt`
Expected: 文件包含所有依賴，格式正確

- [ ] **Step 3: Commit**

```bash
git add requirements.txt
git commit -m "chore: add dependencies for config management and chunking optimization"
```

---

## Task 2: 配置管理 - 創建配置文件

**Files:**
- Create: `config/config.yaml`

- [ ] **Step 1: 創建 config 目錄**

```bash
mkdir -p config
```

- [ ] **Step 2: 創建 config.yaml**

```yaml
# PaperLens-Agent 配置文件

# OpenAI 配置
openai:
  api_key: ${OPENAI_API_KEY}  # 從環境變量讀取
  model: gpt-4o-mini
  temperature: 0.7
  max_tokens: 1000

# RAG 配置
rag:
  chunk_size: 1000           # 每個 chunk 的最大字符數
  chunk_overlap: 200         # chunk 之間的重疊字符數
  top_k: 5                   # 檢索返回的 chunk 數量
  similarity_threshold: 0.7  # 相似度閾值（未來使用）

# 向量存儲配置
vector_store:
  path: ./data/chroma
  collection_name: paperlens

# Embedding 配置
embedding:
  model_name: sentence-transformers/all-MiniLM-L6-v2
  device: cpu

# UI 配置
ui:
  page_title: PaperLens - 學術論文分析器
  max_upload_size: 50  # MB
```

- [ ] **Step 3: 驗證 YAML 格式**

Run: `python -c "import yaml; yaml.safe_load(open('config/config.yaml'))"`
Expected: 無錯誤輸出

- [ ] **Step 4: Commit**

```bash
git add config/config.yaml
git commit -m "feat: add YAML configuration file"
```

---

## Task 3: 配置管理 - 實現 settings.py

**Files:**
- Create: `config/settings.py`

- [ ] **Step 1: 創建 settings.py**

```python
"""配置管理模塊，使用 pydantic-settings 加載配置"""
import os
from pathlib import Path
from typing import Optional

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class OpenAIConfig(BaseSettings):
    """OpenAI 配置"""
    api_key: str = Field(default="")
    model: str = Field(default="gpt-4o-mini")
    temperature: float = Field(default=0.7)
    max_tokens: int = Field(default=1000)


class RAGConfig(BaseSettings):
    """RAG 配置"""
    chunk_size: int = Field(default=1000)
    chunk_overlap: int = Field(default=200)
    top_k: int = Field(default=5)
    similarity_threshold: float = Field(default=0.7)


class VectorStoreConfig(BaseSettings):
    """向量存儲配置"""
    path: str = Field(default="./data/chroma")
    collection_name: str = Field(default="paperlens")


class EmbeddingConfig(BaseSettings):
    """Embedding 配置"""
    model_name: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    device: str = Field(default="cpu")


class UIConfig(BaseSettings):
    """UI 配置"""
    page_title: str = Field(default="PaperLens - 學術論文分析器")
    max_upload_size: int = Field(default=50)


class Settings(BaseSettings):
    """全局配置"""
    openai: OpenAIConfig = Field(default_factory=OpenAIConfig)
    rag: RAGConfig = Field(default_factory=RAGConfig)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    ui: UIConfig = Field(default_factory=UIConfig)

    model_config = SettingsConfigDict(
        env_nested_delimiter="__",
        case_sensitive=False
    )

    @classmethod
    def load_from_yaml(cls, yaml_path: str = "config/config.yaml") -> "Settings":
        """從 YAML 文件加載配置"""
        config_file = Path(yaml_path)
        if not config_file.exists():
            # 如果配置文件不存在，使用默認值
            return cls()
        
        with open(config_file, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f)
        
        # 處理環境變量替換
        config_data = cls._resolve_env_vars(config_data)
        
        return cls(
            openai=OpenAIConfig(**config_data.get("openai", {})),
            rag=RAGConfig(**config_data.get("rag", {})),
            vector_store=VectorStoreConfig(**config_data.get("vector_store", {})),
            embedding=EmbeddingConfig(**config_data.get("embedding", {})),
            ui=UIConfig(**config_data.get("ui", {}))
        )
    
    @staticmethod
    def _resolve_env_vars(data):
        """遞歸解析環境變量"""
        if isinstance(data, dict):
            return {k: Settings._resolve_env_vars(v) for k, v in data.items()}
        elif isinstance(data, str) and data.startswith("${") and data.endswith("}"):
            env_var = data[2:-1]
            return os.getenv(env_var, "")
        return data


# 全局單例
settings = Settings.load_from_yaml()
```

- [ ] **Step 2: 測試配置加載**

Run: `python -c "from config.settings import settings; print(settings.openai.model)"`
Expected: 輸出 "gpt-4o-mini"

- [ ] **Step 3: 測試環境變量覆蓋**

Run: `export OPENAI_API_KEY=test-key && python -c "from config.settings import settings; print(settings.openai.api_key)"`
Expected: 輸出 "test-key"

- [ ] **Step 4: Commit**

```bash
git add config/settings.py
git commit -m "feat: implement configuration management with pydantic-settings"
```

---

## Task 4: 配置管理 - 創建模塊入口

**Files:**
- Create: `config/__init__.py`

- [ ] **Step 1: 創建 __init__.py**

```python
"""配置管理模塊"""
from config.settings import settings

__all__ = ["settings"]
```

- [ ] **Step 2: 測試導入**

Run: `python -c "from config import settings; print(settings.rag.chunk_size)"`
Expected: 輸出 "1000"

- [ ] **Step 3: Commit**

```bash
git add config/__init__.py
git commit -m "feat: add config module entry point"
```

---

## Task 5: 錯誤處理 - 更新 rag/pipeline.py

**Files:**
- Modify: `rag/pipeline.py`

- [ ] **Step 1: 添加錯誤處理和配置導入**

在文件開頭添加導入：
```python
import time
from config import settings
```

修改 `__init__` 方法：
```python
def __init__(self):
    self.embedder = Embedder()
    self.store = VectorStore()
    try:
        self.client = OpenAI(api_key=settings.openai.api_key)
    except Exception as e:
        raise ValueError(f"初始化 OpenAI 客戶端失敗：{str(e)}")
```

- [ ] **Step 2: 添加重試機制的輔助函數**

在 Pipeline 類之前添加：
```python
def retry_with_backoff(func, max_retries=3):
    """帶指數退避的重試機制"""
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            wait_time = 2 ** attempt
            time.sleep(wait_time)
    return None
```

- [ ] **Step 3: 修改 query 方法添加錯誤處理**

```python
def query(self, question: str, doc_ids: list[str] = None) -> QueryResult:
    """查詢，支持指定文檔範圍"""
    route_type = route(question)
    embedding = self.embedder.embed_query(question)
    
    # 檢索
    hits = self.store.query(
        embedding, 
        doc_ids=doc_ids, 
        n_results=settings.rag.top_k
    )
    
    # 檢查檢索結果
    if not hits:
        return QueryResult(
            answer="抱歉，我在已索引的論文中沒有找到與您問題相關的內容。請嘗試換個問法或上傳相關論文。",
            sources=[],
            route=route_type
        )
    
    # 構建 context
    context_parts = []
    for i, hit in enumerate(hits, 1):
        context_parts.append(
            f"=== 片段 {i} ===\n"
            f"來源：{hit['metadata']['section']}（第 {hit['metadata']['page']} 頁）\n"
            f"內容：{hit['document']}"
        )
    context = "\n\n".join(context_parts)
    
    # 構建 prompt
    system_prompt = """你是一個專業的學術論文分析助手。你的任務是基於提供的論文片段回答用戶問題。

回答要求：
1. 準確性：僅基於提供的內容回答，不要編造信息
2. 引用：在回答中使用 [章節名, 第X頁] 格式標註信息來源
3. 完整性：如果信息不足以回答問題，明確說明
4. 語言：使用簡體中文，保持學術性和專業性
"""
    
    user_prompt = f"""以下是從論文中檢索到的相關片段：

{context}

問題：{question}

請基於上述片段回答問題。記得標註信息來源。"""
    
    # 調用 API with 錯誤處理
    try:
        def api_call():
            return self.client.chat.completions.create(
                model=settings.openai.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=settings.openai.temperature,
                max_tokens=settings.openai.max_tokens
            )
        
        response = retry_with_backoff(api_call)
        answer = response.choices[0].message.content
        
    except Exception as e:
        error_msg = str(e)
        if "authentication" in error_msg.lower() or "api_key" in error_msg.lower():
            raise ValueError("OpenAI API 密鑰無效，請檢查配置")
        elif "rate_limit" in error_msg.lower():
            raise ValueError("API 請求頻率超限，請稍後重試")
        else:
            raise ValueError(f"API 請求失敗：{error_msg}")
    
    # 確保包含引用
    if "[" not in answer and "（第" not in answer:
        answer += "\n\n**參考來源：**\n"
        for hit in hits:
            answer += f"- {hit['metadata']['section']}（第 {hit['metadata']['page']} 頁）\n"
    
    sources = [
        Source(section=h["metadata"]["section"], page=h["metadata"]["page"])
        for h in hits
    ]
    return QueryResult(answer=answer, sources=sources, route=route_type)
```

- [ ] **Step 4: 驗證修改**

Run: `python -c "from rag.pipeline import Pipeline; print('Pipeline 導入成功')"`
Expected: 輸出 "Pipeline 導入成功"

- [ ] **Step 5: Commit**

```bash
git add rag/pipeline.py
git commit -m "feat: add error handling and prompt optimization to pipeline"
```

---

## Task 6: 錯誤處理 - 更新 parser/pdf_parser.py

**Files:**
- Modify: `parser/pdf_parser.py`

- [ ] **Step 1: 在 parse_pdf 函數添加錯誤處理**

修改 `parse_pdf` 函數開頭：
```python
def parse_pdf(pdf_path: str) -> list[Section]:
    """
    解析 PDF 並返回章節列表。
    
    Args:
        pdf_path: PDF 文件路徑
        
    Returns:
        章節列表
        
    Raises:
        ValueError: PDF 文件損壞、加密或格式不正確
    """
    try:
        doc = fitz.open(pdf_path)
    except fitz.FileDataError:
        raise ValueError("PDF 文件損壞或格式不正確")
    except fitz.PasswordError:
        raise ValueError("PDF 已加密，請先解密後再上傳")
    except Exception as e:
        raise ValueError(f"無法打開 PDF 文件：{str(e)}")
    
    # 原有邏輯繼續...
```

- [ ] **Step 2: 驗證修改**

Run: `python -c "from parser.pdf_parser import parse_pdf; print('PDF parser 導入成功')"`
Expected: 輸出 "PDF parser 導入成功"

- [ ] **Step 3: Commit**

```bash
git add parser/pdf_parser.py
git commit -m "feat: add error handling to PDF parser"
```

---

## Task 7: 分塊優化 - 重寫 parser/chunker.py

**Files:**
- Modify: `parser/chunker.py`

- [ ] **Step 1: 重寫 chunker.py**

```python
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
```

- [ ] **Step 2: 驗證導入**

Run: `python -c "from parser.chunker import Chunker; print('Chunker 導入成功')"`
Expected: 輸出 "Chunker 導入成功"

- [ ] **Step 3: Commit**

```bash
git add parser/chunker.py
git commit -m "feat: optimize chunking with LangChain RecursiveCharacterTextSplitter"
```

---

## Task 8: 多文檔支持 - 擴展數據模型

**Files:**
- Modify: `models/schemas.py`

- [ ] **Step 1: 添加 Document 模型**

在文件末尾添加：
```python
from datetime import datetime


class Document(BaseModel):
    """文檔元數據"""
    doc_id: str              # 唯一標識符（UUID）
    filename: str            # 原始文件名
    upload_time: datetime    # 上傳時間
    num_chunks: int          # chunk 數量
```

- [ ] **Step 2: 擴展 Chunk 模型**

修改 Chunk 模型，添加 doc_id 字段：
```python
class Chunk(BaseModel):
    chunk_id: str
    doc_id: str = ""         # 新增：關聯到文檔
    text: str
    section: str
    page: int
```

- [ ] **Step 3: 驗證模型**

Run: `python -c "from models.schemas import Document, Chunk; print('Models 導入成功')"`
Expected: 輸出 "Models 導入成功"

- [ ] **Step 4: Commit**

```bash
git add models/schemas.py
git commit -m "feat: add Document model and extend Chunk with doc_id"
```

---

## Task 9: 多文檔支持 - 更新 rag/vector_store.py

**Files:**
- Modify: `rag/vector_store.py`

- [ ] **Step 1: 修改 add 方法支持 doc_id**

```python
def add(self, records: list[dict], doc_id: str = None) -> None:
    """
    添加記錄到向量存儲。
    
    Args:
        records: 記錄列表
        doc_id: 可選的文檔 ID，會添加到每個 record 的 metadata
    """
    if doc_id:
        for record in records:
            if 'metadata' not in record:
                record['metadata'] = {}
            record['metadata']['doc_id'] = doc_id
    
    self.col.add(
        ids=[r["id"] for r in records],
        embeddings=[r["embedding"] for r in records],
        documents=[r["document"] for r in records],
        metadatas=[r["metadata"] for r in records],
    )
```

- [ ] **Step 2: 修改 query 方法支持文檔過濾**

```python
def query(self, embedding: list[float], doc_ids: list[str] = None, 
          n_results: int = 5) -> list[dict]:
    """
    查詢相似的文本塊。
    
    Args:
        embedding: 查詢向量
        doc_ids: 可選的文檔 ID 列表，用於過濾
        n_results: 返回結果數量
        
    Returns:
        相似文本塊列表
    """
    if doc_ids:
        # 使用 ChromaDB 的 where 過濾
        res = self.col.query(
            query_embeddings=[embedding],
            n_results=n_results,
            where={"doc_id": {"$in": doc_ids}}
        )
    else:
        # 查詢所有文檔
        res = self.col.query(query_embeddings=[embedding], n_results=n_results)
    
    return [
        {
            "document": res["documents"][0][i],
            "metadata": res["metadatas"][0][i],
            "distance": res["distances"][0][i],
        }
        for i in range(len(res["documents"][0]))
    ]
```

- [ ] **Step 3: 添加文檔管理方法**

```python
def delete_document(self, doc_id: str) -> None:
    """
    刪除指定文檔的所有 chunks。
    
    Args:
        doc_id: 文檔 ID
    """
    try:
        self.col.delete(where={"doc_id": doc_id})
    except Exception as e:
        raise ValueError(f"刪除文檔失敗：{str(e)}")

def list_documents(self) -> list[str]:
    """
    返回所有已索引的文檔 ID 列表。
    
    Returns:
        文檔 ID 列表
    """
    try:
        results = self.col.get()
        doc_ids = set()
        if results and 'metadatas' in results:
            for metadata in results['metadatas']:
                if metadata and 'doc_id' in metadata:
                    doc_ids.add(metadata['doc_id'])
        return list(doc_ids)
    except Exception as e:
        raise ValueError(f"獲取文檔列表失敗：{str(e)}")
```

- [ ] **Step 4: 驗證修改**

Run: `python -c "from rag.vector_store import VectorStore; print('VectorStore 導入成功')"`
Expected: 輸出 "VectorStore 導入成功"

- [ ] **Step 5: Commit**

```bash
git add rag/vector_store.py
git commit -m "feat: add multi-document support to vector store"
```

---

## Task 10: 多文檔支持 - 重寫 ui/app.py

**Files:**
- Modify: `ui/app.py`

- [ ] **Step 1: 重寫 ui/app.py（第一部分：導入和配置）**

```python
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import tempfile
import uuid
from datetime import datetime
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from config import settings
from parser.pdf_parser import parse_pdf
from parser.chunker import Chunker
from rag.embedder import Embedder
from rag.vector_store import VectorStore
from rag.pipeline import Pipeline

st.set_page_config(
    page_title=settings.ui.page_title,
    layout="wide"
)
st.title("📚 PaperLens — 學術論文分析器")


@st.cache_resource
def get_embedder():
    return Embedder()


@st.cache_resource
def get_pipeline():
    return Pipeline()


@st.cache_resource
def get_vector_store():
    return VectorStore()
```

- [ ] **Step 2: 重寫 ui/app.py（第二部分：文檔管理函數）**

```python
def index_pdf(uploaded_file, doc_id: str):
    """
    索引 PDF 文件。
    
    Args:
        uploaded_file: Streamlit 上傳的文件對象
        doc_id: 文檔 ID
        
    Returns:
        chunk 數量
        
    Raises:
        ValueError: 索引失敗
    """
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        f.write(uploaded_file.read())
        tmp_path = f.name

    try:
        # 解析和分塊
        sections = parse_pdf(tmp_path)
        chunks = Chunker().chunk(sections)
        
        # 為每個 chunk 添加 doc_id
        for chunk in chunks:
            chunk.doc_id = doc_id
        
        # 索引
        embedder = get_embedder()
        store = get_vector_store()
        records = embedder.embed_chunks(chunks)
        store.add(records, doc_id)
        
        return len(chunks)
    finally:
        os.unlink(tmp_path)


# 初始化 session state
if 'documents' not in st.session_state:
    st.session_state.documents = []
```

- [ ] **Step 3: 重寫 ui/app.py（第三部分：側邊欄文檔管理）**

```python
# 側邊欄：文檔管理
with st.sidebar:
    st.header("📚 已索引文檔")
    
    if st.session_state.documents:
        for doc in st.session_state.documents:
            col1, col2, col3 = st.columns([4, 2, 1])
            col1.write(f"📄 {doc['filename']}")
            col2.write(f"{doc['num_chunks']} 塊")
            
            if col3.button("🗑️", key=f"del_{doc['doc_id']}", help="刪除此文檔"):
                try:
                    store = get_vector_store()
                    store.delete_document(doc['doc_id'])
                    st.session_state.documents = [
                        d for d in st.session_state.documents 
                        if d['doc_id'] != doc['doc_id']
                    ]
                    st.success(f"✅ 已刪除文檔：{doc['filename']}")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ 刪除失敗：{str(e)}")
        
        # 文檔選擇器
        st.divider()
        selected_docs = st.multiselect(
            "選擇查詢範圍",
            options=[d['doc_id'] for d in st.session_state.documents],
            format_func=lambda x: next(
                d['filename'] for d in st.session_state.documents 
                if d['doc_id'] == x
            ),
            help="不選擇則查詢所有文檔"
        )
    else:
        st.info("尚未索引任何文檔")
        selected_docs = []
```

- [ ] **Step 4: 重寫 ui/app.py（第四部分：主界面上傳和查詢）**

```python
# 主界面：上傳
uploaded = st.file_uploader("上傳 PDF 論文", type="pdf")

if uploaded:
    # 檢查文件大小
    file_size_mb = len(uploaded.getvalue()) / (1024 * 1024)
    if file_size_mb > settings.ui.max_upload_size:
        st.error(f"❌ 文件過大（{file_size_mb:.1f}MB），最大支持 {settings.ui.max_upload_size}MB")
    else:
        with st.spinner("正在索引論文..."):
            doc_id = str(uuid.uuid4())
            
            try:
                num_chunks = index_pdf(uploaded, doc_id)
                
                # 添加到文檔列表
                st.session_state.documents.append({
                    'doc_id': doc_id,
                    'filename': uploaded.name,
                    'upload_time': datetime.now(),
                    'num_chunks': num_chunks
                })
                
                st.success(f"✅ 成功索引 {num_chunks} 個文本塊")
                
            except ValueError as e:
                st.error(f"❌ 索引失敗：{str(e)}")
            except Exception as e:
                st.error(f"❌ 發生錯誤：{str(e)}")

# 主界面：查詢
st.divider()
question = st.text_input("請輸入您的問題")

if question:
    if not st.session_state.documents:
        st.warning("⚠️ 請先上傳並索引至少一個 PDF 文檔")
    else:
        with st.spinner("正在思考..."):
            try:
                # 獲取選中的文檔 ID（如果有）
                doc_ids = selected_docs if selected_docs else None
                result = get_pipeline().query(question, doc_ids=doc_ids)
                
                st.markdown(f"**回答** `[{result.route}]`\n\n{result.answer}")
                
                with st.expander("📎 參考來源"):
                    for s in result.sources:
                        st.write(f"- {s.section}（第 {s.page} 頁）")
                        
            except ValueError as e:
                st.error(f"❌ 查詢失敗：{str(e)}")
            except Exception as e:
                st.error(f"❌ 發生錯誤：{str(e)}")
```

- [ ] **Step 5: 驗證 UI 代碼語法**

Run: `python -c "import ast; ast.parse(open('ui/app.py').read()); print('UI 代碼語法正確')"`
Expected: 輸出 "UI 代碼語法正確"

- [ ] **Step 6: Commit**

```bash
git add ui/app.py
git commit -m "feat: rewrite UI with multi-document support and Chinese localization"
```

---

## Task 11: 中文化 - 更新 README.md

**Files:**
- Modify: `README.md`

- [ ] **Step 1: 重寫 README.md**

```markdown
# PaperLens-Agent

結構感知的學術論文分析系統 — 理解論文章節結構的 RAG 系統

## 功能特點

- PDF 結構檢測（摘要、引言、方法等）
- 結構感知的智能分塊（非簡單 token 切割）
- 查詢路由代理（摘要 / 章節 / 問答）
- 帶來源標註的回答（章節 + 頁碼）
- 多文檔支持和管理
- 交互式 Streamlit UI

## 技術棧

`streamlit` · `pymupdf` · `sentence-transformers` · `chromadb` · `openai` · `langchain` · `pydantic-settings`

## 安裝

```bash
pip install -r requirements.txt
cp .env.example .env
# 在 .env 中填入你的 OPENAI_API_KEY
```

## 配置

編輯 `config/config.yaml` 調整參數：
- OpenAI 模型和參數
- RAG chunk 大小和 overlap
- 向量存儲路徑
- UI 設置

## 運行

```bash
streamlit run ui/app.py
```

## 項目結構

```
parser/     # PDF 結構檢測 + 分塊
rag/        # Embedding, 向量存儲, RAG pipeline
agent/      # 查詢路由
ui/         # Streamlit 應用
models/     # Pydantic 數據模型
config/     # 配置管理
tests/      # 測試腳本
data/       # 示例 PDF 和向量數據庫
docs/       # 設計文檔和實施計劃
```

## 最近更新

- ✅ 配置管理系統（YAML + pydantic-settings）
- ✅ 增強錯誤處理和中文錯誤提示
- ✅ 優化分塊策略（LangChain + overlap）
- ✅ 多文檔支持和管理界面
- ✅ 改進 Prompt 質量和引用標註
- ✅ 全面中文化

## License

MIT
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: update README with Chinese localization and new features"
```

---

## Task 12: 中文化 - 更新 progress/progress.md

**Files:**
- Modify: `progress/progress.md`

- [ ] **Step 1: 更新 progress.md**

```markdown
# PaperLens-Agent 項目進度

## 項目信息
- 項目名：PaperLens-Agent (Structure-Aware Academic Paper Analyzer)
- 創建時間：2026-04-20
- 最近更新：2026-04-25
- 目標：結構感知的學術論文 RAG 分析系統

## 當前狀態
**階段：優化完成，生產就緒**

## 目錄結構
```
paperlens-agent/
├── parser/         # PDF 結構檢測 + 分塊
├── rag/            # Embedding + ChromaDB + RAG pipeline
├── agent/          # Query routing
├── ui/             # Streamlit app
├── models/         # Pydantic schemas
├── config/         # 配置管理（新增）
├── tests/          # 測試腳本
├── data/           # 測試 PDF 和向量數據庫
├── docs/           # 設計文檔和實施計劃（新增）
├── progress/       # 本文件夾
├── requirements.txt
├── README.md
└── .env.example
```

## 已完成任務

### 初始開發（2026-04-20）
- [x] 項目規劃與思路確認
- [x] 項目初始化（requirements.txt, .env.example, README, 目錄結構）
- [x] models/schemas.py — Pydantic 數據模型
- [x] parser/pdf_parser.py — PDF 結構檢測算法
- [x] parser/chunker.py — Structure-aware 分塊
- [x] rag/embedder.py — sentence-transformers 封裝
- [x] rag/vector_store.py — ChromaDB 封裝
- [x] agent/router.py — Query routing (summary/section/qa)
- [x] rag/pipeline.py — Retrieve → Generate → Attach source
- [x] ui/app.py — Streamlit UI
- [x] tests/ — 測試腳本

### 優化升級（2026-04-25）
- [x] 配置管理系統（config.yaml + pydantic-settings）
- [x] 錯誤處理增強（API、PDF 解析、UI）
- [x] 分塊優化（LangChain RecursiveCharacterTextSplitter + overlap）
- [x] Prompt 優化（結構化 prompt + 引用標註）
- [x] 多文檔支持（文檔管理界面 + 選擇性查詢）
- [x] 全面中文化（UI、錯誤信息、文檔）

## 技術棧
- **前端**: Streamlit
- **PDF 處理**: PyMuPDF
- **向量存儲**: ChromaDB
- **Embedding**: sentence-transformers
- **LLM**: OpenAI GPT-4o-mini
- **分塊**: LangChain text-splitters
- **配置**: pydantic-settings + PyYAML

## 下一步計劃
- [ ] 添加圖表提取功能
- [ ] 實現混合檢索（BM25 + 向量）
- [ ] 添加對話歷史功能
- [ ] 部署到雲端（Docker + K8s）
```

- [ ] **Step 2: Commit**

```bash
git add progress/progress.md
git commit -m "docs: update progress with optimization completion"
```

---

## Task 13: 測試 - 配置管理驗證

**Files:**
- Test: `config/`

- [ ] **Step 1: 測試配置加載**

Run: `cd /tmp/paperlens-agent && python -c "from config import settings; print(f'Model: {settings.openai.model}'); print(f'Chunk size: {settings.rag.chunk_size}')"`
Expected: 輸出配置值

- [ ] **Step 2: 測試環境變量覆蓋**

Run: `cd /tmp/paperlens-agent && export OPENAI_API_KEY=test-key-123 && python -c "from config import settings; print(f'API Key: {settings.openai.api_key}')"`
Expected: 輸出 "API Key: test-key-123"

- [ ] **Step 3: 記錄測試結果**

創建測試記錄：配置管理測試通過

---

## Task 14: 測試 - 端到端功能驗證

**Files:**
- Test: 整個應用

- [ ] **Step 1: 啟動應用**

Run: `cd /tmp/paperlens-agent && streamlit run ui/app.py`
Expected: 應用啟動，無錯誤

- [ ] **Step 2: 測試上傳功能**

手動操作：
1. 在瀏覽器中打開應用
2. 上傳一個測試 PDF 文件
3. 等待索引完成
4. 檢查側邊欄是否顯示文檔

Expected: 文檔成功索引，側邊欄顯示文檔信息

- [ ] **Step 3: 測試查詢功能**

手動操作：
1. 在輸入框輸入問題
2. 檢查回答質量
3. 檢查是否包含引用標註
4. 展開"參考來源"查看來源信息

Expected: 回答準確，包含引用標註

- [ ] **Step 4: 測試多文檔功能**

手動操作：
1. 上傳第二個 PDF 文檔
2. 在側邊欄選擇特定文檔
3. 提問並檢查是否只從選中文檔檢索
4. 測試刪除文檔功能

Expected: 多文檔管理正常工作

- [ ] **Step 5: 測試錯誤處理**

手動操作：
1. 嘗試上傳損壞的 PDF（如果有）
2. 嘗試上傳超大文件
3. 在空庫中查詢

Expected: 所有錯誤都有清晰的中文提示

---

## Task 15: 最終提交

**Files:**
- All modified files

- [ ] **Step 1: 檢查所有修改**

Run: `cd /tmp/paperlens-agent && git status`
Expected: 列出所有修改的文件

- [ ] **Step 2: 確認所有測試通過**

確認：
- 配置管理測試通過
- 端到端功能測試通過
- 錯誤處理測試通過

- [ ] **Step 3: 創建最終提交**

```bash
cd /tmp/paperlens-agent
git add -A
git commit -m "feat: complete PaperLens-Agent optimization

- Add configuration management with YAML and pydantic-settings
- Enhance error handling with Chinese error messages
- Optimize chunking with LangChain RecursiveCharacterTextSplitter
- Add multi-document support with management UI
- Improve prompt quality with structured templates
- Complete Chinese localization for all user-facing content

Co-authored-by: lduo8438-max <lduo8438-max@users.noreply.github.com>"
```

- [ ] **Step 4: 推送到 GitHub**

```bash
git push origin main
```

Expected: 成功推送到遠程倉庫

---

## 驗收標準

完成後，系統應滿足：

1. ✅ **配置管理** - 所有參數可通過 config.yaml 調整
2. ✅ **錯誤處理** - 所有錯誤都有清晰的中文提示，不會崩潰
3. ✅ **分塊質量** - 使用智能分塊，有 200 字符 overlap，邊界合理
4. ✅ **多文檔支持** - 可以同時索引多個文檔，選擇性查詢，刪除單個文檔
5. ✅ **回答質量** - 使用結構化 prompt，包含引用標註
6. ✅ **中文化** - 所有面向用戶的內容使用簡體中文
7. ✅ **向後兼容** - 現有功能不受影響

## 注意事項

1. **環境變量**: 確保 `.env` 文件中設置了有效的 `OPENAI_API_KEY`
2. **依賴安裝**: 運行前執行 `pip install -r requirements.txt`
3. **數據持久化**: 向量數據庫保存在 `./data/chroma`，可以持久化
4. **文件大小限制**: 默認最大上傳 50MB，可在 `config.yaml` 中調整
5. **測試數據**: 建議使用 arXiv 上的公開論文進行測試

