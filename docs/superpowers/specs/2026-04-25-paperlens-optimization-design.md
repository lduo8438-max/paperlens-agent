# PaperLens-Agent 優化設計文檔

**日期：** 2026-04-25  
**作者：** lduo8438-max  
**版本：** 1.0  
**狀態：** 已批准

## 1. 概述

### 1.1 優化目標

在保持現有架構的基礎上，通過 5 項關鍵改進提升 PaperLens-Agent 系統的穩定性、可維護性和用戶體驗。

### 1.2 優化範圍

**5 項核心改進：**
1. 配置管理 - 創建 `config.yaml` 統一管理所有參數
2. 錯誤處理 - 在關鍵路徑添加異常處理和用戶友好提示
3. 分塊優化 - 使用 LangChain 的 RecursiveCharacterTextSplitter + overlap
4. 多文檔支持 - 移除 `store.reset()`，添加文檔 ID 和管理界面
5. Prompt 優化 - 使用結構化 prompt 提升回答質量

**語言要求：**
- 所有面向用戶的內容（UI、錯誤信息、文檔）使用簡體中文
- 代碼變量名、函數名保持英文

### 1.3 不改動的部分

- PDF 解析邏輯（`parser/pdf_parser.py`）
- 查詢路由（`agent/router.py`）
- Embedder 封裝（`rag/embedder.py`）
- 數據模型基礎結構（`models/schemas.py`）

## 2. 配置管理設計

### 2.1 文件結構

```
config/
├── __init__.py          # 導出 settings 對象
├── settings.py          # 配置加載邏輯
└── config.yaml          # 配置文件（新增）
```

### 2.2 config.yaml 結構

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

### 2.3 settings.py 實現

**技術選型：**
- 使用 `pydantic-settings` 進行配置管理
- 支持環境變量覆蓋
- 提供類型驗證和默認值
- 單例模式，全局只加載一次

**使用方式：**
```python
from config import settings

# 訪問配置
model = settings.openai.model
chunk_size = settings.rag.chunk_size
```

### 2.4 依賴更新

```
requirements.txt 添加：
pyyaml>=6.0
pydantic-settings>=2.0.0
```

## 3. 錯誤處理設計

### 3.1 rag/pipeline.py - API 調用錯誤

**處理場景：**
- OpenAI API 密鑰無效或缺失
- API 請求超時或限流
- 網絡連接失敗
- 檢索結果為空

**錯誤處理策略：**
- 捕獲 `openai.OpenAIError` 異常
- 添加重試機制（最多 3 次，指數退避）
- 檢索為空時返回友好提示
- 所有錯誤信息使用簡體中文

**示例：**
```python
try:
    response = self.client.chat.completions.create(...)
except openai.AuthenticationError:
    raise ValueError("OpenAI API 密鑰無效，請檢查配置")
except openai.RateLimitError:
    raise ValueError("API 請求頻率超限，請稍後重試")
except openai.APIError as e:
    raise ValueError(f"API 請求失敗：{str(e)}")
```

### 3.2 parser/pdf_parser.py - PDF 解析錯誤

**處理場景：**
- PDF 文件損壞或加密
- 無法提取文本（掃描版 PDF）
- 文件格式不正確

**錯誤處理策略：**
- 捕獲 `fitz.FileDataError` 等異常
- 返回具體錯誤原因
- 提供降級方案（如果無法檢測結構，按頁分塊）

**示例：**
```python
try:
    doc = fitz.open(pdf_path)
except fitz.FileDataError:
    raise ValueError("PDF 文件損壞或格式不正確")
except fitz.PasswordError:
    raise ValueError("PDF 已加密，請先解密後再上傳")
```

### 3.3 ui/app.py - 文件上傳和索引錯誤

**處理場景：**
- 上傳文件過大
- 上傳非 PDF 文件
- 索引過程中斷
- 向量存儲寫入失敗

**錯誤處理策略：**
- 文件大小檢查（使用 `settings.ui.max_upload_size`）
- 使用 `st.error()` 顯示錯誤信息
- 索引失敗時清理臨時文件
- 使用 try-except 包裹索引邏輯

### 3.4 rag/vector_store.py - 數據庫錯誤

**處理場景：**
- ChromaDB 初始化失敗
- 磁盤空間不足
- 權限問題

**錯誤處理策略：**
- 捕獲 ChromaDB 相關異常
- 檢查數據目錄是否可寫
- 提供清晰的錯誤信息和解決建議

## 4. 分塊優化設計

### 4.1 當前問題

- 使用固定 1000 字符切割
- 只在句號處斷開，可能在段落中間切斷
- 沒有 chunk overlap，丟失上下文信息

### 4.2 新方案

**使用 LangChain 的 RecursiveCharacterTextSplitter：**

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 分隔符優先級：段落 > 句子 > 詞
separators = [
    "\n\n",      # 段落
    "\n",        # 換行
    "。", "！", "？",  # 中文句子結束
    ". ", "! ", "? ",  # 英文句子結束
    "；", "; ",        # 分號
    "，", ", ",        # 逗號
    " ",               # 空格
    ""                 # 字符
]

splitter = RecursiveCharacterTextSplitter(
    chunk_size=settings.rag.chunk_size,      # 從配置讀取
    chunk_overlap=settings.rag.chunk_overlap, # 從配置讀取
    length_function=len,
    separators=separators
)
```

### 4.3 保持結構感知

```python
def chunk_sections(sections: list[Section]) -> list[Chunk]:
    """將論文章節轉換為文本塊，保留結構元數據"""
    splitter = RecursiveCharacterTextSplitter(...)
    chunks = []
    
    for section in sections:
        if not section.text.strip():
            continue
            
        # 使用智能分割
        chunk_texts = splitter.split_text(section.text)
        
        for i, text in enumerate(chunk_texts):
            chunks.append(Chunk(
                chunk_id=f"{section.name}_{i}",
                text=text,
                section=section.name,
                page=section.page
            ))
    
    return chunks
```

### 4.4 Overlap 的好處

- 200 字符重疊確保跨 chunk 的信息不會丟失
- 檢索時更容易找到完整的語義單元
- 對於跨句子的問題回答更準確

### 4.5 依賴更新

```
requirements.txt 添加：
langchain>=0.1.0
langchain-text-splitters>=0.0.1
```

## 5. 多文檔支持設計

### 5.1 當前問題

- 每次上傳新 PDF 會調用 `store.reset()`，清空所有數據
- 無法同時分析多篇論文
- 無法建立持久的知識庫

### 5.2 數據模型擴展

**新增 Document 模型：**
```python
# models/schemas.py
from datetime import datetime
from pydantic import BaseModel

class Document(BaseModel):
    doc_id: str              # 唯一標識符（UUID）
    filename: str            # 原始文件名
    upload_time: datetime    # 上傳時間
    num_chunks: int          # chunk 數量
```

**擴展 Chunk 模型：**
```python
class Chunk(BaseModel):
    chunk_id: str
    doc_id: str              # 新增：關聯到文檔
    text: str
    section: str
    page: int
```

### 5.3 向量存儲修改

**rag/vector_store.py 新增方法：**

```python
class VectorStore:
    def add(self, records: list[dict], doc_id: str) -> None:
        """添加文檔的 chunks，每個 chunk 的 metadata 包含 doc_id"""
        for record in records:
            record['metadata']['doc_id'] = doc_id
        # 原有邏輯...
    
    def query(self, embedding: list[float], doc_ids: list[str] = None, 
              n_results: int = 5) -> list[dict]:
        """查詢，支持按 doc_id 過濾"""
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
        # 原有邏輯...
    
    def delete_document(self, doc_id: str) -> None:
        """刪除指定文檔的所有 chunks"""
        self.col.delete(where={"doc_id": doc_id})
    
    def list_documents(self) -> list[str]:
        """返回所有已索引的文檔 ID 列表"""
        # 從 collection 中獲取所有唯一的 doc_id
        results = self.col.get()
        doc_ids = set()
        for metadata in results['metadatas']:
            if 'doc_id' in metadata:
                doc_ids.add(metadata['doc_id'])
        return list(doc_ids)
```

### 5.4 UI 改進

**ui/app.py 主要變更：**

**1. 使用 session_state 管理文檔列表：**
```python
import uuid
from datetime import datetime

if 'documents' not in st.session_state:
    st.session_state.documents = []
```

**2. 側邊欄：文檔管理**
```python
with st.sidebar:
    st.header("📚 已索引文檔")
    
    if st.session_state.documents:
        for doc in st.session_state.documents:
            col1, col2 = st.columns([3, 1])
            col1.write(f"📄 {doc['filename']}")
            col2.write(f"{doc['num_chunks']} 塊")
            
            if st.button("🗑️", key=f"del_{doc['doc_id']}", help="刪除此文檔"):
                store = VectorStore()
                store.delete_document(doc['doc_id'])
                st.session_state.documents = [
                    d for d in st.session_state.documents 
                    if d['doc_id'] != doc['doc_id']
                ]
                st.rerun()
        
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
```

**3. 主界面：上傳不再清空**
```python
uploaded = st.file_uploader("上傳 PDF 論文", type="pdf")

if uploaded:
    with st.spinner("正在索引論文..."):
        doc_id = str(uuid.uuid4())
        
        # 保存臨時文件
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(uploaded.read())
            tmp_path = f.name
        
        try:
            # 解析和分塊
            parser = PDFParser(tmp_path)
            sections = parser.extract_sections()
            chunks = Chunker().chunk(sections)
            
            # 索引（傳入 doc_id）
            embedder = get_embedder()
            store = VectorStore()
            records = embedder.embed_chunks(chunks)
            
            # 為每個 record 添加 doc_id
            for record in records:
                record['metadata']['doc_id'] = doc_id
            
            store.add(records, doc_id)
            
            # 添加到文檔列表
            st.session_state.documents.append({
                'doc_id': doc_id,
                'filename': uploaded.name,
                'upload_time': datetime.now(),
                'num_chunks': len(chunks)
            })
            
            st.success(f"✅ 成功索引 {len(chunks)} 個文本塊")
            
        except Exception as e:
            st.error(f"❌ 索引失敗：{str(e)}")
        finally:
            os.unlink(tmp_path)
```

**4. 查詢邏輯**
```python
question = st.text_input("請輸入您的問題")
if question:
    with st.spinner("正在思考..."):
        # 獲取選中的文檔 ID（如果有）
        doc_ids = selected_docs if selected_docs else None
        result = get_pipeline().query(question, doc_ids=doc_ids)
    
    st.markdown(f"**回答** `[{result.route}]`\n\n{result.answer}")
    
    with st.expander("📎 參考來源"):
        for s in result.sources:
            st.write(f"- {s.section}（第 {s.page} 頁）")
```

### 5.5 Pipeline 修改

**rag/pipeline.py 支持文檔過濾：**

```python
def query(self, question: str, doc_ids: list[str] = None) -> QueryResult:
    """查詢，支持指定文檔範圍"""
    route_type = route(question)
    embedding = self.embedder.embed_query(question)
    
    # 傳入 doc_ids 進行過濾
    hits = self.store.query(embedding, doc_ids=doc_ids, n_results=settings.rag.top_k)
    
    # 原有邏輯...
```

## 6. Prompt 優化設計

### 6.1 當前問題

- Prompt 過於簡單，缺乏明確指令
- 沒有引導 LLM 標註引用來源
- 沒有處理信息不足的情況
- 沒有使用 system/user 角色分離

### 6.2 結構化 Prompt 模板

**rag/pipeline.py 新增常量：**

```python
SYSTEM_PROMPT = """你是一個專業的學術論文分析助手。你的任務是基於提供的論文片段回答用戶問題。

回答要求：
1. 準確性：僅基於提供的內容回答，不要編造信息
2. 引用：在回答中使用 [章節名, 第X頁] 格式標註信息來源
3. 完整性：如果信息不足以回答問題，明確說明
4. 語言：使用簡體中文，保持學術性和專業性
"""

USER_PROMPT_TEMPLATE = """以下是從論文中檢索到的相關片段：

{context}

問題：{question}

請基於上述片段回答問題。記得標註信息來源。"""
```

### 6.3 Context 格式優化

**改進前：**
```
[Introduction, p1]
This is the text...
```

**改進後：**
```
=== 片段 1 ===
來源：Introduction（第 1 頁）
內容：This is the text...

=== 片段 2 ===
來源：Method（第 3 頁）
內容：Another text...
```

**實現：**
```python
context_parts = []
for i, hit in enumerate(hits, 1):
    context_parts.append(
        f"=== 片段 {i} ===\n"
        f"來源：{hit['metadata']['section']}（第 {hit['metadata']['page']} 頁）\n"
        f"內容：{hit['document']}"
    )
context = "\n\n".join(context_parts)
```

### 6.4 處理特殊情況

**檢索結果為空：**
```python
if not hits:
    return QueryResult(
        answer="抱歉，我在已索引的論文中沒有找到與您問題相關的內容。請嘗試換個問法或上傳相關論文。",
        sources=[],
        route=route_type
    )
```

**相似度過低（未來實現）：**
```python
# 當所有結果的 distance 都超過閾值時
if all(hit['distance'] > settings.rag.similarity_threshold for hit in hits):
    answer_prefix = "找到了一些內容，但相關性較低。以下是我基於現有信息的回答：\n\n"
```

### 6.5 使用 OpenAI 的角色系統

**改進後的 API 調用：**
```python
user_prompt = USER_PROMPT_TEMPLATE.format(context=context, question=question)

response = self.client.chat.completions.create(
    model=settings.openai.model,
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ],
    temperature=settings.openai.temperature,
    max_tokens=settings.openai.max_tokens
)

answer = response.choices[0].message.content
```

### 6.6 回答後處理

**確保包含引用：**
```python
# 如果 LLM 沒有添加引用標註，自動在末尾添加
if "[" not in answer and "（第" not in answer:
    answer += "\n\n**參考來源：**\n"
    for source in sources:
        answer += f"- {source.section}（第 {source.page} 頁）\n"
```

## 7. 中文化設計

### 7.1 UI 界面文本

**ui/app.py 所有面向用戶的文本：**

```python
# 頁面配置
st.set_page_config(
    page_title=settings.ui.page_title,
    layout="wide"
)
st.title("📚 PaperLens — 學術論文分析器")

# 上傳組件
uploaded = st.file_uploader("上傳 PDF 論文", type="pdf")

# 狀態提示
st.spinner("正在索引論文...")
st.success(f"✅ 成功索引 {n} 個文本塊")
st.error(f"❌ 索引失敗：{error_message}")

# 輸入框
question = st.text_input("請輸入您的問題")

# 結果展示
st.markdown(f"**回答** `[{result.route}]`\n\n{result.answer}")
with st.expander("📎 參考來源"):
    for s in result.sources:
        st.write(f"- {s.section}（第 {s.page} 頁）")
```

### 7.2 README.md 更新

**主要內容改為簡體中文：**

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

`streamlit` · `pymupdf` · `sentence-transformers` · `chromadb` · `openai` · `langchain`

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
```
```

### 7.3 代碼註釋

**所有 docstring 和註釋改為簡體中文：**

```python
def chunk_sections(sections: list[Section]) -> list[Chunk]:
    """
    將論文章節轉換為文本塊，保留結構元數據。
    
    Args:
        sections: 論文章節列表
        
    Returns:
        包含結構元數據的文本塊列表
    """
    pass

class VectorStore:
    """向量存儲封裝，基於 ChromaDB"""
    
    def query(self, embedding: list[float], doc_ids: list[str] = None) -> list[dict]:
        """
        查詢相似的文本塊。
        
        Args:
            embedding: 查詢向量
            doc_ids: 可選的文檔 ID 列表，用於過濾
            
        Returns:
            相似文本塊列表
        """
        pass
```

### 7.4 錯誤信息

**所有用戶可見的錯誤信息使用簡體中文：**

```python
raise ValueError("配置文件不存在，請檢查 config/config.yaml")
raise FileNotFoundError("PDF 文件未找到")
raise ValueError("OpenAI API 密鑰無效，請檢查配置")
```

### 7.5 配置文件註釋

**config.yaml 中的所有註釋使用簡體中文：**

```yaml
# PaperLens-Agent 配置文件

# OpenAI 配置
openai:
  api_key: ${OPENAI_API_KEY}  # 從環境變量讀取
  model: gpt-4o-mini          # 使用的模型
  temperature: 0.7            # 生成溫度，越高越隨機
```

### 7.6 保持英文的內容

- 變量名、函數名、類名（遵循 Python 命名規範）
- 技術術語（如 RAG, embedding, chunk, pipeline）
- 依賴包名稱
- Git commit 信息（可選，建議用英文）

## 8. 實施順序和測試策略

### 8.1 實施順序

按依賴關係從底層到上層：

```
1. 配置管理（基礎設施）
   ├─ 創建 config/ 目錄結構
   ├─ 編寫 config.yaml
   ├─ 實現 settings.py
   └─ 更新 requirements.txt
   
2. 錯誤處理（橫切關注點）
   ├─ rag/pipeline.py 添加異常處理
   ├─ parser/pdf_parser.py 添加異常處理
   ├─ ui/app.py 添加異常處理
   └─ rag/vector_store.py 添加異常處理
   
3. 分塊優化（獨立模塊）
   ├─ 更新 requirements.txt（添加 langchain）
   ├─ 重寫 parser/chunker.py
   └─ 更新相關導入
   
4. Prompt 優化（獨立模塊）
   ├─ 在 rag/pipeline.py 添加 prompt 常量
   ├─ 修改 query 方法使用新 prompt
   └─ 添加特殊情況處理
   
5. 多文檔支持（UI + 存儲層）
   ├─ 擴展 models/schemas.py（Document 模型）
   ├─ 修改 rag/vector_store.py（添加方法）
   ├─ 修改 rag/pipeline.py（支持 doc_ids）
   └─ 重寫 ui/app.py（文檔管理界面）
   
6. 中文化和文檔更新（收尾）
   ├─ 更新 README.md
   ├─ 更新所有代碼註釋
   ├─ 更新 UI 文本
   └─ 更新 progress/progress.md
```

### 8.2 測試策略

**每個改進的驗證方法：**

**1. 配置管理**
```bash
# 測試配置加載
python -c "from config import settings; print(settings.openai.model)"

# 測試環境變量覆蓋
export OPENAI_API_KEY=test-key
python -c "from config import settings; print(settings.openai.api_key)"
```

**2. 錯誤處理**
- 上傳損壞的 PDF 文件
- 使用無效的 API key
- 上傳超大文件（> max_upload_size）
- 查詢空的向量庫
- 驗證：每個錯誤都有清晰的中文提示，UI 不崩潰

**3. 分塊優化**
```bash
# 運行測試
pytest tests/test_parser.py

# 手動檢查
python -c "
from parser.pdf_parser import PDFParser
from parser.chunker import Chunker
sections = PDFParser('test.pdf').extract_sections()
chunks = Chunker().chunk(sections)
print(f'生成 {len(chunks)} 個 chunks')
print(f'第一個 chunk: {chunks[0].text[:100]}...')
"
```

**4. Prompt 優化**
- 在 UI 中提問並檢查回答格式
- 確認包含引用標註（[章節, 第X頁]）
- 測試檢索為空的情況
- 對比新舊回答質量

**5. 多文檔支持**
- 上傳 2-3 個測試 PDF
- 檢查側邊欄文檔列表顯示正確
- 刪除單個文檔，驗證其他文檔不受影響
- 測試跨文檔查詢
- 測試指定文檔查詢

**6. 中文化**
- 瀏覽整個應用，檢查所有 UI 文本
- 觸發各種錯誤，檢查錯誤信息
- 閱讀 README 和代碼註釋

### 8.3 回歸測試

每完成一項改進後，運行完整流程：
1. 啟動應用：`streamlit run ui/app.py`
2. 上傳 PDF 文件
3. 等待索引完成
4. 提問並檢查回答
5. 檢查來源標註
6. 測試多文檔功能（如果已實現）

### 8.4 測試數據

使用 arXiv 上的公開論文：
- 機器學習經典論文（如 Attention Is All You Need）
- 不同格式的 PDF（單欄、雙欄）
- 中英文混合論文

## 9. 文件清單

### 9.1 新增文件

```
config/
├── __init__.py
├── settings.py
└── config.yaml

docs/superpowers/specs/
└── 2026-04-25-paperlens-optimization-design.md
```

### 9.2 修改文件

```
parser/chunker.py          # 使用 LangChain splitter
rag/pipeline.py            # 添加錯誤處理、優化 prompt、支持 doc_ids
rag/vector_store.py        # 添加文檔管理方法
ui/app.py                  # 重寫文檔管理界面
models/schemas.py          # 添加 Document 模型，擴展 Chunk
requirements.txt           # 添加新依賴
README.md                  # 更新為簡體中文
progress/progress.md       # 更新進度
```

### 9.3 不修改文件

```
parser/pdf_parser.py       # PDF 解析邏輯保持不變
agent/router.py            # 查詢路由保持不變
rag/embedder.py            # Embedder 封裝保持不變
tests/test_parser.py       # 測試保持不變（可能需要小幅調整）
```

## 10. 依賴更新

**requirements.txt 完整內容：**

```
streamlit>=1.32.0
pymupdf>=1.23.0
sentence-transformers>=2.7.0
chromadb>=0.4.24
openai>=1.30.0
numpy>=1.26.0
pydantic>=2.6.0
python-dotenv>=1.0.0
pyyaml>=6.0
pydantic-settings>=2.0.0
langchain>=0.1.0
langchain-text-splitters>=0.0.1
```

## 11. 成功標準

優化完成後，系統應滿足：

1. **配置管理** - 所有參數可通過 config.yaml 調整
2. **錯誤處理** - 所有錯誤都有清晰的中文提示，不會崩潰
3. **分塊質量** - 使用智能分塊，有 overlap，邊界合理
4. **多文檔支持** - 可以同時索引多個文檔，選擇性查詢
5. **回答質量** - 使用結構化 prompt，包含引用標註
6. **中文化** - 所有面向用戶的內容使用簡體中文
7. **向後兼容** - 現有功能不受影響

## 12. 風險和緩解

**風險 1：LangChain 依賴增加項目複雜度**
- 緩解：只使用 text_splitter 模塊，不引入其他組件
- 如果有問題，可以回退到優化後的自定義分塊邏輯

**風險 2：多文檔支持可能影響查詢性能**
- 緩解：ChromaDB 的 where 過濾是高效的
- 如果文檔數量很大，可以添加索引優化

**風險 3：UI 重寫可能引入新 bug**
- 緩解：保持核心邏輯不變，只修改界面部分
- 充分測試各種場景

**風險 4：配置管理可能破壞現有部署**
- 緩解：保留 .env 文件支持，config.yaml 作為補充
- 提供清晰的遷移指南

