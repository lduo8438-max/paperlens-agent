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
