# PaperLens-Agent 项目进度

## 项目信息
- 项目名：PaperLens-Agent (Structure-Aware Academic Paper Analyzer)
- 创建时间：2026-04-20
- 目标：结构感知的学术论文 RAG 分析系统

## 当前状态
**阶段：规划完成，待开始编码**

## 目录结构规划
```
paperlens-agent/
├── parser/         # PDF 结构检测 + 分块
├── rag/            # Embedding + ChromaDB + RAG pipeline
├── agent/          # Query routing
├── ui/             # Streamlit app
├── tests/          # 测试脚本
├── data/           # 测试 PDF
├── models/         # Pydantic schemas
├── progress/       # 本文件夹
├── requirements.txt
├── README.md
└── .env.example
```

## 待完成任务
- [ ] 1. 项目初始化（requirements.txt, .env.example, README）
- [x] 2. models/schemas.py — Pydantic 数据模型
- [x] 3. parser/pdf_parser.py — PDF 结构检测算法
- [x] 4. parser/chunker.py — Structure-aware 分块
- [x] 5. rag/embedder.py — sentence-transformers 封装
- [x] 6. rag/vector_store.py — ChromaDB 封装
- [x] 7. agent/router.py — Query routing (summary/section/qa)
- [x] 8. rag/pipeline.py — Retrieve → Generate → Attach source
- [x] 9. ui/app.py — Streamlit UI
- [x] 10. tests/ — 测试脚本 + 测试数据

## 已完成
- [x] 项目规划与思路确认
- [x] 创建进度追踪文件夹
- [x] 1. 项目初始化（requirements.txt, .env.example, README, 目录结构, __init__.py）

## 下次继续
读取本文件，从"待完成任务"第一个未勾选项开始。
