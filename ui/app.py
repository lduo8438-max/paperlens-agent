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
