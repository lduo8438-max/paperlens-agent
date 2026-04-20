import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import tempfile
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from parser.pdf_parser import PDFParser
from parser.chunker import Chunker
from rag.embedder import Embedder
from rag.vector_store import VectorStore
from rag.pipeline import Pipeline

st.set_page_config(page_title="PaperLens", layout="wide")
st.title("PaperLens — Academic Paper Analyzer")


@st.cache_resource
def get_embedder():
    return Embedder()


@st.cache_resource
def get_pipeline():
    return Pipeline()


def index_pdf(uploaded_file):
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        f.write(uploaded_file.read())
        tmp_path = f.name

    parser = PDFParser(tmp_path)
    sections = parser.extract_sections()
    chunks = Chunker().chunk(sections)

    embedder = get_embedder()
    store = VectorStore()
    store.reset()
    records = embedder.embed_chunks(chunks)
    store.add(records)
    os.unlink(tmp_path)
    return len(chunks)


uploaded = st.file_uploader("Upload a PDF paper", type="pdf")

if uploaded:
    with st.spinner("Indexing paper..."):
        n = index_pdf(uploaded)
    st.success(f"Indexed {n} chunks.")

    question = st.text_input("Ask a question about the paper")
    if question:
        with st.spinner("Thinking..."):
            result = get_pipeline().query(question)
        st.markdown(f"**Answer** `[{result.route}]`\n\n{result.answer}")
        with st.expander("Sources"):
            for s in result.sources:
                st.write(f"- {s.section}, page {s.page}")
