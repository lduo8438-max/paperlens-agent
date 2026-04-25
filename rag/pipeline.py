import time
from openai import OpenAI
from agent.router import route
from models.schemas import QueryResult, Source
from rag.embedder import Embedder
from rag.vector_store import VectorStore
from config import settings


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


class Pipeline:
    def __init__(self):
        self.embedder = Embedder()
        self.store = VectorStore()
        try:
            self.client = OpenAI(api_key=settings.openai.api_key)
        except Exception as e:
            raise ValueError(f"初始化 OpenAI 客戶端失敗：{str(e)}")

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
