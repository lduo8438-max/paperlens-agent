import os
import chromadb
from chromadb.config import Settings

_DEFAULT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "chroma")


class VectorStore:
    def __init__(self, path: str = _DEFAULT_PATH, collection: str = "paperlens"):
        self.client = chromadb.PersistentClient(
            path=path, settings=Settings(anonymized_telemetry=False)
        )
        self.col = self.client.get_or_create_collection(collection)

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

    def reset(self) -> None:
        self.client.delete_collection(self.col.name)
        self.col = self.client.get_or_create_collection(self.col.name)
