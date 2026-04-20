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

    def add(self, records: list[dict]) -> None:
        self.col.add(
            ids=[r["id"] for r in records],
            embeddings=[r["embedding"] for r in records],
            documents=[r["document"] for r in records],
            metadatas=[r["metadata"] for r in records],
        )

    def query(self, embedding: list[float], n_results: int = 5) -> list[dict]:
        res = self.col.query(query_embeddings=[embedding], n_results=n_results)
        return [
            {
                "document": res["documents"][0][i],
                "metadata": res["metadatas"][0][i],
                "distance": res["distances"][0][i],
            }
            for i in range(len(res["documents"][0]))
        ]

    def reset(self) -> None:
        self.client.delete_collection(self.col.name)
        self.col = self.client.get_or_create_collection(self.col.name)
