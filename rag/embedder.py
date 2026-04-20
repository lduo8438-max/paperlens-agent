from sentence_transformers import SentenceTransformer
from models.schemas import Chunk


class Embedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_chunks(self, chunks: list[Chunk]) -> list[dict]:
        texts = [c.text for c in chunks]
        vectors = self.model.encode(texts, show_progress_bar=False)
        return [
            {
                "id": c.chunk_id,
                "embedding": vectors[i].tolist(),
                "document": c.text,
                "metadata": {"section": c.section, "page": c.page},
            }
            for i, c in enumerate(chunks)
        ]

    def embed_query(self, query: str) -> list[float]:
        return self.model.encode(query).tolist()
