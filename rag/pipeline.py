import os
from openai import OpenAI
from agent.router import route
from models.schemas import QueryResult, Source
from rag.embedder import Embedder
from rag.vector_store import VectorStore


class Pipeline:
    def __init__(self):
        self.embedder = Embedder()
        self.store = VectorStore()
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    def query(self, question: str) -> QueryResult:
        route_type = route(question)
        embedding = self.embedder.embed_query(question)
        hits = self.store.query(embedding, n_results=5)

        context = "\n\n".join(
            f"[{h['metadata']['section']}, p{h['metadata']['page']}]\n{h['document']}"
            for h in hits
        )
        prompt = f"Answer based on the following excerpts:\n\n{context}\n\nQuestion: {question}"

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
        )
        answer = response.choices[0].message.content

        sources = [
            Source(section=h["metadata"]["section"], page=h["metadata"]["page"])
            for h in hits
        ]
        return QueryResult(answer=answer, sources=sources, route=route_type)
