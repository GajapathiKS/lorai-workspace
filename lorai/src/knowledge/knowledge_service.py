"""LorAI Knowledge Service — ChromaDB-backed RAG.

Handles document ingestion, semantic search, and question answering
using ChromaDB for vector storage and Ollama for LLM responses.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import chromadb

VECTOR_DB_PATH = "/data/vectors"
SUPPORTED_EXTENSIONS = {
    ".txt", ".md", ".py", ".js", ".ts", ".json", ".csv", ".html",
    ".xml", ".yaml", ".yml", ".toml", ".cfg", ".ini", ".sh", ".sql",
    ".r", ".java", ".go", ".rs",
}


class KnowledgeService:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=VECTOR_DB_PATH)

    def get_collection(self, name: str = "default"):
        return self.client.get_or_create_collection(
            name=name, metadata={"hnsw:space": "cosine"},
        )

    def ingest(
        self,
        sources: list[str],
        collection: str = "default",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ) -> dict:
        coll = self.get_collection(collection)
        total_docs = 0
        total_chunks = 0

        for source in sources:
            source = os.path.expanduser(source)
            if os.path.isdir(source):
                files = [f for f in Path(source).rglob("*") if f.is_file()]
            elif os.path.isfile(source):
                files = [Path(source)]
            else:
                continue

            for filepath in files:
                if filepath.suffix.lower() not in SUPPORTED_EXTENSIONS:
                    continue
                try:
                    text = filepath.read_text(encoding="utf-8", errors="ignore")
                    if not text.strip():
                        continue
                    chunks = self._chunk(text, chunk_size, chunk_overlap)
                    file_id = hashlib.md5(str(filepath).encode()).hexdigest()

                    ids = [f"{file_id}_{i}" for i in range(len(chunks))]
                    metadatas = [
                        {
                            "source": str(filepath),
                            "chunk": i,
                            "total_chunks": len(chunks),
                        }
                        for i in range(len(chunks))
                    ]

                    coll.upsert(ids=ids, documents=chunks, metadatas=metadatas)
                    total_docs += 1
                    total_chunks += len(chunks)
                except Exception:
                    continue

        return {
            "documents_ingested": total_docs,
            "chunks_created": total_chunks,
            "collection": collection,
        }

    def search(
        self,
        query: str,
        collection: str = "default",
        top_k: int = 5,
        threshold: float = 0.7,
    ) -> list[dict]:
        coll = self.get_collection(collection)
        if coll.count() == 0:
            return []

        results = coll.query(
            query_texts=[query], n_results=min(top_k, coll.count()),
        )

        output = []
        for i in range(len(results["ids"][0])):
            score = 1.0 - (results["distances"][0][i] if results["distances"] else 1.0)
            if score >= threshold:
                output.append(
                    {
                        "content": results["documents"][0][i],
                        "source": results["metadatas"][0][i].get("source", ""),
                        "score": round(score, 4),
                        "metadata": results["metadatas"][0][i],
                    }
                )
        return output

    def ask(
        self,
        question: str,
        collection: str = "default",
        model: str = "auto",
    ) -> str:
        results = self.search(question, collection, top_k=5, threshold=0.5)
        if not results:
            return "No relevant documents found. Ingest documents first with ai.knowledge.ingest()."

        context = "\n\n---\n\n".join(
            [f"Source: {r['source']}\n{r['content']}" for r in results[:5]]
        )

        import httpx

        model_name = model if model != "auto" else os.getenv("LORAI_MODEL", "phi3:mini")
        resp = httpx.post(
            "http://localhost:11434/v1/chat/completions",
            json={
                "model": model_name,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "Answer based on the context below. "
                            "Cite sources when possible.\n\nContext:\n" + context
                        ),
                    },
                    {"role": "user", "content": question},
                ],
            },
            timeout=60.0,
        )
        return resp.json()["choices"][0]["message"]["content"]

    def _chunk(self, text: str, size: int = 1000, overlap: int = 200) -> list[str]:
        chunks = []
        start = 0
        while start < len(text):
            chunks.append(text[start : start + size])
            start += size - overlap
        return chunks if chunks else [text]
