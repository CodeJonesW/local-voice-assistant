"""Simple in-memory retriever for RAG support."""

import os
import pickle
from typing import List, Tuple
from collections import Counter
import math


class Retriever:
    def __init__(self, db_path: str = "vector_store.pkl", chunk_size: int = 200):
        self.db_path = db_path
        self.chunk_size = chunk_size
        self.documents: List[str] = []
        self.vectors: List[Counter] = []
        self.cache = {}
        if os.path.exists(self.db_path):
            self.load()

    def load(self) -> None:
        with open(self.db_path, "rb") as f:
            data = pickle.load(f)
        self.documents = data.get("documents", [])
        self.vectors = [Counter(doc.split()) for doc in self.documents]
        self.cache = {}

    def save(self) -> None:
        with open(self.db_path, "wb") as f:
            pickle.dump({"documents": self.documents}, f)

    def _chunk_text(self, text: str) -> List[str]:
        words = text.split()
        chunks = []
        for i in range(0, len(words), self.chunk_size):
            chunk = " ".join(words[i:i + self.chunk_size])
            chunks.append(chunk)
        return chunks

    def add_files(self, paths: List[str]) -> None:
        for path in paths:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
            new_chunks = self._chunk_text(text)
            self.documents.extend(new_chunks)
            self.vectors.extend(Counter(c.split()) for c in new_chunks)
        self.save()

    @staticmethod
    def _cosine(a: Counter, b: Counter) -> float:
        terms = set(a) | set(b)
        dot = sum(a[t] * b[t] for t in terms)
        norm_a = math.sqrt(sum(v * v for v in a.values()))
        norm_b = math.sqrt(sum(v * v for v in b.values()))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        if query in self.cache:
            return self.cache[query]
        if not self.documents:
            return []
        q_vec = Counter(query.split())
        sims = [self._cosine(q_vec, vec) for vec in self.vectors]
        ranked = sorted(enumerate(sims), key=lambda x: x[1], reverse=True)
        results = [self.documents[i] for i, s in ranked[:top_k] if s > 0]
        self.cache[query] = results
        return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Manage retrieval database")
    parser.add_argument("--add", nargs="*", help="Paths of text files to add")
    parser.add_argument("--db", default="vector_store.pkl", help="Path to DB file")
    args = parser.parse_args()

    retriever = Retriever(db_path=args.db)
    if args.add:
        retriever.add_files(args.add)
        print(f"Added {len(args.add)} file(s) to {args.db}")
    else:
        print("No action specified. Use --add to add documents.")
