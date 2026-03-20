"""Vector store and retrieval using ChromaDB."""

import os
import shutil
from pathlib import Path

import chromadb

from app.config import RetrieverConfig


class Retriever:
    def __init__(self, config: RetrieverConfig | None = None):
        self.config = config or RetrieverConfig()
        self.client = chromadb.PersistentClient(path=self.config.db_path)
        self.collection = self.client.get_or_create_collection(
            name="documents",
        )

    def _chunk_text(self, text: str) -> list[str]:
        """Split text into word-based chunks."""
        words = text.split()
        chunks = []
        for i in range(0, len(words), self.config.chunk_size):
            chunk = " ".join(words[i : i + self.config.chunk_size])
            if chunk.strip():
                chunks.append(chunk)
        return chunks

    def add_files(self, paths: list[str]) -> None:
        """Add text files to the vector store."""
        all_chunks = []
        all_ids = []
        existing_count = self.collection.count()

        for path in paths:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
            chunks = self._chunk_text(text)
            for chunk in chunks:
                all_chunks.append(chunk)
                all_ids.append(f"doc_{existing_count + len(all_ids)}")

        if all_chunks:
            self.collection.add(documents=all_chunks, ids=all_ids)

    def add_folder(self, folder: str | None = None,
                   processed_folder: str | None = None) -> None:
        """Add all files from a folder and move them to processed_folder."""
        folder = folder or self.config.train_folder
        processed_folder = processed_folder or self.config.processed_folder

        if not os.path.isdir(folder):
            return

        paths = [
            os.path.join(folder, name)
            for name in os.listdir(folder)
            if os.path.isfile(os.path.join(folder, name))
        ]
        if not paths:
            return

        self.add_files(paths)
        os.makedirs(processed_folder, exist_ok=True)
        for path in paths:
            shutil.move(path, os.path.join(processed_folder, os.path.basename(path)))

    def retrieve(self, query: str, n_results: int | None = None) -> list[str]:
        """Retrieve the most relevant chunks for a query."""
        n_results = n_results or self.config.n_results
        if self.collection.count() == 0:
            return []
        results = self.collection.query(
            query_texts=[query],
            n_results=min(n_results, self.collection.count()),
        )
        return results["documents"][0] if results["documents"] else []
