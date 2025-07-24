"""Utility for building RAG prompts."""
from typing import List


def build_prompt(query: str, chunks: List[str]) -> str:
    """Append retrieved chunks to the query for RAG."""
    if not chunks:
        return query

    prompt_parts = [query, "\n\nRelevant information:"]
    for idx, chunk in enumerate(chunks, 1):
        prompt_parts.append(f"[{idx}] {chunk}")
    final_prompt = "\n".join(prompt_parts)

    print("--- Added Chunks ---")
    for idx, chunk in enumerate(chunks, 1):
        snippet = chunk[:60].replace("\n", " ")
        print(f"{idx}: {snippet}")

    return final_prompt
