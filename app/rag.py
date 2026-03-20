"""RAG prompt building."""


def build_prompt(query: str, chunks: list[str]) -> str:
    """Build a RAG-augmented prompt from query and retrieved chunks."""
    if not chunks:
        return query

    parts = [query, "\n\nRelevant information:"]
    for idx, chunk in enumerate(chunks, 1):
        parts.append(f"[{idx}] {chunk}")

    return "\n".join(parts)
